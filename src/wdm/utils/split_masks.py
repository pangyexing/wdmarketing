"""Shared train/valid/oot mask construction over the raw table.

Single source of truth for the split semantics both stages must agree on:
data.exclude_rows filtering, split.embargo_days purging, and scattering masks
computed on the surviving rows back to full-table length. Stage-1
(wdm.pipeline.stage1) and Stage-2 (wdm.model.dataset.build_dataset) both call
compute_split_masks, so "train" means the same set of rows in every fitted
statistic and in the final model.
"""
import logging

import numpy as np
import pandas as pd

from wdm.utils.time_utils import (
    split_by_yyyymmdd, split_psi_halves, split_stratified)

logger = logging.getLogger(__name__)


def exclude_mask(df, exclude_rows):
    """Boolean mask of rows KEPT after applying data.exclude_rows rules."""
    keep = np.ones(len(df), dtype=bool)
    for rule in exclude_rows or []:
        col = rule["column"]
        vals = [float(v) for v in rule["values"]]
        num = pd.to_numeric(df[col], errors="coerce")
        drop = num.isin(vals).values
        keep &= ~drop
        logger.info("exclude_rows: %s in %s drops %d rows", col, rule["values"],
                    int(drop.sum()))
    return keep


def scatter_masks(included, sub_masks):
    """Scatter masks computed on df[included] back to full-table length.

    Excluded rows are False in every returned mask.
    """
    included = np.asarray(included, dtype=bool)
    if included.all():
        return tuple(np.asarray(m, dtype=bool) for m in sub_masks)
    inc_idx = np.where(included)[0]
    out = []
    for sm in sub_masks:
        full = np.zeros(included.size, dtype=bool)
        full[inc_idx[np.asarray(sm, dtype=bool)]] = True
        out.append(full)
    return tuple(out)


def compute_split_masks(df, cfg):
    """(train, valid, oot, included) boolean masks over the raw table.

    Rows dropped by data.exclude_rows are False in all four masks; the split
    is computed on the surviving rows only. Time splits purge
    split.embargo_days after each cut day.
    """
    data_cfg = cfg["data"]
    # No fallback defaults here: training.split / random_seed defaults live
    # only in configs/global.yaml (config._validate asserts presence).
    split_cfg = cfg["training"]["split"]
    strategy = split_cfg["strategy"]
    ratios = list(split_cfg["ratios"])
    seed = int(cfg["training"]["random_seed"])
    exclude_rows = data_cfg.get("exclude_rows") or []
    included = exclude_mask(df, exclude_rows)
    sub = df[included].reset_index(drop=True) if exclude_rows else df
    time_col = data_cfg.get("time_column")
    if strategy == "time":
        if not time_col:
            raise ValueError("split.strategy='time' requires data.time_column")
        sub_masks = split_by_yyyymmdd(
            sub[time_col], ratios,
            embargo_days=int(split_cfg.get("embargo_days", 0) or 0))
    elif strategy == "stratified":
        if time_col:
            logger.warning("time_column is configured but split.strategy="
                           "'stratified' — using stratified anyway")
        sub_masks = split_stratified(sub[data_cfg["label_column"]].values,
                                     ratios, seed=seed)
    else:
        raise ValueError("Unknown split strategy: {0}".format(strategy))
    m_tr, m_va, m_oot = scatter_masks(included, sub_masks)
    return m_tr, m_va, m_oot, included


def psi_partition_masks(cfg, meta_df, masks=None):
    """(mask_expected, mask_actual, has_time) per analysis.psi_partition.

    The single implementation of the selection-PSI partition, shared by
    Stage-1 and the per-feature plots so both show the same drift:
      train_halves  — earlier vs later half WITHIN the train split (default)
      halves        — earlier vs later half of the whole (included) window
      train_vs_rest — train vs valid+oot (deployment-facing, opt-in)
    Without a time column returns seeded random halves and has_time=False —
    the partition is noise and callers must treat the PSI as informational.
    masks: optional precomputed compute_split_masks output.
    """
    time_col = cfg["data"].get("time_column")
    if masks is None:
        masks = compute_split_masks(meta_df, cfg)
    m_tr, m_va, m_oot, included = masks
    psi_partition = str(cfg["analysis"]["psi_partition"]).lower()
    if time_col and time_col in meta_df.columns:
        if psi_partition == "train_vs_rest":
            return m_tr, np.asarray(m_va | m_oot, dtype=bool), True
        if psi_partition == "halves":
            m_e, m_a = scatter_masks(
                included, split_psi_halves(meta_df[time_col][included]))
            return m_e, m_a, True
        m_e, m_a = scatter_masks(
            m_tr, split_psi_halves(meta_df[time_col][m_tr]))
        return m_e, m_a, True
    rng = np.random.RandomState(int(cfg["training"]["random_seed"]))
    r = rng.rand(len(meta_df))
    return (r < 0.5) & included, (r >= 0.5) & included, False
