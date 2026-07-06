"""Stage-1 exit module: combines PSI / IV / Lift / correlation / family signals
into a single ranked feature report, writes CSVs + index.html + v1_auto.txt.

Rank score:
    rank_score = z(iv) + z(lift_at_k) + z(gini)
               − psi_penalty_weight · z(psi)              # soft, configurable
               − 0.5 · 1[missing_rate > 0.5]
               − window_penalty(window, group)
               + probing_weight · z(gain_rank_pct)        # when probing enabled

Auto-keep rule (the feature passes into v1_auto.txt):
    family_kept AND group_kept
    AND (cluster_id is singleton OR is cluster's max-rank member)
    AND (psi_mode != 'hard' OR psi < psi_cutoff)
    AND missing_rate < missing_rate_max_for_window
    AND iv >= iv_min

PSI role is deliberately **soft by default** (psi_mode='soft'):
  - 'hard': psi >= psi_cutoff drops the feature outright. Legacy behavior.
  - 'soft' (default): high-PSI only penalizes rank_score; features stay in
    the pool. Tree models often extract conditional signal from drifted
    features (rank relations can survive mean shifts).
  - 'off': PSI is informational only, no effect on selection.

missing_rate_max_for_window: short-window features (analysis.short_windows,
default 7d/30d) get the softer analysis.short_window_missing_rate_max cap
(default 0.98) instead of the global missing_rate_max, since "business didn't
happen in the window" ≠ "data quality".
"""
import datetime
import hashlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from wdm.analysis.correlation import cluster_correlated, cluster_id_per_feature
from wdm.analysis.family import (
    apply_group_correlation,
    build_families_summary,
    build_semantic_groups_summary,
    discover_derivation_candidates,
    effective_family_policy,
    parse_families,
    parse_semantic_groups,
    rank_within_family,
    rank_within_semantic_group,
)
from wdm.utils.paths import (
    analysis_dir, ensure_dirs, inject_cn_column,
    load_column_mapping, report_dir, selected_features_dir,
)
from wdm.utils.progress import StageProgress

logger = logging.getLogger(__name__)


def _zscore(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s.fillna(mu) - mu) / sigma


def _build_ranked_report(iv_df, psi_df, lift_df, missing_df, family_df, semantic_df,
                         cluster_map, cfg, probing_df=None):
    """Merge all per-feature tables on 'feature' into one.

    probing_df: optional DataFrame with columns (feature, gain, weight, cover,
    gain_rank_pct, ...) from Stage-1 probing model. Left-joined when present.
    """
    df = iv_df.merge(psi_df, on="feature", how="outer")
    df = df.merge(lift_df, on="feature", how="outer")
    df = df.merge(missing_df, on="feature", how="outer")
    df = df.merge(family_df, on="feature", how="left")
    df = df.merge(semantic_df.drop(columns=["group_description"], errors="ignore"),
                  on="feature", how="left")
    df["corr_cluster"] = df["feature"].map(cluster_map).fillna(-1).astype(int)

    fillna_map = {
        "iv": 0.0, "psi": 0.0, "lift_at_k": 1.0, "gini": 0.0,
        "concentration": 0.0, "missing_rate": 0.0, "n_unique": 0,
    }

    if probing_df is not None and len(probing_df):
        keep = [c for c in ("feature", "gain", "weight", "cover",
                            "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
                            "coverage")
                if c in probing_df.columns]
        pr = probing_df[keep].rename(columns={
            "gain": "probe_gain", "weight": "probe_weight", "cover": "probe_cover"})
        df = df.merge(pr, on="feature", how="left")
        for c in ("probe_gain", "probe_weight", "probe_cover",
                  "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
                  "coverage"):
            if c in df.columns:
                fillna_map[c] = 0.0

    df = df.fillna(fillna_map)
    return df


def _resolve_psi_knobs(cfg):
    """Pull psi_mode / psi_penalty_weight with sane fallbacks.

    psi_mode ∈ {"soft", "hard", "off"}. Historically PSI was a hard filter
    (implicit 'hard'). We default to 'soft' now — high PSI only dampens
    rank_score, it does not drop the feature. Products that really want the
    old behavior can set analysis.psi_mode: hard.
    """
    ana = cfg.get("analysis", {}) or {}
    mode = str(ana.get("psi_mode", "soft")).lower()
    if mode not in ("soft", "hard", "off"):
        mode = "soft"
    weight = float(ana.get("psi_penalty_weight", 0.25))
    return mode, weight


def _apply_hard_filters(df, cfg):
    miss_max = float(cfg["analysis"]["missing_rate_max"])
    iv_min = float(cfg["analysis"]["iv_min"])
    psi_cutoff = float(cfg["analysis"]["psi_cutoff"])
    psi_mode, _ = _resolve_psi_knobs(cfg)
    lift_keep_min = cfg["analysis"].get("lift_keep_min")
    lift_keep_min = float(lift_keep_min) if lift_keep_min is not None else None
    short_windows = set(cfg["analysis"].get("short_windows") or ["7d", "30d"])
    short_cap = float(cfg["analysis"].get("short_window_missing_rate_max", 0.98))

    if "window" in df.columns:
        is_short = df["window"].isin(short_windows).values
    else:
        is_short = np.zeros(len(df), dtype=bool)
    mr_cap = np.where(is_short, max(miss_max, short_cap), miss_max)

    # NaN comparisons are False, matching the legacy per-row behavior.
    with np.errstate(invalid="ignore"):
        constant = df["n_unique"].values <= 1
        high_missing = df["missing_rate"].values > mr_cap
        low_iv = df["iv"].values < iv_min
        if lift_keep_min is not None:
            # Positive-oriented soft gate: keep a weak-IV feature if it still
            # ranks positives well (lift_at_k >= lift_keep_min).
            if "lift_at_k" in df.columns:
                lift_vals = df["lift_at_k"].astype(float).values
            else:
                lift_vals = np.zeros(len(df), dtype=np.float64)
            low_iv = low_iv & (lift_vals < lift_keep_min)
        # PSI only hard-drops in psi_mode='hard'; soft/off keep the feature
        # (rank_score penalty / informational flag handle it downstream).
        if psi_mode == "hard":
            high_psi = df["psi"].values >= psi_cutoff
        else:
            high_psi = np.zeros(len(df), dtype=bool)

    reasons = []
    for c, hm, li, hp in zip(constant, high_missing, low_iv, high_psi):
        drop = []
        if c:
            drop.append("constant")
        if hm:
            drop.append("high_missing")
        if li:
            drop.append("low_iv")
        if hp:
            drop.append("high_psi")
        reasons.append(";".join(drop))
    df["_hard_drop"] = [bool(r) for r in reasons]
    df["_hard_drop_reason"] = reasons
    # Informational flag: annotate high-PSI features even in soft/off mode
    # so the report still shows drift risk without dropping the feature.
    df["psi_over_cutoff"] = (df["psi"] >= psi_cutoff).astype(bool)
    return df


def _penalty_table_for(policy, cfg):
    """Resolve window_penalty_table for a given effective policy; fall back to linear."""
    table = policy.get("window_penalty_table") or {}
    table = {str(k): float(v) for k, v in table.items()}
    if not table:
        order = list((cfg.get("feature_groups") or {}).get("window_order") or [])
        n = max(len(order), 1)
        table = {w: (i / n) * 0.3 for i, w in enumerate(order)}
    return table


def _row_penalty_contribution(df, cfg):
    """Per-row `gamma * penalty(window)` resolved from each row's semantic_group.

    When a semantic_group declares its own family_policy, that row uses the
    group's gamma and penalty_table; otherwise falls back to the global policy.
    Rows without a canonical string window contribute 0.
    """
    default_policy = (cfg.get("feature_groups") or {}).get("family_policy") or {}
    # Pre-resolve per semantic_group to avoid repeated dict merges.
    group_policies: Dict[Optional[str], Dict[str, Any]] = {None: default_policy}
    seen_groups = df.get("semantic_group")
    if seen_groups is not None:
        for g in seen_groups.dropna().unique():
            group_policies[str(g)] = effective_family_policy(str(g), cfg)
    group_gamma = {g: float(p.get("window_penalty_gamma", 0.0))
                   for g, p in group_policies.items()}
    group_table = {g: _penalty_table_for(p, cfg) for g, p in group_policies.items()}

    def _score(row):
        w = row.get("window")
        if not isinstance(w, str):
            return 0.0
        g = row.get("semantic_group")
        key = str(g) if isinstance(g, str) else None
        tbl = group_table.get(key, group_table[None])
        gamma = group_gamma.get(key, group_gamma[None])
        return gamma * float(tbl.get(w, 0.0))

    return df.apply(_score, axis=1).astype(float)


def _rank_and_auto_keep(df, cfg):
    df = df.copy()
    psi_mode, psi_weight = _resolve_psi_knobs(cfg)
    w = (cfg.get("analysis") or {}).get("rank_weights") or {}
    w_iv = float(w.get("iv", 1.0))
    w_lift = float(w.get("lift", 1.0))
    w_gini = float(w.get("gini", 1.0))
    w_conc = float(w.get("concentration", 0.0))
    w_miss = float(w.get("missing_penalty", 0.5))
    miss_thr = float(w.get("missing_penalty_threshold", 0.5))
    # PSI contribution to rank_score:
    #   - 'off'  : zero weight — PSI is informational only
    #   - else   : rank_weights.psi when explicitly configured, otherwise
    #              analysis.psi_penalty_weight (default 0.25 — meaningfully
    #              smaller than z(iv) / z(lift_at_k) / z(gini) each at 1.0)
    w_psi = float(w["psi"]) if "psi" in w else psi_weight
    effective_psi_weight = 0.0 if psi_mode == "off" else w_psi
    conc_term = (w_conc * _zscore(df["concentration"])
                 if "concentration" in df.columns else 0.0)
    df["rank_score"] = (
        w_iv * _zscore(df["iv"])
        + w_lift * _zscore(df["lift_at_k"])
        + w_gini * _zscore(df["gini"])
        + conc_term
        - effective_psi_weight * _zscore(df["psi"])
        - w_miss * (df["missing_rate"] > miss_thr).astype(float)
        - _row_penalty_contribution(df, cfg)
    )

    # Probing model contribution: add when Stage 1 probing wrote gain_rank_pct.
    # Weight is config-driven (analysis.probing.weight_in_rank_score, default 0.25).
    probing_cfg = (cfg.get("analysis") or {}).get("probing") or {}
    w_probe = float(probing_cfg.get("weight_in_rank_score", 0.25))
    if "gain_rank_pct" in df.columns and w_probe > 0:
        # gain_rank_pct is already in [0,1]; z-score makes it commensurate with
        # the other z-scored terms.
        df["rank_score"] = df["rank_score"] + w_probe * _zscore(df["gain_rank_pct"])

        # Coverage-stratified gain rank: features compete for "high gain" only
        # against peers of similar support. Raw gain/split-count is biased
        # toward dense features (more non-missing rows → more splits), so a
        # sparse feature with real conditional signal can be unfairly parked
        # in "noise" by the global ranking. Binning by coverage quintile and
        # ranking gain within each bin neutralizes that bias.
        if "coverage" in df.columns:
            try:
                cov_bin = pd.qcut(df["coverage"], q=5, labels=False,
                                   duplicates="drop")
            except ValueError:
                # Degenerate coverage (all equal) — fall back to a single bin.
                cov_bin = pd.Series(0, index=df.index)
            df["gain_rank_pct_by_coverage"] = (
                df.groupby(cov_bin)["gain_rank_pct"]
                  .rank(pct=True, method="average")
                  .fillna(0.0)
            )
            gain_high = df["gain_rank_pct_by_coverage"] > 0.7
        else:
            gain_high = df["gain_rank_pct"] > 0.7

        # Quadrant labels — expose what probing adds vs what IV already said.
        iv_high = df["iv"].rank(pct=True) > 0.7
        df["discover"] = (~iv_high) & gain_high
        df["stable"]   = iv_high & gain_high
        df["interp"]   = iv_high & (~gain_high)
        df["noise"]    = (~iv_high) & (~gain_high)

    # Within each correlation cluster, only the top-score survivor passes.
    # Winner pick keeps the legacy stable sort so rank_score ties resolve to
    # the member appearing first in the frame.
    cluster_winners = set()
    winner_by_cid = {}
    for cid, block in df.groupby("corr_cluster"):
        if cid == -1 or len(block) <= 1:
            cluster_winners.update(block["feature"].tolist())
            continue
        best = block.sort_values("rank_score", ascending=False).iloc[0]["feature"]
        cluster_winners.add(best)
        winner_by_cid[cid] = best

    n = len(df)
    if "family_kept" in df.columns:
        # bool(NaN) is True in the legacy row loop → fillna(True).
        fam_dropped = ~df["family_kept"].fillna(True).astype(bool).values
    else:
        fam_dropped = np.zeros(n, dtype=bool)
    if "group_kept" in df.columns:
        grp_dropped = ~df["group_kept"].fillna(True).astype(bool).values
    else:
        grp_dropped = np.zeros(n, dtype=bool)
    not_winner = (~df["feature"].isin(cluster_winners)).values
    hard_reasons = (df["_hard_drop_reason"].tolist()
                    if "_hard_drop_reason" in df.columns else [None] * n)
    cids = df["corr_cluster"].tolist()

    auto_keep = []
    drop_reason = []
    for k, feat_reason in enumerate(hard_reasons):
        reasons = []
        if feat_reason:
            reasons.append(feat_reason)
        if fam_dropped[k]:
            reasons.append("family_dropped_by_policy")
        if grp_dropped[k]:
            reasons.append("group_dropped_by_policy")
        if not_winner[k]:
            # A non-winner always belongs to a multi-member cluster, which
            # always has a recorded winner.
            reasons.append("corr_dup_of:{0}".format(winner_by_cid.get(cids[k], "?")))
        auto_keep.append(not reasons)
        drop_reason.append(";".join(reasons))
    df["auto_keep"] = auto_keep
    df["drop_reason"] = drop_reason
    return df.drop(columns=["_hard_drop", "_hard_drop_reason"], errors="ignore")


def _apply_column_ordering(df, mapping):
    """Put feature,feature_cn first; append the rest in a stable order."""
    df = inject_cn_column(df, mapping, feature_col="feature", cn_col="feature_cn")
    preferred = [
        "feature", "feature_cn",
        "family_base", "window", "pattern_id", "semantic_group",
        "dtype", "n_total", "n_unique", "missing_rate",
        "iv", "monotonic", "missing_n", "missing_woe",
        "psi", "flag", "psi_over_cutoff",
        "lift_at_k", "gini", "concentration",
        "probe_gain", "probe_weight", "probe_cover", "coverage",
        "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
        "gain_rank_pct_by_coverage",
        "corr_cluster",
        "family_size", "in_family_rank", "family_kept",
        "group_size", "in_group_rank", "group_kept",
        "rank_score", "discover", "stable", "interp", "noise",
        "auto_keep", "drop_reason",
    ]
    cols_present = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols_present]
    return df[cols_present + rest]


def _write_index_html(report_dir_path, mapping):
    """A small, dependency-free HTML summary that loads the CSVs via JS fetch.

    Uses a tiny embedded sort helper so tables are sortable by clicking headers.
    Works fully offline.
    """
    report_dir_path = Path(report_dir_path)
    csvs = sorted([p.name for p in report_dir_path.glob("*.csv")])
    summary_first = ["summary.csv", "families.csv",
                     "family_derivation_candidates.csv", "semantic_groups.csv",
                     "probing_importance.csv",
                     "iv_woe.csv", "psi.csv", "lift.csv", "missing.csv",
                     "correlation_edges.csv"]
    ordered = [n for n in summary_first if n in csvs]
    ordered += [n for n in csvs if n not in ordered]

    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN"><head>',
        '<meta charset="utf-8">',
        '<title>Feature Report</title>',
        '<style>',
        'body{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;margin:18px;color:#222}',
        'h1{margin-top:12px}h2{margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}',
        'table{border-collapse:collapse;font-size:12px;margin:8px 0 24px 0;max-width:100%}',
        'th{background:#f6f6f6;position:sticky;top:0;cursor:pointer;user-select:none;padding:4px 8px;text-align:left;border:1px solid #ccc}',
        'th:hover{background:#eee}',
        'td{padding:3px 8px;border:1px solid #e4e4e4}',
        'tr:nth-child(even){background:#fafafa}',
        '.hint{color:#888;font-size:12px}',
        '.nav a{margin-right:12px}',
        '</style></head><body>',
        '<h1>Feature Analysis Report</h1>',
        '<p class="hint">Click a column header to sort. All data is loaded from the sibling CSV files — this page works offline.</p>',
        '<div class="nav">' + ''.join(
            ['<a href="#{0}">{0}</a>'.format(n) for n in ordered]) + '</div>',
    ]
    for name in ordered:
        p = report_dir_path / name
        try:
            df = pd.read_csv(p)
        except Exception as e:
            html_parts.append('<h2 id="{0}">{0}</h2><p>Error loading: {1}</p>'.format(name, e))
            continue
        # Round numeric for readability (don't mutate the CSV, only the HTML)
        disp = df.copy()
        for c in disp.select_dtypes(include=[np.floating]).columns:
            disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else "{0:.4f}".format(v))
        html_parts.append('<h2 id="{0}">{0}</h2>'.format(name))
        html_parts.append('<p class="hint">rows: {0}, cols: {1}</p>'.format(len(df), len(df.columns)))
        html_parts.append(disp.to_html(index=False, classes="sortable", escape=False,
                                       table_id="t-{0}".format(name)))

    html_parts.append("""
<script>
// Minimal sortable-table JS (no deps). Click a header to sort asc; click again for desc.
document.querySelectorAll('table').forEach(tbl => {
  const headers = tbl.querySelectorAll('th');
  headers.forEach((th, col) => {
    let asc = true;
    th.addEventListener('click', () => {
      const tbody = tbl.querySelector('tbody') || tbl;
      const rows = Array.from(tbl.querySelectorAll('tr')).slice(1);
      rows.sort((a, b) => {
        const av = a.children[col].innerText.trim();
        const bv = b.children[col].innerText.trim();
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv, 'zh-CN') : bv.localeCompare(av, 'zh-CN');
      });
      asc = !asc;
      rows.forEach(r => tbl.appendChild(r));
    });
  });
});
</script>
""")
    html_parts.append('</body></html>')
    out_path = report_dir_path / "index.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return out_path


def _write_auto_features_txt(df, out_path, top_n, report_hash, parent=None,
                             source="analysis/selector.py"):
    auto = df[df["auto_keep"] == True].copy()
    auto = auto.sort_values("rank_score", ascending=False).head(top_n)
    header = [
        "# Auto-generated feature list",
        "# created_at: {0}".format(datetime.datetime.now().isoformat(timespec="seconds")),
        "# parent: {0}".format(parent or "null"),
        "# report_hash: {0}".format(report_hash),
        "# feature_count: {0}".format(len(auto)),
        "# source: {0}".format(source),
        "",
    ]
    lines = [row["feature"] for _, row in auto.iterrows()]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")
    return out_path


def _load_probing_importance(cfg):
    """Return the probing importance DataFrame, or None if not present.

    The probing step writes probing_importance.csv into the report directory
    before run_stage1 touches selector output. We read it lazily so Stage 1
    still runs end-to-end when probing is disabled.
    """
    rdir = report_dir(cfg)
    p = rdir / "probing_importance.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s; ignoring probing signal.", p, e)
        return None
    logger.info("Loaded probing importance: %s (%d rows)", p, len(df))
    return df


def _report_hash(summary_csv_path):
    h = hashlib.sha1()
    with open(summary_csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _make_scan_cache_dir(cfg, scan_cache_cfg):
    """Create a run-private cache dir for the single-pass scan's .npy blocks.

    Default location is artifacts/<product>/analysis/scan_cache/scan_XXXX;
    io.scan_cache.dir overrides the base (absolute, or repo-root relative —
    e.g. a scratch disk for very wide datasets).
    """
    base = scan_cache_cfg.get("dir")
    if base:
        base_dir = Path(base)
        if not base_dir.is_absolute():
            base_dir = Path(cfg["_repo_root"]) / base_dir
    else:
        base_dir = analysis_dir(cfg) / "scan_cache"
    ensure_dirs(base_dir)
    return Path(tempfile.mkdtemp(prefix="scan_", dir=str(base_dir)))


def _cleanup_scan_cache(cache_dir, scan_cache_cfg):
    if cache_dir is None:
        return
    if bool(scan_cache_cfg.get("keep", False)):
        logger.info("Keeping scan cache (io.scan_cache.keep=true): %s", cache_dir)
        return
    shutil.rmtree(str(cache_dir), ignore_errors=True)


def run_stage1(cfg):
    """Run the full Stage-1 pipeline and materialize all artifacts.

    Returns a dict summarizing what was written.
    """
    from wdm.analysis.correlation import (
        compute_correlation_edges, compute_edges_from_cache)
    from wdm.analysis.feature_scan import run_feature_scan
    from wdm.io.column_scanner import scan_columns
    from wdm.preprocess.missing import build_missing_spec, get_spec
    from wdm.utils.time_utils import split_psi_halves

    prog = StageProgress("Stage 1", total=5)

    with prog.step("scan columns + load label/time"):
        idx = scan_columns(cfg)
        features = idx["features"]
        label_col = idx["label_column"]
        time_col = idx["time_column"]
        path = idx["data_path"]

        spec_map = build_missing_spec(cfg)
        chunk_size = int(cfg["io"]["column_chunk_size"])

        # Only the label and time columns are needed up-front; feature columns
        # stream through the single-pass chunked scan below.
        meta_cols = [label_col]
        if time_col and time_col != label_col:
            meta_cols.append(time_col)
        meta_df = pd.read_csv(path, usecols=meta_cols)
        y = meta_df[label_col]

        # PSI expected/actual masks — by time when available, else random
        # halves as a placeholder.
        if time_col and time_col in meta_df.columns:
            m_e, m_a = split_psi_halves(meta_df[time_col])
        else:
            rng = np.random.RandomState(cfg["training"]["random_seed"])
            r = rng.rand(len(meta_df))
            m_e, m_a = (r < 0.5), (r >= 0.5)
            logger.warning("No time_column configured — PSI computed on random halves "
                           "(useful only as a smoke check).")

    n_chunks = (len(features) + chunk_size - 1) // chunk_size
    logger.info("Stage 1 starting: %d features × %d rows (%d column chunks)",
                len(features), len(meta_df), n_chunks)

    scan_cache_cfg = (cfg.get("io") or {}).get("scan_cache") or {}
    cache_dir = None
    try:
        # IV / missing / lift / PSI + correlation Pass-1 stats in ONE pass
        # over the CSV; blocks cached as .npy for Pass-2 unless disabled.
        with prog.step("single-pass scan (IV/missing/lift/PSI + corr stats)"):
            if bool(scan_cache_cfg.get("enabled", True)):
                cache_dir = _make_scan_cache_dir(cfg, scan_cache_cfg)
            scan = run_feature_scan(path, features, y, m_e, m_a,
                                    spec_map, get_spec, cfg,
                                    cache_dir=cache_dir)
            iv_df = scan.iv_df
            bin_specs = scan.bin_specs
            miss_df = scan.miss_df
            lift_df = scan.lift_df
            psi_df = scan.psi_df

        # Correlation Pass-2 (global cutoff)
        with prog.step("correlation pass-2"):
            corr_threshold = float(cfg["analysis"]["corr_cutoff"])
            min_overlap = float(cfg["analysis"].get("corr_min_overlap_frac", 0.10))
            if cache_dir is not None:
                edges = compute_edges_from_cache(
                    features, scan.blocks, cache_dir,
                    scan.col_count, scan.col_sum, scan.col_sum_sq, scan.n_rows,
                    threshold=corr_threshold, min_overlap_frac=min_overlap,
                    mmap=bool(scan_cache_cfg.get("mmap", True)))
            else:
                # Fallback path: re-reads the CSV per block pair — much slower,
                # but needs no scratch disk.
                edges = compute_correlation_edges(
                    features, path, always=[label_col],
                    spec_map=spec_map, get_spec_fn=get_spec,
                    chunk_size=chunk_size, threshold=corr_threshold,
                    min_overlap_frac=min_overlap)
            if edges.empty:
                logger.info("No feature pairs with |r| >= %.2f — correlation_edges.csv will "
                            "be empty (this is expected when no features are highly "
                            "collinear; lower analysis.corr_cutoff in the product config "
                            "to inspect weaker correlations).", corr_threshold)
    finally:
        _cleanup_scan_cache(cache_dir, scan_cache_cfg)

    with prog.step("family/group ranking + auto-keep"):
        # Family + semantic groups
        family_df = parse_families(features, cfg)
        semantic_df, missing_by_group = parse_semantic_groups(features, cfg)

        # Probing importance — Stage-1 probing model writes this sidecar; pick
        # it up if it exists so rank_score can incorporate the gain-based signal.
        probing_df = _load_probing_importance(cfg)

        # Merge into a report base; rank within family/group; cluster
        base = _build_ranked_report(iv_df, psi_df, lift_df, miss_df,
                                    family_df, semantic_df, cluster_map={}, cfg=cfg,
                                    probing_df=probing_df)
        base = rank_within_family(base, cfg)
        base = rank_within_semantic_group(base, cfg)

        # Tighten correlation edges inside family/semantic groups
        edges_tight = apply_group_correlation(edges, family_df, semantic_df, cfg)
        clusters = cluster_correlated(edges_tight, features)
        cmap = cluster_id_per_feature(clusters)
        base["corr_cluster"] = base["feature"].map(cmap).fillna(-1).astype(int)

        # Hard filters → rank_score → auto_keep → drop_reason
        base = _apply_hard_filters(base, cfg)
        base = _rank_and_auto_keep(base, cfg)

    with prog.step("write report artifacts"):
        # Column ordering + 中文列名
        mapping = load_column_mapping(cfg)
        summary = _apply_column_ordering(base, mapping)

        rdir = report_dir(cfg)
        ensure_dirs(rdir)

        summary_path = rdir / "summary.csv"
        summary.to_csv(summary_path, index=False)

        # Per-signal CSVs, each left-joined with Chinese names
        iv_out = inject_cn_column(iv_df, mapping)
        iv_out.to_csv(rdir / "iv_woe.csv", index=False)

        psi_out = inject_cn_column(psi_df, mapping)
        psi_out.to_csv(rdir / "psi.csv", index=False)

        lift_out = inject_cn_column(lift_df, mapping)
        lift_out.to_csv(rdir / "lift.csv", index=False)

        miss_out = inject_cn_column(miss_df, mapping)
        miss_out.to_csv(rdir / "missing.csv", index=False)

        edges_out = edges.copy() if edges is not None else pd.DataFrame()
        if not edges_out.empty:
            # Map Chinese on both columns
            edges_out["f1_cn"] = edges_out["f1"].map(lambda x: mapping.get(x, x))
            edges_out["f2_cn"] = edges_out["f2"].map(lambda x: mapping.get(x, x))
            edges_out = edges_out[["f1", "f1_cn", "f2", "f2_cn", "r", "n_pairs", "low_overlap"]]
        edges_out.to_csv(rdir / "correlation_edges.csv", index=False)

        fam_summary = build_families_summary(summary, cfg)
        fam_summary.to_csv(rdir / "families.csv", index=False)

        sem_summary = build_semantic_groups_summary(summary, missing_by_group, cfg)
        sem_summary.to_csv(rdir / "semantic_groups.csv", index=False)

        cand_df = discover_derivation_candidates(summary, cfg)
        cand_df.to_csv(rdir / "family_derivation_candidates.csv", index=False)
        logger.info("Derivation candidates: %d multi-window families flagged → %s",
                    len(cand_df), rdir / "family_derivation_candidates.csv")

        # Optional XLSX — only if openpyxl happens to be available
        try:
            import openpyxl  # noqa: F401
            xlsx_path = analysis_dir(cfg) / "feature_report.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
                summary.to_excel(w, sheet_name="Summary", index=False)
                iv_out.to_excel(w, sheet_name="IV_WOE", index=False)
                psi_out.to_excel(w, sheet_name="PSI", index=False)
                lift_out.to_excel(w, sheet_name="Lift", index=False)
                miss_out.to_excel(w, sheet_name="Missing", index=False)
                edges_out.to_excel(w, sheet_name="Correlation_Edges", index=False)
                fam_summary.to_excel(w, sheet_name="Families", index=False)
                sem_summary.to_excel(w, sheet_name="SemanticGroups", index=False)
                cand_df.to_excel(w, sheet_name="DerivationCandidates", index=False)
            logger.info("Wrote optional XLSX: %s", xlsx_path)
        except Exception as e:
            logger.info("Skipped XLSX generation: %s", e)

        _write_index_html(rdir, mapping)

        # v1_auto.txt — analysis.stage1_top_n explicitly overrides the size;
        # otherwise, when the two-stage funnel is enabled (stage2_candidate_count
        # > final_feature_count), Stage-1 writes the wider candidate pool here
        # and Stage-2 narrows it via exploratory XGB ranking. Otherwise this
        # file is already the final feature set.
        sf_dir = selected_features_dir(cfg)
        ensure_dirs(sf_dir)
        auto_path = sf_dir / "v1_auto.txt"
        training_cfg = cfg["training"]
        final_n = int(training_cfg["final_feature_count"])
        candidate_n = training_cfg.get("stage2_candidate_count")
        funnel_n = int(candidate_n) if candidate_n and int(candidate_n) > final_n else final_n
        top_n = int(cfg["analysis"].get("stage1_top_n") or funnel_n)
        rh = _report_hash(summary_path)
        _write_auto_features_txt(summary, auto_path, top_n=top_n, report_hash=rh)

    prog.finish()
    logger.info("Stage 1 done. Report: %s  Auto features: %s", rdir, auto_path)

    return {
        "summary_path": str(summary_path),
        "report_dir": str(rdir),
        "auto_features": str(auto_path),
        "n_features": len(features),
        "n_auto_kept": int(summary["auto_keep"].sum()),
        "bin_specs": bin_specs,  # kept in memory for downstream plotting
    }
