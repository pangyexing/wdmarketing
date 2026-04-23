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

missing_rate_max_for_window: short-window features (7d/30d) get 0.98 cap
instead of the global 0.95, since "business didn't happen" ≠ "data quality".
"""
import datetime
import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)

_SHORT_WINDOWS = {"7d", "30d"}


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
        "missing_rate": 0.0, "n_unique": 0,
    }

    if probing_df is not None and len(probing_df):
        keep = [c for c in ("feature", "gain", "weight", "cover",
                            "gain_rank_pct", "weight_rank_pct", "cover_rank_pct")
                if c in probing_df.columns]
        pr = probing_df[keep].rename(columns={
            "gain": "probe_gain", "weight": "probe_weight", "cover": "probe_cover"})
        df = df.merge(pr, on="feature", how="left")
        for c in ("probe_gain", "probe_weight", "probe_cover",
                  "gain_rank_pct", "weight_rank_pct", "cover_rank_pct"):
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

    reasons = []
    for _, row in df.iterrows():
        drop = []
        mr_cap = miss_max
        if row.get("window") in _SHORT_WINDOWS:
            mr_cap = max(mr_cap, 0.98)
        if row["n_unique"] <= 1:
            drop.append("constant")
        if row["missing_rate"] > mr_cap:
            drop.append("high_missing")
        if row["iv"] < iv_min:
            drop.append("low_iv")
        if psi_mode == "hard" and row["psi"] >= psi_cutoff:
            drop.append("high_psi")
        reasons.append(";".join(drop))
    df = df.copy()
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
    # PSI contribution to rank_score:
    #   - 'off'     : zero weight — PSI is informational only
    #   - 'soft'    : configurable weight (default 0.25 — meaningfully
    #                 smaller than z(iv) / z(lift_at_k) / z(gini) each at 1.0)
    #   - 'hard'    : same as soft; features past the cutoff were already
    #                 dropped in _apply_hard_filters
    effective_psi_weight = 0.0 if psi_mode == "off" else psi_weight
    df["rank_score"] = (
        _zscore(df["iv"])
        + _zscore(df["lift_at_k"])
        + _zscore(df["gini"])
        - effective_psi_weight * _zscore(df["psi"])
        - 0.5 * (df["missing_rate"] > 0.5).astype(float)
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
        # Quadrant labels — expose what probing adds vs what IV already said.
        iv_high = df["iv"].rank(pct=True) > 0.7
        gain_high = df["gain_rank_pct"] > 0.7
        df["discover"] = (~iv_high) & gain_high
        df["stable"]   = iv_high & gain_high
        df["interp"]   = iv_high & (~gain_high)
        df["noise"]    = (~iv_high) & (~gain_high)

    # Within each correlation cluster, only the top-score survivor passes.
    cluster_winners = set()
    for cid, block in df.groupby("corr_cluster"):
        if cid == -1 or len(block) <= 1:
            cluster_winners.update(block["feature"].tolist())
            continue
        best = block.sort_values("rank_score", ascending=False).iloc[0]["feature"]
        cluster_winners.add(best)

    auto_keep = []
    drop_reason = []
    for _, row in df.iterrows():
        reasons = []
        if row.get("_hard_drop_reason"):
            reasons.append(row["_hard_drop_reason"])
        if not bool(row.get("family_kept", True)):
            reasons.append("family_dropped_by_policy")
        if not bool(row.get("group_kept", True)):
            reasons.append("group_dropped_by_policy")
        if row["feature"] not in cluster_winners:
            winner = "?"
            cid = row["corr_cluster"]
            if cid != -1:
                winners = df[(df["corr_cluster"] == cid) &
                             (df["feature"].isin(cluster_winners))]
                if len(winners):
                    winner = winners.iloc[0]["feature"]
            reasons.append("corr_dup_of:{0}".format(winner))
        keep = not reasons
        auto_keep.append(bool(keep))
        drop_reason.append(";".join([r for r in reasons if r]))
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
        "probe_gain", "probe_weight", "probe_cover",
        "gain_rank_pct", "weight_rank_pct", "cover_rank_pct",
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


def _write_auto_features_txt(df, out_path, top_n, report_hash, parent=None):
    auto = df[df["auto_keep"] == True].copy()
    auto = auto.sort_values("rank_score", ascending=False).head(top_n)
    header = [
        "# Auto-generated feature list",
        "# created_at: {0}".format(datetime.datetime.now().isoformat(timespec="seconds")),
        "# parent: {0}".format(parent or "null"),
        "# report_hash: {0}".format(report_hash),
        "# feature_count: {0}".format(len(auto)),
        "# source: analysis/selector.py",
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


def run_stage1(cfg):
    """Run the full Stage-1 pipeline and materialize all artifacts.

    Returns a dict summarizing what was written.
    """
    from wdm.analysis.correlation import compute_correlation_edges
    from wdm.analysis.iv_woe import compute_iv_table
    from wdm.analysis.lift import compute_feature_lift_table
    from wdm.analysis.missing import compute_missing_stats
    from wdm.analysis.psi import compute_psi_table_single_source
    from wdm.io.chunked_reader import iter_column_chunks
    from wdm.io.column_scanner import scan_columns
    from wdm.preprocess.missing import build_missing_spec, get_spec
    from wdm.utils.time_utils import split_psi_halves

    idx = scan_columns(cfg)
    features = idx["features"]
    label_col = idx["label_column"]
    time_col = idx["time_column"]
    path = idx["data_path"]

    spec_map = build_missing_spec(cfg)
    chunk_size = int(cfg["io"]["column_chunk_size"])

    full_df = pd.read_csv(path)
    y = full_df[label_col]

    logger.info("Stage 1 starting: %d features × %d rows", len(features), len(full_df))

    # IV / WOE
    iv_df, bin_specs = compute_iv_table(
        iter_column_chunks(path, features, always=[label_col], chunk_size=chunk_size),
        spec_map, y, features, cfg, get_spec)

    # Missing
    miss_df = compute_missing_stats(
        iter_column_chunks(path, features, always=[], chunk_size=chunk_size),
        spec_map, get_spec)

    # Lift
    lift_df = compute_feature_lift_table(
        iter_column_chunks(path, features, always=[label_col], chunk_size=chunk_size),
        spec_map, y, cfg, get_spec)

    # PSI — split by time if possible else random halves as placeholder
    if time_col and time_col in full_df.columns:
        m_e, m_a = split_psi_halves(full_df[time_col])
    else:
        rng = np.random.RandomState(cfg["training"]["random_seed"])
        r = rng.rand(len(full_df))
        m_e, m_a = (r < 0.5), (r >= 0.5)
        logger.warning("No time_column configured — PSI computed on random halves "
                       "(useful only as a smoke check).")
    psi_df = compute_psi_table_single_source(
        iter_column_chunks(path, features, always=[label_col], chunk_size=chunk_size),
        m_e, m_a, spec_map, cfg, get_spec)

    # Correlation (global cutoff)
    corr_threshold = float(cfg["analysis"]["corr_cutoff"])
    edges = compute_correlation_edges(
        features, path, always=[label_col],
        spec_map=spec_map, get_spec_fn=get_spec,
        chunk_size=chunk_size, threshold=corr_threshold)
    if edges.empty:
        logger.info("No feature pairs with |r| >= %.2f — correlation_edges.csv will "
                    "be empty (this is expected when no features are highly "
                    "collinear; lower analysis.corr_cutoff in the product config "
                    "to inspect weaker correlations).", corr_threshold)

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

    # Column ordering + 中文列名
    mapping = load_column_mapping(cfg)
    summary = _apply_column_ordering(base, mapping)

    # ---- Write artifacts ----
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

    # v1_auto.txt
    sf_dir = selected_features_dir(cfg)
    ensure_dirs(sf_dir)
    auto_path = sf_dir / "v1_auto.txt"
    top_n = int(cfg["training"]["final_feature_count"])
    rh = _report_hash(summary_path)
    _write_auto_features_txt(summary, auto_path, top_n=top_n, report_hash=rh)

    logger.info("Stage 1 done. Report: %s  Auto features: %s", rdir, auto_path)

    return {
        "summary_path": str(summary_path),
        "report_dir": str(rdir),
        "auto_features": str(auto_path),
        "n_features": len(features),
        "n_auto_kept": int(summary["auto_keep"].sum()),
        "bin_specs": bin_specs,  # kept in memory for downstream plotting
    }
