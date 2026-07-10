"""Stage-1 pipeline orchestration: run the single-pass feature scan,
correlation pass-2, family/group ranking and auto-keep, then materialize all
report artifacts (summary.csv, index.html, v1_auto.txt, ...).

Scoring and filtering logic (rank_score, hard filters, cluster winners) lives
in wdm.analysis.selector; this module owns the run order, the scan cache
lifecycle, and artifact writing.
"""
import datetime
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Pass-2 edge computation is called via the module (not bound at import time)
# so tests can monkeypatch wdm.analysis.correlation regardless of import order.
from wdm.analysis import correlation
from wdm.analysis.correlation import cluster_correlated, cluster_id_per_feature
from wdm.analysis.family import (
    apply_group_correlation,
    build_families_summary,
    build_semantic_groups_summary,
    parse_families,
    parse_semantic_groups,
    rank_within_family,
    rank_within_semantic_group,
)
from wdm.analysis.feature_scan import run_feature_scan
from wdm.analysis.selector import (
    apply_hard_filters,
    build_ranked_report,
    rank_and_auto_keep,
)
from wdm.io.column_scanner import scan_columns
from wdm.preprocess.missing import build_missing_spec, get_spec
from wdm.utils.paths import (
    analysis_dir, ensure_dirs, inject_cn_column,
    load_column_mapping, report_dir, selected_features_dir,
)
from wdm.utils.labels import validate_binary_label
from wdm.utils.progress import StageProgress
from wdm.utils.split_masks import compute_split_masks, psi_partition_masks

logger = logging.getLogger(__name__)




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
    summary_first = ["summary.csv", "families.csv", "semantic_groups.csv",
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


def write_auto_features_txt(df, out_path, top_n, report_hash, parent=None,
                            source="pipeline/stage1.py"):
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


def _probing_is_stale(cfg, meta_path):
    """True when probing_meta.json's cache fingerprint no longer matches the
    configured train CSV — i.e. probing_importance.csv was produced from
    different data and must not be merged into rank_score.
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Unreadable %s (%s) — treating probing output as stale.",
                       meta_path, e)
        return True
    fp = meta.get("cache_fingerprint") or {}
    if fp.get("csv_size_bytes") is None:
        return False  # older probing runs recorded no fingerprint; trust them
    csv_path = Path(cfg["_repo_root"]) / cfg["data"]["train_path"]
    if not csv_path.is_file():
        return False
    st = csv_path.stat()
    return (fp.get("csv_size_bytes") != st.st_size
            or abs(float(fp.get("csv_mtime", 0)) - st.st_mtime) > 1e-3)


def _load_probing_importance(cfg):
    """Return the probing importance DataFrame, or None if absent/stale.

    The probing step writes probing_importance.csv + probing_meta.json into
    the report directory before run_stage1 touches selector output. The meta
    fingerprint guards against a leftover CSV from an older run (different
    data / feature set) silently leaking into rank_score.
    """
    rdir = report_dir(cfg)
    p = rdir / "probing_importance.csv"
    if not p.is_file():
        return None
    meta_path = rdir / "probing_meta.json"
    if meta_path.is_file():
        if _probing_is_stale(cfg, meta_path):
            logger.warning(
                "probing_importance.csv at %s is STALE (source CSV changed "
                "since probing ran) — ignoring the probing signal. Re-run "
                "probing (scripts/run_analysis.py --probing) to refresh.", p)
            return None
    else:
        logger.warning("probing_meta.json missing next to %s — freshness "
                       "cannot be verified; using the file as-is.", p)
    try:
        df = pd.read_csv(p)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s; ignoring probing signal.", p, e)
        return None
    logger.info("Loaded probing importance: %s (%d rows)", p, len(df))
    return df


def report_hash(summary_csv_path):
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
    prog = StageProgress("Stage 1", total=5)

    with prog.step("scan columns + load label/time"):
        idx = scan_columns(cfg)
        features = idx["features"]
        label_col = idx["label_column"]
        time_col = idx["time_column"]
        path = idx["data_path"]

        spec_map = build_missing_spec(cfg)
        chunk_size = int(cfg["io"]["column_chunk_size"])

        # Only the label, time and exclude-rule columns are needed up-front;
        # feature columns stream through the single-pass chunked scan below.
        exclude_rows = cfg["data"].get("exclude_rows") or []
        meta_cols = [label_col]
        if time_col and time_col != label_col:
            meta_cols.append(time_col)
        for rule in exclude_rows:
            if rule["column"] not in meta_cols:
                meta_cols.append(rule["column"])
        meta_df = pd.read_csv(path, usecols=meta_cols)
        y = meta_df[label_col]

        # Train/valid/oot masks shared with Stage-2 (wdm.utils.split_masks):
        # exclude_rows filtering and embargo purging match build_dataset
        # exactly, so every fitted Stage-1 statistic sees the same "train"
        # rows the final model trains on.
        split_cfg = cfg["training"]["split"]
        m_tr, m_va, m_oot, included = compute_split_masks(meta_df, cfg)
        validate_binary_label(y[included], label_col)

        # Supervised statistics (IV/WOE bins, Lift@K, Gini) are fit on the
        # train split only, so valid/OOT labels never influence which
        # features are selected. analysis.supervised_stats_split: full
        # restores the legacy full-data behavior.
        sup_mode = str(cfg["analysis"]["supervised_stats_split"]).lower()
        if sup_mode == "train_only":
            supervised_mask = m_tr
            logger.info("Supervised Stage-1 stats (IV/Lift/Gini + bin edges) "
                        "fit on the train split only: %d/%d rows (split=%s).",
                        int(m_tr.sum()), len(meta_df),
                        split_cfg["strategy"])
        else:
            # "full" restores the legacy all-rows behavior for the labels,
            # but rows dropped by data.exclude_rows still never enter any
            # fitted statistic.
            supervised_mask = included if exclude_rows else None
            logger.info("analysis.supervised_stats_split=full — supervised "
                        "Stage-1 stats use ALL rows (valid/OOT labels "
                        "included in feature selection).")

        # Label-free selection statistics (missing rate → hard filter,
        # correlation → which cluster member survives de-duplication) follow
        # the same train-only discipline by default; `full` restores the
        # legacy all-rows behavior.
        unsup_mode = str(cfg["analysis"]["unsupervised_stats_split"]).lower()
        if unsup_mode == "train_only":
            unsupervised_mask = m_tr
            logger.info("Unsupervised Stage-1 stats (missing rate, "
                        "correlation) computed on the train split only: "
                        "%d/%d rows.", int(m_tr.sum()), len(meta_df))
        else:
            unsupervised_mask = included if exclude_rows else None
            logger.info("analysis.unsupervised_stats_split=full — missing "
                        "rate and correlation use ALL rows (valid/OOT "
                        "feature distributions influence selection).")

        # PSI expected/actual masks.
        #   train_halves (default): earlier vs later half WITHIN the train
        #                       split — a drift signal that can feed
        #                       rank_score without valid/OOT feature rows
        #                       influencing selection.
        #   halves            : earlier vs later half of the whole window
        #                       (legacy; includes valid/OOT rows).
        #   train_vs_rest     : train split vs valid+oot — the drift the
        #                       model actually faces at deployment. Explicit
        #                       opt-in: with psi_mode soft/hard this lets
        #                       valid/OOT feature distributions move
        #                       rank_score — deliberate, config-visible.
        # Without a time column any partition is noise: compute PSI on random
        # halves for the report, but force psi_mode='off' (local copy) so the
        # noise cannot move rank_score or drop features.
        psi_partition = str(cfg["analysis"]["psi_partition"]).lower()
        m_e, m_a, psi_has_time = psi_partition_masks(
            cfg, meta_df, masks=(m_tr, m_va, m_oot, included))
        if psi_has_time:
            if psi_partition == "train_vs_rest":
                logger.info("PSI partition: train split (n=%d) vs valid+oot "
                            "(n=%d) — deployment-facing drift; valid/OOT "
                            "rows DO enter the selection PSI.",
                            int(m_e.sum()), int(m_a.sum()))
            elif psi_partition == "train_halves":
                logger.info("PSI partition: train-split halves (%d vs %d "
                            "rows) — valid/OOT rows never enter the "
                            "selection PSI.", int(m_e.sum()), int(m_a.sum()))
        else:
            cfg = dict(cfg)
            cfg["analysis"] = dict(cfg["analysis"])
            cfg["analysis"]["psi_mode"] = "off"
            logger.warning(
                "No time_column configured — PSI is computed on RANDOM halves "
                "(pure noise) and reported for reference only; psi_mode forced "
                "to 'off' for this run so it cannot affect rank_score or "
                "filtering.")

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
                                    cache_dir=cache_dir,
                                    supervised_mask=supervised_mask,
                                    unsupervised_mask=unsupervised_mask)
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
                edges = correlation.compute_edges_from_cache(
                    features, scan.blocks, cache_dir,
                    scan.col_count, scan.col_sum, scan.col_sum_sq, scan.n_rows,
                    threshold=corr_threshold, min_overlap_frac=min_overlap,
                    mmap=bool(scan_cache_cfg.get("mmap", True)))
            else:
                # Fallback path: re-reads the CSV per block pair — needs no
                # scratch disk but the cost is quadratic in chunk count. On
                # wide tables that silently turns a minutes-long run into
                # hours, so it is a hard error unless explicitly allowed.
                n_pair_parses = n_chunks * (n_chunks + 1) // 2
                slow_corr_max = int(cfg["analysis"].get(
                    "slow_correlation_max_features", 500))
                if (len(features) > slow_corr_max
                        and not bool(cfg["analysis"].get(
                            "allow_slow_correlation", False))):
                    raise RuntimeError(
                        "scan_cache disabled with {0} features (> {1}): "
                        "correlation pass-2 would re-parse the full CSV ~{2} "
                        "times. Enable io.scan_cache (recommended; "
                        "~rows×features×8 bytes of scratch disk) or set "
                        "analysis.allow_slow_correlation: true to accept the "
                        "quadratic cost.".format(
                            len(features), slow_corr_max, n_pair_parses))
                logger.warning(
                    "scan_cache disabled — correlation pass-2 will re-parse "
                    "the full CSV ~%d times (%d chunks, one parse per block "
                    "pair). Enable io.scan_cache to bring this down to zero "
                    "extra parses at ~rows×features×8 bytes of scratch disk.",
                    n_pair_parses, n_chunks)
                edges = correlation.compute_correlation_edges(
                    features, path, always=[label_col],
                    spec_map=spec_map, get_spec_fn=get_spec,
                    chunk_size=chunk_size, threshold=corr_threshold,
                    min_overlap_frac=min_overlap,
                    row_mask=unsupervised_mask)
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
        base = build_ranked_report(iv_df, psi_df, lift_df, miss_df,
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
        base = apply_hard_filters(base, cfg)
        base = rank_and_auto_keep(base, cfg)

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

        # Optional XLSX — only if openpyxl happens to be available. A missing
        # package is an expected skip; any other failure (disk full,
        # permissions) is a real error and must be visible as such.
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            openpyxl = None
            logger.info("Skipped XLSX generation: openpyxl not installed "
                        "(CSV + HTML reports are complete).")
        if openpyxl is not None:
            try:
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
                logger.info("Wrote optional XLSX: %s", xlsx_path)
            except Exception:
                logger.warning("XLSX generation failed (CSV + HTML reports "
                               "are complete):", exc_info=True)

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
        rh = report_hash(summary_path)
        write_auto_features_txt(summary, auto_path, top_n=top_n, report_hash=rh)

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
