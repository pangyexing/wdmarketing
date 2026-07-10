"""Byte-level Stage-1 regression guard.

Runs the full Stage-1 pipeline on the deterministic synthetic dataset and
compares every report CSV (and v1_auto.txt, modulo its created_at line)
against the committed snapshot in tests/stage1_golden/expected/.

If this fails after a refactor, the refactor changed Stage-1 numbers.
If it fails after an environment upgrade (numpy/pandas float formatting),
regenerate the snapshot with scripts/dev_make_stage1_golden.py and review
the diff intentionally — see snapshot_meta.json for the recorded versions.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stage1_golden"))

import dataset_gen  # noqa: E402

EXPECTED_DIR = HERE / "stage1_golden" / "expected"

pytestmark = pytest.mark.skipif(
    not EXPECTED_DIR.is_dir(),
    reason="golden snapshot missing — run scripts/dev_make_stage1_golden.py first")


def test_environment_matches_snapshot():
    """Byte-level comparison is only meaningful in the environment the
    snapshot was generated under — numpy/pandas float formatting differs
    across versions. Fail HERE with an actionable message instead of a
    baffling byte-diff in the tests below.
    """
    import numpy
    import pandas
    meta_path = EXPECTED_DIR / "snapshot_meta.json"
    if not meta_path.is_file():
        pytest.skip("snapshot_meta.json missing — old-style snapshot")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    drift = []
    if meta.get("numpy") != numpy.__version__:
        drift.append("numpy {0} (snapshot) vs {1} (current)".format(
            meta.get("numpy"), numpy.__version__))
    if meta.get("pandas") != pandas.__version__:
        drift.append("pandas {0} (snapshot) vs {1} (current)".format(
            meta.get("pandas"), pandas.__version__))
    assert not drift, (
        "Environment drift — golden byte comparison is not meaningful: "
        "{0}. Either switch to the canonical env (see README 环境) or "
        "intentionally regenerate via scripts/dev_make_stage1_golden.py "
        "and review the diff.".format("; ".join(drift)))


def _strip_created_at(text):
    return "\n".join(line for line in text.splitlines()
                     if not line.startswith("# created_at:")) + "\n"


@pytest.fixture(scope="module")
def stage1_artifacts(tmp_path_factory):
    from wdm.pipeline.stage1 import run_stage1
    from wdm.utils.paths import report_dir, selected_features_dir

    repo = tmp_path_factory.mktemp("golden_repo")
    dataset_gen.prepare_repo(repo)
    cfg = dataset_gen.build_cfg(repo)
    run_stage1(cfg)
    return report_dir(cfg), selected_features_dir(cfg)


def test_csv_file_set_matches(stage1_artifacts):
    rdir, _ = stage1_artifacts
    produced = {p.name for p in rdir.glob("*.csv")}
    expected = {p.name for p in EXPECTED_DIR.glob("*.csv")}
    assert produced == expected


def test_report_csvs_byte_identical(stage1_artifacts):
    rdir, _ = stage1_artifacts
    mismatched = []
    for exp in sorted(EXPECTED_DIR.glob("*.csv")):
        got = (rdir / exp.name).read_bytes()
        want = exp.read_bytes()
        if got != want:
            mismatched.append(exp.name)
    assert not mismatched, (
        "Stage-1 output differs from golden snapshot: {0}".format(mismatched))


def test_v1_auto_txt_identical(stage1_artifacts):
    _, sf_dir = stage1_artifacts
    with open(sf_dir / "v1_auto.txt", "r", encoding="utf-8") as f:
        got = _strip_created_at(f.read())
    with open(EXPECTED_DIR / "v1_auto.txt", "r", encoding="utf-8") as f:
        want = f.read()
    assert got == want
