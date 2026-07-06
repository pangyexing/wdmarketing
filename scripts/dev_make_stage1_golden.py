#!/usr/bin/env python
"""Regenerate the Stage-1 golden snapshot under tests/stage1_golden/expected/.

Run this ONCE with the canonical analysis environment (env36 — see README
"环境" for the $PY convention) BEFORE refactoring, and re-run only when
(a) the environment (numpy/pandas) changes, or (b) an intentional behavior
change is made to Stage-1:

    PYTHONPATH=src $PY scripts/dev_make_stage1_golden.py

tests/test_stage1_golden.py then asserts that the current code reproduces
these artifacts byte-for-byte.
"""
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "stage1_golden"))

import dataset_gen  # noqa: E402


def strip_created_at(text):
    return "\n".join(line for line in text.splitlines()
                     if not line.startswith("# created_at:")) + "\n"


def main():
    from wdm.pipeline.stage1 import run_stage1
    from wdm.utils.paths import report_dir, selected_features_dir

    tmp = Path(tempfile.mkdtemp(prefix="stage1_golden_"))
    try:
        dataset_gen.prepare_repo(tmp)
        cfg = dataset_gen.build_cfg(tmp)
        run_stage1(cfg)

        expected = ROOT / "tests" / "stage1_golden" / "expected"
        if expected.exists():
            shutil.rmtree(expected)
        expected.mkdir(parents=True)

        rdir = report_dir(cfg)
        copied = []
        for csv in sorted(rdir.glob("*.csv")):
            shutil.copy2(csv, expected / csv.name)
            copied.append(csv.name)

        auto_path = selected_features_dir(cfg) / "v1_auto.txt"
        with open(auto_path, "r", encoding="utf-8") as f:
            auto_text = f.read()
        with open(expected / "v1_auto.txt", "w", encoding="utf-8") as f:
            f.write(strip_created_at(auto_text))
        copied.append("v1_auto.txt")

        import numpy
        import pandas
        meta = {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "dataset_seed": dataset_gen.DATASET_SEED,
            "n_rows": dataset_gen.N_ROWS,
            "files": copied,
        }
        with open(expected / "snapshot_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("Golden snapshot written to {0}".format(expected))
        for name in copied:
            print("  {0}".format(name))
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


if __name__ == "__main__":
    main()
