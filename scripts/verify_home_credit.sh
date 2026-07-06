#!/usr/bin/env bash
# End-to-end verification for the home_credit pipeline.
# Runs: env check -> pytest -> Stage 1 analysis -> simulate manual v2 ->
#       Stage 2 training -> deploy self-test -> tamper test.
#
# Expected wall-clock on 8-core laptop: ~15-30 min.
# Smoke-only shortcut: pass SMOKE=1 to skip the full scoring pass and use
# even smaller hyperopt budget:  SMOKE=1 bash scripts/verify_home_credit.sh

set -euo pipefail

# Resolve repo root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=src

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  MAX_EVALS=3
  STAGE1_EXTRA="--no-plots"
else
  MAX_EVALS=5
  STAGE1_EXTRA=""
fi

section() { echo; echo "===== $* ====="; }

# ─── 0. Environment & data presence ─────────────────────────────
section "0. Environment & data"
# Canonical env is Python 3.6.13 (conda env36 — see README 环境).
python3 -c "import sys; assert sys.version_info[:2]>=(3,6), sys.version"
pip install -q -r requirements.txt
python3 -c "import xgboost; assert xgboost.__version__.startswith('1.5.'), xgboost.__version__; print('xgb ok')"
test -s data/home_credit_wide.csv || {
  echo "data/home_credit_wide.csv missing — run scripts/build_home_credit_wide.py first"
  exit 1
}
wc -l data/home_credit_wide.csv

# ─── 1. Unit tests ──────────────────────────────────────────────
section "1. pytest"
pytest tests/ -v

# ─── 2. Stage 1: feature analysis ───────────────────────────────
section "2. Stage 1 (run_analysis)"
python3 scripts/run_analysis.py --product home_credit $STAGE1_EXTRA
R1=artifacts/home_credit/analysis
for f in summary.csv psi.csv iv_woe.csv lift.csv missing.csv \
         correlation_edges.csv families.csv semantic_groups.csv index.html; do
  test -s "$R1/report/$f" || { echo "MISS $R1/report/$f"; exit 1; }
done
test -s artifacts/home_credit/selected_features/v1_auto.txt

python3 - <<'PY'
import pandas as pd
s = pd.read_csv('artifacts/home_credit/analysis/report/summary.csv')
n_total = len(s); n_keep = int(s['auto_keep'].sum())
print('summary rows=%d, auto_keep=%d' % (n_total, n_keep))
assert n_total >= 900, 'expected ~972 columns analyzed, got %d' % n_total
assert n_keep >= 50,   'auto_keep=%d < 50; check iv_min / psi_cutoff' % n_keep
g = pd.read_csv('artifacts/home_credit/analysis/report/semantic_groups.csv')
missing = {'bureau','prev','instal','pos','cc'} - set(g['group_name'])
assert not missing, 'missing semantic groups: %s' % missing
print('Stage 1 ok — all 5 semantic groups present')
PY

wc -l artifacts/home_credit/selected_features/v1_auto.txt

# ─── 3. Simulate manual v2 (comment out the first selected feature) ─
section "3. v2_verify (simulate manual edit)"
cp artifacts/home_credit/selected_features/v1_auto.txt \
   artifacts/home_credit/selected_features/v2_verify.txt
python3 - <<'PY'
p = 'artifacts/home_credit/selected_features/v2_verify.txt'
lines = open(p).read().splitlines(); out=[]; dropped=None
for l in lines:
    if dropped is None and l.strip() and not l.startswith('#'):
        out.append('# ' + l); dropped = l.strip()
    else:
        out.append(l)
open(p, 'w').write('\n'.join(out) + '\n')
print('dropped feature for v2_verify:', dropped)
PY

# ─── 4. Stage 2: training + export ──────────────────────────────
section "4. Stage 2 (run_training, --max-evals=$MAX_EVALS)"
python3 scripts/run_training.py \
  --product home_credit --run-id verify \
  --features-version v2_verify --max-evals "$MAX_EVALS"
R2=artifacts/home_credit/models/verify
for f in booster.json feature_list.txt missing_spec.json metrics.json \
         importance.csv predict.py validation_samples.csv run_manifest.json \
         calibration.json \
         plots/roc_pr.png plots/ks.png plots/lift_decile.png \
         plots/importance_gain.png; do
  test -s "$R2/$f" || { echo "MISS $R2/$f"; exit 1; }
done

python3 - <<PY
import json
m = json.load(open('$R2/metrics.json'))
oot = m['oot']
print('OOT PR-AUC=%.3f ROC-AUC=%.3f Lift@10%%=%.2f' %
      (oot['pr_auc'], oot['roc_auc'], oot['lift_at_k']))
assert oot['roc_auc']   >= 0.70, 'ROC-AUC too low — training pipeline may be broken'
assert oot['lift_at_k'] >= 2.0,  'Lift@10%% too low'
print('Stage 2 ok')
PY

python3 - <<PY
import pandas as pd
vs = pd.read_csv('$R2/validation_samples.csv')
assert len(vs) == 100
days_cols = [c for c in vs.columns if c.lower().startswith('days_') or '_days_' in c.lower()]
if days_cols:
    col = days_cols[0]
    assert (vs[col] < 0).any(), 'validation_samples %s has no negatives — may have been pre-filled' % col
    print('validation_samples raw values ok —', col, 'min=', vs[col].min())
PY

# ─── 5. Deploy self-test ────────────────────────────────────────
section "5. Deploy self-test"
python3 "$R2/predict.py" --validate --tol 1e-6
if [[ "$SMOKE" != "1" ]]; then
  python3 "$R2/predict.py" --input data/home_credit_wide.csv --output /tmp/hc_scores.csv
  wc -l /tmp/hc_scores.csv
  head -3 /tmp/hc_scores.csv
fi

# ─── 6. Tamper test ─────────────────────────────────────────────
# Swap the first and last feature lines: a reordered feature_list.txt keeps
# the column COUNT intact (so the Predictor's consistency guard stays quiet)
# but feeds the booster a permuted matrix — --validate must catch the score
# drift. Strategy-independent, unlike mutating fill values (keep_nan products
# replay no fills at all).
section "6. Tamper test (should FAIL --validate)"
cp "$R2/feature_list.txt" "$R2/feature_list.txt.bak"
python3 - <<PY
p = '$R2/feature_list.txt'
with open(p, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
idx = [i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith('#')]
assert len(idx) >= 2, 'need at least two features to tamper'
i, j = idx[0], idx[-1]
lines[i], lines[j] = lines[j], lines[i]
with open(p, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('tampered ok (swapped %s <-> %s)' % (lines[j], lines[i]))
PY
if python3 "$R2/predict.py" --validate --tol 1e-6 2>/dev/null; then
  echo 'FAIL: --validate passed after tampering (feature order not actually honored)'
  mv "$R2/feature_list.txt.bak" "$R2/feature_list.txt"
  exit 1
fi
mv "$R2/feature_list.txt.bak" "$R2/feature_list.txt"
echo 'Tamper test ok — --validate correctly rejected the reordered feature list'

echo
echo '✅ home_credit ALL CHECKS PASSED'
