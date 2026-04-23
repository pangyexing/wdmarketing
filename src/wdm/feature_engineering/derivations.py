"""Time-window feature derivations — delta / ratio / incremental / velocity.

This module is intentionally self-contained: it builds derivative columns
from an *already-loaded* DataFrame of raw window-keyed features. It does
NOT read or write disk, does NOT depend on Stage-1 analytics, and is
pure-functional given a `DerivationPlan`. Callers (Stage-2 dataset builder
in a future PR; the generated `predict.py` for deployment) apply the plan
to a DataFrame, producing a new frame with derived columns appended (and
optionally the original windowed columns dropped).

Config schema (under `feature_derivations` in the product / global YAML):

    feature_derivations:
      enabled: true
      default_keep_original: both      # both | replace | drop_original
      default_nan_policy:
        ratio_zero_denominator: nan    # nan | zero | inf_clipped
        ratio_clip: 1.0e6              # used with inf_clipped
        both_sides_nan: nan
        one_side_nan: nan
      families:
        - family_base: bureau_amt
          ops:
            - op: delta
              left:  7d
              right: 180d
              output: bureau_amt_delta_7d_vs_180d
            - op: ratio
              numerator:   30d
              denominator: all
              output: bureau_amt_ratio_30d_over_all
              nan_policy: {ratio_zero_denominator: zero}
          keep_original: replace

Operator semantics:
  delta(left, right)     : col[left] - col[right]
  ratio(num, denom)      : col[num] / col[denom]
  incremental(chain=[w1, w2, w3, ...])
                         : pairwise col[w_{i+1}] - col[w_i]; emits len(chain)-1
                           outputs named <output_prefix>_<w_{i+1}>_minus_<w_i>
  velocity(short, long)  : (col[long] - col[short]) / (days(long) - days(short))
                           requires both windows to be convertible to days.

NaN rules (per op, overriding defaults):
  * `both_sides_nan`     : output NaN when both operands are NaN (or 'zero' for 0.0).
  * `one_side_nan`       : output NaN when exactly one operand is NaN.
  * `ratio_zero_denominator`: 'nan' | 'zero' | 'inf_clipped' (clip to ratio_clip).

Determinism: all ops are pure functions of the inputs + recipe. The JSON
recipe emitted by `to_recipe_json` is sufficient for `apply_recipe` to
reproduce the same columns on any compatible input frame — this is what
a deployed `predict.py` replays at scoring time.

Not yet wired into Stage-2 `build_dataset` or `predict_template.py` — that's
a follow-up PR. This module stands on its own with complete tests so the
operator semantics are locked down before integration.
"""
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical window → days. Used only for `velocity`; other ops are unit-free.
# Extend when new canonical keys are added to feature_groups.window_order.
WINDOW_DAYS: Dict[str, Optional[int]] = {
    "7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90,
    "180d": 180, "360d": 360, "720d": 720, "1080d": 1080,
    "1800d": 1800, "3600d": 3600,
    "1mon": 30, "3mon": 90, "6mon": 180, "12mon": 360, "24mon": 720,
    "1y": 360, "2y": 720, "3y": 1080, "5y": 1800, "10y": 3600,
    "all": None, "life": None, "hist": None,
}

_VALID_OPS = {"delta", "ratio", "incremental", "velocity"}
_VALID_KEEP = {"both", "replace", "drop_original"}
_VALID_RATIO_POLICY = {"nan", "zero", "inf_clipped"}
_VALID_BOTH_NAN = {"nan", "zero"}
_VALID_ONE_NAN = {"nan", "fill_with_zero_then_op"}

_DEFAULT_NAN_POLICY: Dict[str, Any] = {
    "ratio_zero_denominator": "nan",
    "ratio_clip": 1.0e6,
    "both_sides_nan": "nan",
    "one_side_nan": "nan",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Op:
    """A single derivation operation over a family's window-keyed columns.

    `inputs` is the ordered list of resolved raw column names the op reads.
    `op` names the operator; `output` is the new column name (or
    output_prefix for `incremental`, which emits multiple columns).
    `nan_policy` overlays on the plan's defaults for this op only.
    """
    family_base: str
    op: str                     # delta | ratio | incremental | velocity
    inputs: Tuple[str, ...]     # resolved column names, in op-specific order
    output: str                 # column name or prefix
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    nan_policy: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def with_resolved_policy(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Merge plan-level default nan_policy with op-level override."""
        merged = dict(_DEFAULT_NAN_POLICY)
        merged.update(defaults or {})
        merged.update(self.nan_policy or {})
        return merged


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    """Rules for one family: which ops to run + whether to keep originals."""
    family_base: str
    ops: Tuple[Op, ...]
    keep_original: str          # both | replace | drop_original


@dataclasses.dataclass(frozen=True)
class DerivationPlan:
    """Parsed, validated plan. `None` when `feature_derivations.enabled` is false."""
    enabled: bool
    default_keep_original: str
    default_nan_policy: Dict[str, Any]
    families: Tuple[FamilySpec, ...]

    def all_ops(self) -> List[Op]:
        return [op for fam in self.families for op in fam.ops]

    def required_base_columns(self) -> List[str]:
        """Columns that must exist in the input DataFrame. Deduplicated."""
        seen = []
        for op in self.all_ops():
            for c in op.inputs:
                if c not in seen:
                    seen.append(c)
        return seen

    def originals_to_drop(self) -> List[str]:
        """Flatten across families: originals that should be removed after op."""
        dropped = []
        for fam in self.families:
            if fam.keep_original == "drop_original":
                for op in fam.ops:
                    for c in op.inputs:
                        if c not in dropped:
                            dropped.append(c)
            elif fam.keep_original == "replace":
                # Only drop the exact inputs of this op (the specific windows used)
                for op in fam.ops:
                    for c in op.inputs:
                        if c not in dropped:
                            dropped.append(c)
        return dropped


# ---------------------------------------------------------------------------
# Plan loading / validation
# ---------------------------------------------------------------------------

def load_derivation_plan(cfg: Dict[str, Any]) -> Optional[DerivationPlan]:
    """Parse `feature_derivations` from a resolved config dict.

    Returns None when the block is absent or disabled — callers should skip
    derivation entirely in that case.

    Raises ValueError on invalid schema (unknown op, wrong keep_original,
    missing required fields, etc.) so misconfigurations fail fast rather
    than produce silently-wrong features.
    """
    fd = (cfg.get("feature_derivations") or {})
    if not fd.get("enabled"):
        return None

    keep_default = fd.get("default_keep_original", "both")
    _assert_in(keep_default, _VALID_KEEP, "default_keep_original")

    nan_default = dict(_DEFAULT_NAN_POLICY)
    nan_default.update(fd.get("default_nan_policy") or {})
    _validate_nan_policy(nan_default, "default_nan_policy")

    families = []
    for i, raw_fam in enumerate(fd.get("families") or []):
        families.append(_parse_family(raw_fam, i, keep_default, nan_default))

    return DerivationPlan(
        enabled=True,
        default_keep_original=keep_default,
        default_nan_policy=nan_default,
        families=tuple(families),
    )


def _parse_family(raw: Dict[str, Any], idx: int,
                  keep_default: str, nan_default: Dict[str, Any]) -> FamilySpec:
    base = raw.get("family_base")
    if not base:
        raise ValueError("feature_derivations.families[{0}]: missing 'family_base'".format(idx))
    keep = raw.get("keep_original", keep_default)
    _assert_in(keep, _VALID_KEEP, "families[{0}].keep_original".format(idx))
    ops_raw = raw.get("ops") or []
    if not ops_raw:
        raise ValueError("feature_derivations.families[{0}]: empty ops".format(idx))
    ops = tuple(_parse_op(base, raw_op, j, nan_default) for j, raw_op in enumerate(ops_raw))
    return FamilySpec(family_base=str(base), ops=ops, keep_original=str(keep))


def _parse_op(base: str, raw: Dict[str, Any], j: int,
              nan_default: Dict[str, Any]) -> Op:
    op = raw.get("op")
    if op not in _VALID_OPS:
        raise ValueError("family '{0}' op[{1}]: 'op' must be one of {2}"
                         .format(base, j, sorted(_VALID_OPS)))
    nan_policy = dict(raw.get("nan_policy") or {})
    _validate_nan_policy(nan_policy, "family '{0}' op[{1}].nan_policy".format(base, j),
                         strict=False)
    if op == "delta":
        left = _require(raw, "left", base, j)
        right = _require(raw, "right", base, j)
        output = _require(raw, "output", base, j)
        inputs = (_col(base, left), _col(base, right))
        meta = {"left": left, "right": right}
    elif op == "ratio":
        num = _require(raw, "numerator", base, j)
        den = _require(raw, "denominator", base, j)
        output = _require(raw, "output", base, j)
        inputs = (_col(base, num), _col(base, den))
        meta = {"numerator": num, "denominator": den}
    elif op == "incremental":
        chain = raw.get("chain")
        if not isinstance(chain, (list, tuple)) or len(chain) < 2:
            raise ValueError("family '{0}' op[{1}]: 'chain' must be a list of >=2 windows"
                             .format(base, j))
        output = _require(raw, "output_prefix", base, j)
        inputs = tuple(_col(base, w) for w in chain)
        meta = {"chain": list(chain)}
    elif op == "velocity":
        short = _require(raw, "short", base, j)
        long_ = _require(raw, "long", base, j)
        output = _require(raw, "output", base, j)
        if WINDOW_DAYS.get(short) is None or WINDOW_DAYS.get(long_) is None:
            raise ValueError("family '{0}' op[{1}]: 'velocity' requires both windows "
                             "to map to a known day count (short={2}, long={3})"
                             .format(base, j, short, long_))
        if WINDOW_DAYS[long_] <= WINDOW_DAYS[short]:
            raise ValueError("family '{0}' op[{1}]: velocity 'long' must represent "
                             "more days than 'short'".format(base, j))
        inputs = (_col(base, short), _col(base, long_))
        meta = {"short": short, "long": long_,
                "days_short": WINDOW_DAYS[short], "days_long": WINDOW_DAYS[long_]}
    else:
        raise AssertionError("unreachable: op={0}".format(op))

    return Op(family_base=base, op=op, inputs=inputs, output=str(output),
              meta=meta, nan_policy=nan_policy)


def _require(raw: Dict[str, Any], key: str, base: str, j: int) -> Any:
    v = raw.get(key)
    if v in (None, ""):
        raise ValueError("family '{0}' op[{1}]: missing required key '{2}'".format(base, j, key))
    return v


def _col(base: str, window: str) -> str:
    """Compose the column name `<family_base>_<window>`.

    This is the inverse of `parse_families`' split — if your dataset uses a
    different naming convention, emit the full column name directly via
    the raw config instead of relying on this helper (future PR will allow
    per-op `input_columns` override).
    """
    return "{0}_{1}".format(base, window)


def _assert_in(v: Any, allowed: set, name: str) -> None:
    if v not in allowed:
        raise ValueError("{0}: expected one of {1}, got {2!r}".format(name, sorted(allowed), v))


def _validate_nan_policy(policy: Dict[str, Any], name: str, strict: bool = True) -> None:
    for k, v in (policy or {}).items():
        if k == "ratio_zero_denominator":
            _assert_in(v, _VALID_RATIO_POLICY, name + "." + k)
        elif k == "ratio_clip":
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(name + ".ratio_clip must be a positive number")
        elif k == "both_sides_nan":
            _assert_in(v, _VALID_BOTH_NAN, name + "." + k)
        elif k == "one_side_nan":
            _assert_in(v, _VALID_ONE_NAN, name + "." + k)
        elif strict:
            raise ValueError("{0}: unknown key {1!r}".format(name, k))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def apply_derivations(df: pd.DataFrame,
                      plan: DerivationPlan) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Apply a plan to a DataFrame. Returns (new_df, added_columns, dropped_columns).

    Input columns that the plan depends on must already exist in `df`;
    missing inputs raise KeyError so callers can surface config/data drift
    fast.
    """
    if plan is None or not plan.enabled:
        return df, [], []

    missing = [c for c in plan.required_base_columns() if c not in df.columns]
    if missing:
        raise KeyError("derivations need these columns but they're not in the frame: "
                       "{0}".format(missing))

    out = df.copy()
    added: List[str] = []
    for fam in plan.families:
        for op in fam.ops:
            policy = op.with_resolved_policy(plan.default_nan_policy)
            if op.op == "delta":
                out[op.output] = _op_delta(out[op.inputs[0]], out[op.inputs[1]], policy)
                added.append(op.output)
            elif op.op == "ratio":
                out[op.output] = _op_ratio(out[op.inputs[0]], out[op.inputs[1]], policy)
                added.append(op.output)
            elif op.op == "velocity":
                days = float(op.meta["days_long"] - op.meta["days_short"])
                out[op.output] = _op_velocity(out[op.inputs[0]], out[op.inputs[1]], days, policy)
                added.append(op.output)
            elif op.op == "incremental":
                chain = op.meta["chain"]
                for i in range(len(chain) - 1):
                    src_short = out[op.inputs[i]]
                    src_long = out[op.inputs[i + 1]]
                    col = "{0}_{1}_minus_{2}".format(op.output, chain[i + 1], chain[i])
                    out[col] = _op_delta(src_long, src_short, policy)
                    added.append(col)

    dropped = plan.originals_to_drop()
    # Only drop originals that still exist (a later op may have overwritten
    # one with the same name — highly unusual but cheap to guard).
    dropped_actual = [c for c in dropped if c in out.columns and c not in added]
    if dropped_actual:
        out = out.drop(columns=dropped_actual)
    return out, added, dropped_actual


def _prep_pair(a: pd.Series, b: pd.Series, policy: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast both series to float, honour one_side_nan policy, return arrays + nan_mask of the result."""
    aa = pd.to_numeric(a, errors="coerce").astype(float).values
    bb = pd.to_numeric(b, errors="coerce").astype(float).values
    a_nan = np.isnan(aa)
    b_nan = np.isnan(bb)
    one_side = policy.get("one_side_nan", "nan")
    if one_side == "fill_with_zero_then_op":
        aa = np.where(a_nan & ~b_nan, 0.0, aa)
        bb = np.where(b_nan & ~a_nan, 0.0, bb)
    # Recompute mask after potential fill
    return aa, bb, (np.isnan(aa) | np.isnan(bb))


def _apply_both_nan(out: np.ndarray, a: pd.Series, b: pd.Series,
                    policy: Dict[str, Any]) -> np.ndarray:
    if policy.get("both_sides_nan", "nan") == "zero":
        a_nan = pd.to_numeric(a, errors="coerce").isna().values
        b_nan = pd.to_numeric(b, errors="coerce").isna().values
        both_nan = a_nan & b_nan
        out = np.where(both_nan, 0.0, out)
    return out


def _op_delta(left: pd.Series, right: pd.Series, policy: Dict[str, Any]) -> np.ndarray:
    aa, bb, _ = _prep_pair(left, right, policy)
    out = aa - bb
    return _apply_both_nan(out, left, right, policy)


def _op_ratio(num: pd.Series, den: pd.Series, policy: Dict[str, Any]) -> np.ndarray:
    aa, bb, _ = _prep_pair(num, den, policy)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = aa / bb
    zero_den = (bb == 0.0) & ~np.isnan(bb)
    strategy = policy.get("ratio_zero_denominator", "nan")
    if strategy == "nan":
        out = np.where(zero_den, np.nan, out)
    elif strategy == "zero":
        out = np.where(zero_den, 0.0, out)
    elif strategy == "inf_clipped":
        clip = float(policy.get("ratio_clip", 1.0e6))
        # Numerator sign determines +clip / -clip; 0/0 stays NaN regardless.
        sign = np.where(aa > 0, 1.0, np.where(aa < 0, -1.0, 0.0))
        clipped = sign * clip
        out = np.where(zero_den, np.where(aa == 0.0, np.nan, clipped), out)
    return _apply_both_nan(out, num, den, policy)


def _op_velocity(short: pd.Series, long_: pd.Series, days: float,
                 policy: Dict[str, Any]) -> np.ndarray:
    aa, bb, _ = _prep_pair(short, long_, policy)
    out = (bb - aa) / days
    return _apply_both_nan(out, short, long_, policy)


# ---------------------------------------------------------------------------
# Recipe serialization (deploy-time replay)
# ---------------------------------------------------------------------------

_RECIPE_VERSION = 1


def to_recipe_json(plan: DerivationPlan) -> Dict[str, Any]:
    """Serialize a plan to the minimal JSON recipe deployed with the model.

    The recipe only stores resolved column names, the operator, and the
    per-op nan_policy — deployers don't need to know about window parsing.
    """
    if plan is None or not plan.enabled:
        return {"version": _RECIPE_VERSION, "enabled": False, "ops": [], "drop_originals": []}
    ops_json = []
    for fam in plan.families:
        for op in fam.ops:
            if op.op == "incremental":
                chain = op.meta["chain"]
                for i in range(len(chain) - 1):
                    ops_json.append({
                        "op": "delta",
                        "inputs": [op.inputs[i + 1], op.inputs[i]],
                        "output": "{0}_{1}_minus_{2}".format(op.output, chain[i + 1], chain[i]),
                        "nan_policy": op.with_resolved_policy(plan.default_nan_policy),
                    })
            elif op.op == "velocity":
                ops_json.append({
                    "op": "velocity",
                    "inputs": list(op.inputs),
                    "output": op.output,
                    "days": float(op.meta["days_long"] - op.meta["days_short"]),
                    "nan_policy": op.with_resolved_policy(plan.default_nan_policy),
                })
            else:
                ops_json.append({
                    "op": op.op,
                    "inputs": list(op.inputs),
                    "output": op.output,
                    "nan_policy": op.with_resolved_policy(plan.default_nan_policy),
                })
    return {
        "version": _RECIPE_VERSION,
        "enabled": True,
        "ops": ops_json,
        "drop_originals": plan.originals_to_drop(),
    }


def apply_recipe(df: pd.DataFrame, recipe: Dict[str, Any]) -> pd.DataFrame:
    """Deploy-time replay of a recipe. Equivalent to `apply_derivations` but
    reads the JSON structure directly so the generated `predict.py` can ship
    this function verbatim without importing the DerivationPlan dataclasses.
    """
    if not recipe or not recipe.get("enabled"):
        return df
    if recipe.get("version") != _RECIPE_VERSION:
        raise ValueError("unsupported derivations recipe version: {0}".format(recipe.get("version")))
    out = df.copy()
    for op in recipe.get("ops") or []:
        policy = dict(_DEFAULT_NAN_POLICY)
        policy.update(op.get("nan_policy") or {})
        name = op["op"]
        inputs = op["inputs"]
        output = op["output"]
        if name == "delta":
            out[output] = _op_delta(out[inputs[0]], out[inputs[1]], policy)
        elif name == "ratio":
            out[output] = _op_ratio(out[inputs[0]], out[inputs[1]], policy)
        elif name == "velocity":
            days = float(op["days"])
            out[output] = _op_velocity(out[inputs[0]], out[inputs[1]], days, policy)
        else:
            raise ValueError("unknown op in recipe: {0}".format(name))
    drop = [c for c in (recipe.get("drop_originals") or []) if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out


def dumps(recipe: Dict[str, Any]) -> str:
    """Stable JSON serialization for committing into a deploy bundle."""
    return json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=False)
