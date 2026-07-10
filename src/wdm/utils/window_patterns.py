"""Window-pattern presets and config-level pattern resolution.

Lives in utils (below config) so wdm.config can validate
feature_groups.window_patterns at load time without importing the analysis
layer — the import used to run config -> analysis.family, an inverted layer
dependency worked around with a lazy import.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

# Named presets for the window_patterns config. Each entry has:
#   pattern:    a regex with named groups (?P<base>...) and (?P<window>...)
#   alias:      optional dict mapping raw window token -> canonical key
#   alias_rule: optional str template; formatted as rule.format(window=raw_token)
# When neither alias nor alias_rule is provided, the raw match is kept verbatim.
WINDOW_PATTERN_PRESETS: Dict[str, Dict[str, Any]] = {
    # 天粒度
    "suffix_day":         {"pattern": r'^(?P<base>.+?)_(?P<window>7d|30d|90d|180d|360d|all|life|hist)$'},
    "prefix_d":           {"pattern": r'^(?P<base>.+?)_d(?P<window>7|30|90|180|360)$',
                           "alias_rule": "{window}d"},
    "chinese_jin_days":   {"pattern": r'^(?P<base>.+?)_近(?P<window>\d+)天$',
                           "alias_rule": "{window}d"},
    "english_last_days":  {"pattern": r'^(?P<base>.+?)_last(?P<window>\d+)days$',
                           "alias_rule": "{window}d"},
    "number_only":        {"pattern": r'^(?P<base>.+?)_(?P<window>7|14|30|60|90|180|360)$',
                           "alias_rule": "{window}d"},
    # 月粒度（保留 mon 作为 canonical key，不折算天）
    "suffix_mon":         {"pattern": r'^(?P<base>.+?)_(?P<window>1mon|3mon|6mon|12mon|24mon|all|life|hist)$'},
    "chinese_jin_months": {"pattern": r'^(?P<base>.+?)_近(?P<window>\d+)月$',
                           "alias_rule": "{window}mon"},
    "english_last_months":{"pattern": r'^(?P<base>.+?)_last(?P<window>\d+)months$',
                           "alias_rule": "{window}mon"},
    # 月粒度 → 折算到天 canonical key（若与天粒度混用推荐）
    "suffix_mon_to_days": {"pattern": r'^(?P<base>.+?)_(?P<window>\d+)mon$',
                           "alias": {"1": "30d", "3": "90d", "6": "180d",
                                     "12": "360d", "24": "720d"}},
    # 年粒度（保留 y 作为 canonical key）
    "suffix_year":        {"pattern": r'^(?P<base>.+?)_(?P<window>1y|2y|3y|5y|10y|all|life|hist)$'},
    "chinese_jin_years":  {"pattern": r'^(?P<base>.+?)_近(?P<window>\d+)年$',
                           "alias_rule": "{window}y"},
    "english_last_years": {"pattern": r'^(?P<base>.+?)_last(?P<window>\d+)years$',
                           "alias_rule": "{window}y"},
    # 年粒度 → 折算到天 canonical key
    "suffix_year_to_days":{"pattern": r'^(?P<base>.+?)_(?P<window>\d+)y$',
                           "alias": {"1": "360d", "2": "720d", "3": "1080d",
                                     "5": "1800d", "10": "3600d"}},
}


def resolve_patterns(cfg: Dict[str, Any]) -> List[Tuple[Any, Dict[str, str], Optional[str], str]]:
    """Return compiled patterns as [(regex, alias_map, alias_rule, pattern_id), ...]

    Accepts either `feature_groups.window_patterns` (list form) or the older
    singular `feature_groups.window_pattern` (string form). An empty list is
    returned if neither is configured.
    """
    fg = cfg.get("feature_groups") or {}
    raw = fg.get("window_patterns")
    if not raw:
        single = fg.get("window_pattern")
        if not single:
            return []
        raw = [{"pattern": single, "id": "legacy_window_pattern"}]

    resolved = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("window_patterns[{0}] must be a dict; got {1!r}".format(i, item))
        if "preset" in item:
            preset_name = item["preset"]
            if preset_name not in WINDOW_PATTERN_PRESETS:
                raise ValueError("Unknown window pattern preset: {0!r}. "
                                 "Available: {1}".format(preset_name,
                                                         sorted(WINDOW_PATTERN_PRESETS)))
            merged = dict(WINDOW_PATTERN_PRESETS[preset_name])
            for k, v in item.items():
                if k != "preset":
                    merged[k] = v
            item = merged
        pat = item.get("pattern")
        if not pat:
            raise ValueError("window_patterns[{0}] must provide 'preset' or 'pattern'".format(i))
        regex = re.compile(pat)
        if "base" not in regex.groupindex or "window" not in regex.groupindex:
            raise ValueError("window_patterns[{0}] pattern must contain named "
                             "groups (?P<base>...) and (?P<window>...); got {1!r}"
                             .format(i, pat))
        alias = item.get("alias") or {}
        # Coerce alias keys/values to str for safety (YAML may parse "7" as int)
        alias = {str(k): str(v) for k, v in alias.items()}
        alias_rule = item.get("alias_rule")
        pid = item.get("id") or item.get("preset") or "pattern_{0}".format(i)
        resolved.append((regex, alias, alias_rule, pid))
    return resolved
