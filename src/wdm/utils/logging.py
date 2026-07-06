"""Module logger setup. Call setup_logging() once from scripts/notebooks.

Log level can be overridden without touching code via the WDM_LOG_LEVEL
environment variable (e.g. WDM_LOG_LEVEL=DEBUG or WDM_LOG_LEVEL=30).
"""
import logging
import os
import sys


def _level_from_env(default):
    raw = os.environ.get("WDM_LOG_LEVEL")
    if not raw:
        return default
    if raw.isdigit():
        return int(raw)
    resolved = logging.getLevelName(raw.strip().upper())
    return resolved if isinstance(resolved, int) else default


def setup_logging(level=logging.INFO, name="wdm"):
    root = logging.getLogger(name)
    if root.handlers:
        return root
    root.setLevel(_level_from_env(level))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.propagate = False
    return root
