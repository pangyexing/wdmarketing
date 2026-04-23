"""Module logger setup. Call setup_logging() once from scripts/notebooks."""
import logging
import sys


def setup_logging(level=logging.INFO, name="wdm"):
    root = logging.getLogger(name)
    if root.handlers:
        return root
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.propagate = False
    return root
