"""Minimal packaging so `pip install -e .` makes `wdm` importable without the
per-script sys.path.insert hack (the hack is kept for back-compat — scripts
still run standalone from a bare checkout).
"""
from setuptools import find_packages, setup

setup(
    name="wdm",
    version="0.1.0",
    description="wdmarketing: Stage-1 feature screening + Stage-2 XGBoost training pipelines",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6",
)
