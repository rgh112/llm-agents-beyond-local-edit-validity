#!/usr/bin/env python3
"""Run scorer-leakage ablations for planning controllers.

Main paper search should use `text_visible`. `proxy` is a secondary
observation-level scorer, and `oracle` is a marked upper bound.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.runners.run_planning_wrappers import main as planning_main


if __name__ == "__main__":
    # This module intentionally reuses run_planning_wrappers via CLI by
    # documenting the canonical invocations in EXPERIMENT_ROBUSTNESS.md.
    # Keeping a thin entry point avoids diverging behavior between the main
    # planning runner and the scorer-ablation runner.
    planning_main()
