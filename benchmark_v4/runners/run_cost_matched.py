#!/usr/bin/env python3
"""Cost-matched controller comparison helpers.

This runner executes two complementary conditions:
  1. Regular planning controllers with reported calls/tokens per episode.
  2. Independent greedy restarts (`greedy_sampled`) at the same sampling
     temperature, so search gains can be compared against sampling alone.

For strict best-of-K over independent full episodes, run this script with
`--controllers greedy_sampled` and aggregate task-level successes externally
by seed bucket. The raw logs preserve all per-run trajectories.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.runners.run_planning_wrappers import main as planning_main


if __name__ == "__main__":
    planning_main()
