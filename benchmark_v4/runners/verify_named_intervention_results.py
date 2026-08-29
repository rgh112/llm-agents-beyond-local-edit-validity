#!/usr/bin/env python3
"""Validate named-intervention analyzer outputs before manuscript use.

This checker is intentionally about structural completeness, not about making a
positive result pass. It verifies that a target-specific intervention summary
matches the planned budget/coverage, that no smoke/closed/primary outputs are
mixed into the wrong target, and that preregistered H1--H4 records are present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


DEFAULT_BUDGET = Path("results_v4/strong_accept_interventions/named_intervention_budget.json")
DEFAULT_SUMMARIES = {
    "primary_smoke": Path(
        "results_v4/strong_accept_interventions/"
        "primary_smoke_intervention_prediction_summary.json"
    ),
    "primary": Path(
        "results_v4/strong_accept_interventions/"
        "primary_intervention_prediction_summary.json"
    ),
    "closed": Path(
        "results_v4/strong_accept_interventions/"
        "closed_intervention_prediction_summary.json"
    ),
}
EXPECTED_SUFFIXES = {
    "primary_smoke": {
        "primary_smoke_word_interface",
        "primary_smoke_alloy_gb1_interface",
        "primary_smoke_word_self_consistency",
        "primary_smoke_alloy_gb1_self_consistency",
        "primary_smoke_word_tot_style",
        "primary_smoke_alloy_gb1_tot_style",
    },
    "primary": {
        "primary_word_interface",
        "primary_alloy_gb1_interface",
        "primary_word_self_consistency",
        "primary_alloy_gb1_self_consistency",
        "primary_word_tot_style",
        "primary_alloy_gb1_tot_style",
    },
    "closed": {
        "closed_alloy_gb1_interface",
        "closed_alloy_gb1_self_consistency",
        "closed_alloy_gb1_tot_style",
    },
}
FULL_NAMED_ENVS = {"word_ladder", "alloy", "gb1_sequence"}
FULL_NAMED_CONTROLLERS = {"reflexion_retry", "self_consistency", "tot_style_beam"}
VALID_FINAL_STATUSES = {"supported", "weakened", "mixed"}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON: {path}")
    with path.open() as f:
        return json.load(f)


def budget_target(budget: Dict[str, Any], target: str) -> Dict[str, Any]:
    for row in budget.get("targets", []):
        if row.get("target") == target:
            return row
    fail(f"budget target not found: {target}")


def expected_rows(target_budget: Dict[str, Any]) -> Tuple[int, Dict[Tuple[str, str], int], Set[str]]:
    source_rows = 0
    by_env_controller: Dict[Tuple[str, str], int] = defaultdict(int)
    envs: Set[str] = set()
    for block in target_budget.get("blocks", []):
        for row in block.get("rows", []):
            source_rows += 1
            key = (row["env"], row["controller"])
            by_env_controller[key] += int(row["seeds"])
            envs.add(row["env"])
    return source_rows, dict(by_env_controller), envs


def aggregate_map(summary: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out = {}
    for row in summary.get("aggregates", []):
        key = (row.get("env"), row.get("controller"))
        out[key] = row.get("aggregate") or {}
    return out


def all_source_summaries(summary: Dict[str, Any]) -> Iterable[str]:
    for row in summary.get("aggregates", []):
        for source in (row.get("aggregate") or {}).get("source_summaries", []):
            yield str(source)


def source_has_expected_suffix(source: str, suffixes: Set[str]) -> bool:
    return any(f"planning_wrappers_{suffix}_" in source for suffix in suffixes)


def check_hypotheses(summary: Dict[str, Any], target: str, envs: Set[str]) -> None:
    checks = {
        row.get("id"): row
        for row in summary.get("preregistered_hypothesis_checks", [])
    }
    if set(checks) != {"H1", "H2", "H3", "H4"}:
        fail(f"unexpected hypothesis IDs: {sorted(checks)}")
    require_full = target in {"primary_smoke", "primary"} and envs == FULL_NAMED_ENVS
    for hid, row in checks.items():
        status = str(row.get("status"))
        if require_full:
            if row.get("named_panel_complete") is not True:
                fail(f"{hid} did not mark named panel complete")
            if status.startswith("preliminary_") or status.startswith("pending"):
                fail(f"{hid} has non-final status for full named panel: {status}")
            if status not in VALID_FINAL_STATUSES:
                fail(f"{hid} has unknown final status: {status}")
        else:
            if row.get("named_panel_complete") is True:
                fail(f"{hid} unexpectedly marked boundary panel as full named panel")


def validate(target: str, budget_path: Path, summary_path: Path) -> None:
    budget = load_json(budget_path)
    summary = load_json(summary_path)
    target_budget = budget_target(budget, target)
    expected_source_rows, expected_groups, envs = expected_rows(target_budget)
    expected_episodes = int(target_budget["episodes"])

    if int(summary.get("n_source_rows") or 0) != expected_source_rows:
        fail(
            f"n_source_rows mismatch for {target}: "
            f"{summary.get('n_source_rows')} vs {expected_source_rows}"
        )
    aggregates = aggregate_map(summary)
    if set(aggregates) != set(expected_groups):
        fail(
            f"aggregate group mismatch for {target}: "
            f"got {sorted(aggregates)}, expected {sorted(expected_groups)}"
        )
    total_n = 0
    for key, expected_n in sorted(expected_groups.items()):
        actual_n = int(aggregates[key].get("n") or 0)
        total_n += actual_n
        if actual_n != expected_n:
            fail(f"aggregate n mismatch for {target} {key}: {actual_n} vs {expected_n}")
    if total_n != expected_episodes:
        fail(f"total aggregate episodes mismatch for {target}: {total_n} vs {expected_episodes}")

    greedy_envs = {env for env, controller in expected_groups if controller == "greedy"}
    expected_delta_count = len(expected_groups) - len(greedy_envs)
    if len(summary.get("deltas_vs_greedy", [])) != expected_delta_count:
        fail(
            f"delta count mismatch for {target}: "
            f"{len(summary.get('deltas_vs_greedy', []))} vs {expected_delta_count}"
        )

    suffixes = EXPECTED_SUFFIXES.get(target)
    if suffixes:
        for source in all_source_summaries(summary):
            if not source_has_expected_suffix(source, suffixes):
                fail(f"source summary for {target} has unexpected suffix: {source}")

    check_hypotheses(summary, target, envs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(DEFAULT_SUMMARIES), required=True)
    ap.add_argument("--summary-json")
    ap.add_argument("--budget-json", default=str(DEFAULT_BUDGET))
    args = ap.parse_args()

    summary_path = Path(args.summary_json) if args.summary_json else DEFAULT_SUMMARIES[args.target]
    validate(args.target, Path(args.budget_json), summary_path)
    print(f"PASS named-intervention {args.target}: {summary_path}")


if __name__ == "__main__":
    main()
