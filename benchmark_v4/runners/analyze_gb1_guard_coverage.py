#!/usr/bin/env python3
"""Counterfactual coverage audit for exact GB1 recoverability guards.

This is not an agent-performance baseline. It asks which stored GB1 failures
would be flagged by a non-model guard that rejects a chosen edit when exact
recoverability changes from true before the edit to false after the edit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from benchmark_v4.runners.analyze_gb1_exact_recoverability import (
    SURFACE_EVENTS,
    build_exact_solver,
    iter_raw_files,
    make_env,
    step_exact_labels,
)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _cell_key(ep: Dict[str, Any]) -> str:
    return f"{ep.get('model')}::{ep.get('prompt_condition')}"


def analyze(paths: Iterable[str]) -> Dict[str, Any]:
    env = make_env()
    exact = build_exact_solver(env)

    n = 0
    success = 0
    surface_clean = 0
    surface_clean_success = 0
    surface_clean_failures = 0
    guard_flagged = 0
    guard_flagged_surface_clean_failures = 0
    guard_flagged_successes = 0
    local_improving_guard_flags = 0
    residual_recoverable_surface_clean_failures = 0
    residual_unrecoverable_or_unknown_surface_clean_failures = 0
    first_flag_steps: Counter[str] = Counter()
    cell_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    examples = []

    for path in iter_raw_files(paths):
        try:
            with open(path) as f:
                ep = json.load(f)
        except Exception:
            continue
        if ep.get("env_name") != "gb1_sequence":
            continue

        n += 1
        ep_success = bool(ep.get("success"))
        success += int(ep_success)
        steps = ep.get("steps") or []
        events = [event for step in steps for event in (step.get("events") or [])]
        ep_surface_clean = not any(event in SURFACE_EVENTS for event in events)
        surface_clean += int(ep_surface_clean)
        surface_clean_success += int(ep_surface_clean and ep_success)
        surface_clean_failure = bool(ep_surface_clean and not ep_success)
        surface_clean_failures += int(surface_clean_failure)

        flagged = False
        first_flag_step = None
        first_flag_local_delta = None
        final_exact_recoverable = None
        last_labels = None

        for step in steps:
            labels = step_exact_labels(env, exact, step)
            if "after" in labels:
                last_labels = labels
                final_exact_recoverable = bool(labels["after"]["exact_recoverable"])
            if "before" not in labels or "after" not in labels:
                continue
            before = labels["before"]
            after = labels["after"]
            crosses_boundary = bool(
                before["exact_recoverable"] and not after["exact_recoverable"]
            )
            if crosses_boundary and not flagged:
                flagged = True
                first_flag_step = step.get("t")
                if (
                    before.get("local_proxy_score") is not None
                    and after.get("local_proxy_score") is not None
                ):
                    first_flag_local_delta = (
                        float(after["local_proxy_score"])
                        - float(before["local_proxy_score"])
                    )

        guard_flagged += int(flagged)
        guard_flagged_successes += int(flagged and ep_success)
        if flagged and first_flag_step is not None:
            first_flag_steps[str(first_flag_step)] += 1
        if flagged and first_flag_local_delta is not None and first_flag_local_delta > 0:
            local_improving_guard_flags += 1
        if surface_clean_failure and flagged:
            guard_flagged_surface_clean_failures += 1
        if surface_clean_failure and not flagged:
            if final_exact_recoverable is True:
                residual_recoverable_surface_clean_failures += 1
            else:
                residual_unrecoverable_or_unknown_surface_clean_failures += 1

        cell = cell_counts[_cell_key(ep)]
        cell["episodes"] += 1
        cell["success"] += int(ep_success)
        cell["surface_clean"] += int(ep_surface_clean)
        cell["surface_clean_failures"] += int(surface_clean_failure)
        cell["guard_flagged"] += int(flagged)
        cell["guard_flagged_surface_clean_failures"] += int(
            surface_clean_failure and flagged
        )

        if flagged and len(examples) < 20:
            examples.append(
                {
                    "path": str(path),
                    "model": ep.get("model"),
                    "prompt_condition": ep.get("prompt_condition"),
                    "seed": ep.get("seed"),
                    "success": ep_success,
                    "surface_clean": ep_surface_clean,
                    "first_flag_step": first_flag_step,
                    "first_flag_local_delta": first_flag_local_delta,
                    "final_exact_recoverable": final_exact_recoverable,
                    "last_exact_label_present": bool(last_labels),
                }
            )

    by_cell = []
    for key, counter in sorted(cell_counts.items()):
        model, prompt = key.split("::", 1)
        failures = counter["surface_clean_failures"]
        flagged_failures = counter["guard_flagged_surface_clean_failures"]
        by_cell.append(
            {
                "model": model,
                "prompt_condition": prompt,
                "episodes": counter["episodes"],
                "success": counter["success"],
                "surface_clean": counter["surface_clean"],
                "surface_clean_failures": failures,
                "guard_flagged_episodes": counter["guard_flagged"],
                "guard_flagged_surface_clean_failures": flagged_failures,
                "guard_coverage_surface_clean_failures": _rate(
                    flagged_failures, failures
                ),
            }
        )

    return {
        "note": (
            "Counterfactual coverage only: a recoverability guard can flag the "
            "stored boundary-crossing action, but this audit does not choose a "
            "replacement action or measure agent success after rejection."
        ),
        "n_gb1_episodes": n,
        "success": success,
        "success_rate": _rate(success, n),
        "surface_clean_episodes": surface_clean,
        "surface_clean_success": surface_clean_success,
        "surface_clean_success_rate": _rate(surface_clean_success, surface_clean),
        "surface_clean_failures": surface_clean_failures,
        "guard_flagged_episodes": guard_flagged,
        "guard_flagged_episode_rate": _rate(guard_flagged, n),
        "guard_flagged_successes": guard_flagged_successes,
        "guard_flagged_surface_clean_failures": guard_flagged_surface_clean_failures,
        "guard_coverage_surface_clean_failures": _rate(
            guard_flagged_surface_clean_failures, surface_clean_failures
        ),
        "residual_surface_clean_failures_not_guard_flagged": (
            surface_clean_failures - guard_flagged_surface_clean_failures
        ),
        "residual_recoverable_surface_clean_failures": (
            residual_recoverable_surface_clean_failures
        ),
        "residual_unrecoverable_or_unknown_surface_clean_failures": (
            residual_unrecoverable_or_unknown_surface_clean_failures
        ),
        "local_improving_guard_flags": local_improving_guard_flags,
        "local_improving_guard_flag_rate": _rate(
            local_improving_guard_flags, guard_flagged
        ),
        "first_flag_step_histogram": dict(sorted(first_flag_steps.items())),
        "by_model_prompt": by_cell,
        "examples": examples,
    }


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# GB1 Recoverability-Guard Coverage",
        "",
        payload["note"],
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| GB1 episodes | {payload['n_gb1_episodes']} |",
        f"| Surface-clean episodes | {payload['surface_clean_episodes']} |",
        f"| Surface-clean failures | {payload['surface_clean_failures']} |",
        f"| Guard-flagged surface-clean failures | {payload['guard_flagged_surface_clean_failures']} |",
        f"| Guard coverage of surface-clean failures | {payload['guard_coverage_surface_clean_failures']:.3f} |",
        f"| Residual failures not guard-flagged | {payload['residual_surface_clean_failures_not_guard_flagged']} |",
        f"| Residual recoverable failures | {payload['residual_recoverable_surface_clean_failures']} |",
        f"| Guard flags with local proxy improvement | {payload['local_improving_guard_flags']} |",
        "",
        "## First Flag Step Histogram",
        "",
        "| Step | Count |",
        "| --- | ---: |",
    ]
    for step, count in payload["first_flag_step_histogram"].items():
        lines.append(f"| {step} | {count} |")
    lines.extend(["", "## Model/Prompt Cells", "", "| Model | Prompt | Failures | Flagged | Coverage |", "| --- | --- | ---: | ---: | ---: |"])
    for row in payload["by_model_prompt"]:
        coverage = row["guard_coverage_surface_clean_failures"]
        coverage_s = "" if coverage is None else f"{coverage:.3f}"
        lines.append(
            f"| {row['model']} | {row['prompt_condition']} | "
            f"{row['surface_clean_failures']} | "
            f"{row['guard_flagged_surface_clean_failures']} | {coverage_s} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Experiment dirs/raw dirs/raw JSONs.")
    ap.add_argument(
        "--output-json",
        default="results_v4/gb1_exact_recoverability/gb1_guard_coverage.json",
    )
    ap.add_argument(
        "--output-md",
        default="results_v4/gb1_exact_recoverability/gb1_guard_coverage.md",
    )
    args = ap.parse_args()

    payload = analyze(args.paths)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(payload, out_md)
    print(json.dumps({k: v for k, v in payload.items() if k not in {"by_model_prompt", "examples"}}, indent=2))
    print(f"Saved to {out_json}")
    print(f"Saved to {out_md}")


if __name__ == "__main__":
    main()
