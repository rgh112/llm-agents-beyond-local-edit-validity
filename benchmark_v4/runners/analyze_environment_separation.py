#!/usr/bin/env python3
"""Matched-cell environment-separation audit for cross-model summaries.

The paper's headline comparison pools six models and environment-specific
structural prompts. This audit asks whether the environment ordering is driven
by one model/prompt cell or recurs over matched cells. It treats each
model x prompt-class combination as a paired cluster and bootstraps those
clusters, so the resulting intervals are a descriptive robustness check rather
than a mixed-effects model.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


ENVS = ("word_ladder", "alloy", "gb1_sequence")
ENV_LABELS = {
    "word_ladder": "Word Ladder",
    "alloy": "Alloy-like",
    "gb1_sequence": "GB1",
}
PAIRS = (
    ("word_ladder", "alloy"),
    ("word_ladder", "gb1_sequence"),
    ("alloy", "gb1_sequence"),
)


def prompt_class(prompt: str) -> str:
    if prompt in {"zero_shot", "few_shot_format"}:
        return prompt
    if prompt in {"scaffold", "self_check", "few_shot_strategy"}:
        return "structural"
    return prompt


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.1f}"


def ci(vals: List[float], lo_q: float = 0.025, hi_q: float = 0.975) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    vals = sorted(vals)
    lo_idx = min(len(vals) - 1, max(0, int(lo_q * len(vals))))
    hi_idx = min(len(vals) - 1, max(0, int(hi_q * len(vals))))
    return vals[lo_idx], vals[hi_idx]


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        rows.extend(data.get("results", []))
    return [r for r in rows if r.get("env") in ENVS]


def build_cells(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), prompt_class(str(row["prompt"])), str(row["env"]))
        by_key[key] = row

    models = sorted({str(row["model"]) for row in rows})
    prompt_classes = ("zero_shot", "few_shot_format", "structural")
    cells: List[Dict[str, Any]] = []
    for model in models:
        for pc in prompt_classes:
            if not all((model, pc, env) in by_key for env in ENVS):
                continue
            env_rows = {env: by_key[(model, pc, env)] for env in ENVS}
            cell = {"model": model, "prompt_class": pc, "envs": {}}
            for env, row in env_rows.items():
                clean = int(row.get("n_surface_clean_episodes", 0))
                clean_success = int(row.get("n_surface_clean_successes", 0))
                total = int(row.get("n_total", 0))
                success = int(row.get("n_success", 0))
                cell["envs"][env] = {
                    "n_total": total,
                    "n_success": success,
                    "success_rate": success / total if total else None,
                    "n_surface_clean_episodes": clean,
                    "n_surface_clean_successes": clean_success,
                    "surface_clean_success_rate": (
                        clean_success / clean if clean else None
                    ),
                }
            cells.append(cell)
    return cells


def aggregate_envs(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for env in ENVS:
        n_total = sum(int(cell["envs"][env]["n_total"]) for cell in cells)
        n_success = sum(int(cell["envs"][env]["n_success"]) for cell in cells)
        n_clean = sum(int(cell["envs"][env]["n_surface_clean_episodes"]) for cell in cells)
        n_clean_success = sum(
            int(cell["envs"][env]["n_surface_clean_successes"]) for cell in cells
        )
        cell_surface_rates = [
            cell["envs"][env]["surface_clean_success_rate"]
            for cell in cells
            if cell["envs"][env]["surface_clean_success_rate"] is not None
        ]
        cell_success_rates = [
            cell["envs"][env]["success_rate"]
            for cell in cells
            if cell["envs"][env]["success_rate"] is not None
        ]
        out.append(
            {
                "env": env,
                "n_cells": len(cells),
                "n_total": n_total,
                "n_success": n_success,
                "success_rate": n_success / n_total if n_total else None,
                "n_surface_clean_episodes": n_clean,
                "n_surface_clean_successes": n_clean_success,
                "surface_clean_success_rate": (
                    n_clean_success / n_clean if n_clean else None
                ),
                "mean_cell_success_rate": mean(cell_success_rates),
                "mean_cell_surface_clean_success_rate": mean(cell_surface_rates),
            }
        )
    return out


def paired_difference(
    cells: List[Dict[str, Any]],
    metric: str,
    env_a: str,
    env_b: str,
) -> List[float]:
    diffs = []
    for cell in cells:
        a = cell["envs"][env_a][metric]
        b = cell["envs"][env_b][metric]
        if a is None or b is None:
            continue
        diffs.append(float(a) - float(b))
    return diffs


def bootstrap_pairs(
    cells: List[Dict[str, Any]],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for env_a, env_b in PAIRS:
        observed = paired_difference(cells, metric, env_a, env_b)
        boots = []
        for _ in range(n_boot):
            sample = [rng.choice(cells) for _ in cells]
            diffs = paired_difference(sample, metric, env_a, env_b)
            boots.append(mean(diffs))
        lo, hi = ci(boots)
        out.append(
            {
                "metric": metric,
                "env_a": env_a,
                "env_b": env_b,
                "n_cells": len(observed),
                "mean_difference": mean(observed),
                "bootstrap_ci": [lo, hi],
                "positive_cells": sum(1 for d in observed if d > 0),
                "negative_cells": sum(1 for d in observed if d < 0),
                "zero_cells": sum(1 for d in observed if d == 0),
                "min_cell_difference": min(observed),
                "max_cell_difference": max(observed),
            }
        )
    return out


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Matched Environment-Separation Audit",
        "",
        (
            "Each cell is a matched model x prompt-class cluster. Structural "
            "maps to the environment-specific structural prompt "
            "(`scaffold` for Word Ladder and `self_check` for Alloy/GB1). "
            "Intervals are paired-cell bootstrap CIs over clusters."
        ),
        "",
        "## Aggregate Rates",
        "",
        "| Environment | Cells | Success | Surface-clean success | Surface-clean episodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["environment_aggregates"]:
        lines.append(
            "| {env} | {cells} | {succ}/{total} ({sr}%) | "
            "{clean_succ}/{clean} ({scsr}%) | {clean}/{total} |".format(
                env=ENV_LABELS[row["env"]],
                cells=row["n_cells"],
                succ=row["n_success"],
                total=row["n_total"],
                sr=pct(row["success_rate"]),
                clean_succ=row["n_surface_clean_successes"],
                clean=row["n_surface_clean_episodes"],
                scsr=pct(row["surface_clean_success_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Cell Differences",
            "",
            "| Metric | Difference | Mean diff | 95% CI | Positive cells | Range |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    metric_names = {
        "success_rate": "Success",
        "surface_clean_success_rate": "Surface-clean success",
    }
    for row in result["paired_differences"]:
        lines.append(
            "| {metric} | {a} - {b} | {mean} | [{lo}, {hi}] | "
            "{pos}/{n} | [{mn}, {mx}] |".format(
                metric=metric_names[row["metric"]],
                a=ENV_LABELS[row["env_a"]],
                b=ENV_LABELS[row["env_b"]],
                mean=pct(row["mean_difference"]),
                lo=pct(row["bootstrap_ci"][0]),
                hi=pct(row["bootstrap_ci"][1]),
                pos=row["positive_cells"],
                n=row["n_cells"],
                mn=pct(row["min_cell_difference"]),
                mx=pct(row["max_cell_difference"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="cross_model_summary.json files")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.summaries)
    cells = build_cells(rows)
    if not cells:
        raise SystemExit("No matched environment cells found")

    result: Dict[str, Any] = {
        "note": (
            "Descriptive matched-cell bootstrap over model x prompt-class "
            "clusters; not a mixed-effects model."
        ),
        "n_rows": len(rows),
        "n_cells": len(cells),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "environment_aggregates": aggregate_envs(cells),
        "paired_differences": [],
        "cells": cells,
    }
    for metric in ("success_rate", "surface_clean_success_rate"):
        result["paired_differences"].extend(
            bootstrap_pairs(
                cells,
                metric=metric,
                n_boot=args.bootstrap_iters,
                seed=args.seed + (0 if metric == "success_rate" else 100000),
            )
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(result, Path(args.output_md))
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    print(f"Saved to {out_json}")
    print(f"Saved to {args.output_md}")


if __name__ == "__main__":
    main()
