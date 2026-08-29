#!/usr/bin/env python3
"""Summarize planned named-intervention hosted-run budgets.

This is a no-API helper for reviewer-facing and operator-facing run planning.
It mirrors the target structure in ``supplementary/run_strong_accept_interventions.sh``
and reports episode counts plus dry-run model-call upper bounds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.model_panels import expand_model_selection
from benchmark_v4.runners.run_planning_wrappers import (
    ENV_DEFAULT_HORIZON,
    controller_call_budget,
    expand_prompt,
)


TARGETS: Dict[str, List[Dict[str, Any]]] = {
    "primary_smoke": [
        dict(panel="small_block", envs=["word_ladder"], controllers=["greedy", "reflexion_retry"], seeds=10, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["greedy", "reflexion_retry"], seeds=10, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["word_ladder"], controllers=["self_consistency"], seeds=10, k=5, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["self_consistency"], seeds=10, k=5, width=2, depth=2),
        dict(panel="small_block", envs=["word_ladder"], controllers=["tot_style_beam"], seeds=10, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["tot_style_beam"], seeds=10, k=3, width=2, depth=2),
    ],
    "primary": [
        dict(panel="small_block", envs=["word_ladder"], controllers=["greedy", "reflexion_retry"], seeds=30, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["greedy", "reflexion_retry"], seeds=50, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["word_ladder"], controllers=["self_consistency"], seeds=30, k=5, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["self_consistency"], seeds=50, k=5, width=2, depth=2),
        dict(panel="small_block", envs=["word_ladder"], controllers=["tot_style_beam"], seeds=30, k=3, width=2, depth=2),
        dict(panel="small_block", envs=["alloy", "gb1_sequence"], controllers=["tot_style_beam"], seeds=50, k=3, width=2, depth=2),
    ],
    "closed": [
        dict(models=["openrouter:openai/gpt-4o-mini", "openrouter:openai/gpt-4o"], envs=["alloy", "gb1_sequence"], controllers=["greedy", "reflexion_retry"], seeds=20, k=3, width=2, depth=2),
        dict(models=["openrouter:openai/gpt-4o-mini", "openrouter:openai/gpt-4o"], envs=["alloy", "gb1_sequence"], controllers=["self_consistency"], seeds=20, k=5, width=2, depth=2),
        dict(models=["openrouter:openai/gpt-4o-mini", "openrouter:openai/gpt-4o"], envs=["alloy", "gb1_sequence"], controllers=["tot_style_beam"], seeds=20, k=3, width=2, depth=2),
    ],
}


def summarize_block(block: Dict[str, Any]) -> Dict[str, Any]:
    models = block.get("models") or expand_model_selection(None, block.get("panel"))
    rows = []
    episodes = 0
    max_calls = 0
    for model in models:
        for env in block["envs"]:
            horizon = ENV_DEFAULT_HORIZON[env]
            prompt = expand_prompt(env, "__structural__")
            for controller in block["controllers"]:
                calls_per_action = controller_call_budget(
                    controller,
                    k=int(block["k"]),
                    beam_width=int(block["width"]),
                    beam_depth=int(block["depth"]),
                )
                per_episode = horizon * calls_per_action
                n = int(block["seeds"])
                episodes += n
                max_calls += n * per_episode
                rows.append(
                    {
                        "model": model,
                        "env": env,
                        "prompt": prompt,
                        "controller": controller,
                        "seeds": n,
                        "horizon": horizon,
                        "calls_per_action": calls_per_action,
                        "max_calls_per_episode": per_episode,
                        "max_calls": n * per_episode,
                    }
                )
    return {
        "models": models,
        "envs": block["envs"],
        "controllers": block["controllers"],
        "seeds_per_cell": int(block["seeds"]),
        "k": int(block["k"]),
        "beam_width": int(block["width"]),
        "beam_depth": int(block["depth"]),
        "episodes": episodes,
        "max_calls": max_calls,
        "rows": rows,
    }


def summarize_targets() -> Dict[str, Any]:
    targets = []
    for name, blocks in TARGETS.items():
        block_summaries = [summarize_block(block) for block in blocks]
        targets.append(
            {
                "target": name,
                "episodes": sum(block["episodes"] for block in block_summaries),
                "max_calls": sum(block["max_calls"] for block in block_summaries),
                "blocks": block_summaries,
            }
        )
    return {
        "note": (
            "No-API dry-run budget summary for named-intervention hosted targets. "
            "Max calls are planning-time upper bounds; actual calls can be lower "
            "when episodes terminate early."
        ),
        "targets": targets,
    }


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Named-Intervention Hosted-Run Budget",
        "",
        summary["note"],
        "",
        "| Target | Episodes | Max calls | Role |",
        "| --- | ---: | ---: | --- |",
    ]
    roles = {
        "primary_smoke": "low-cost launch-risk check; not completion evidence by itself",
        "primary": "full open-family named-intervention case study",
        "closed": "optional GPT-4o-mini/GPT-4o boundary panel",
    }
    for target in summary["targets"]:
        lines.append(
            f"| `{target['target']}` | {target['episodes']:,} | "
            f"{target['max_calls']:,} | {roles[target['target']]} |"
        )
    lines.extend(["", "## Blocks", ""])
    for target in summary["targets"]:
        lines.extend(
            [
                f"### `{target['target']}`",
                "",
                "| Environments | Controllers | Seeds/cell | Episodes | Max calls |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for block in target["blocks"]:
            lines.append(
                "| {envs} | {controllers} | {seeds} | {episodes:,} | {calls:,} |".format(
                    envs=", ".join(block["envs"]),
                    controllers=", ".join(block["controllers"]),
                    seeds=block["seeds_per_cell"],
                    episodes=block["episodes"],
                    calls=block["max_calls"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    summary = summarize_targets()
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, Path(args.output_md))
    print(json.dumps({"targets": summary["targets"]}, indent=2))
    print(f"Saved to {out_json}")
    print(f"Saved to {args.output_md}")


if __name__ == "__main__":
    main()
