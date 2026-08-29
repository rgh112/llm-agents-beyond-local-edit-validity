#!/usr/bin/env python3
"""Extract compact representative trajectory examples from raw episode logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENV_LABELS = {
    "word_ladder": "Word Ladder",
    "alloy": "Alloy-like composition editing",
    "gb1_sequence": "GB1 landscape editing",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def compact_structure(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return " ".join(text.split())


def pick_examples(raw_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    examples: dict[str, dict[str, dict[str, Any]]] = {env: {} for env in ENV_LABELS}
    for path in sorted(raw_dir.glob("*.json")):
        episode = load_json(path)
        env = episode.get("env_name")
        if env not in examples:
            continue
        bucket = "success" if episode.get("success") else "failure"
        candidate = {"path": path, "episode": episode}
        if bucket not in examples[env]:
            examples[env][bucket] = candidate
            continue
        current = examples[env][bucket]["episode"]
        if is_better_example(episode, current, bucket):
            examples[env][bucket] = candidate
    return examples


def violation_count(episode: dict[str, Any]) -> int:
    final_metrics = episode.get("final_metrics") or {}
    if "total_violations" in final_metrics:
        return int(final_metrics["total_violations"])
    return sum(len(step.get("events", [])) for step in episode.get("steps", []))


def is_better_example(candidate: dict[str, Any], current: dict[str, Any], bucket: str) -> bool:
    if bucket == "success":
        return (violation_count(candidate), len(candidate.get("steps", []))) < (
            violation_count(current),
            len(current.get("steps", [])),
        )
    candidate_events = sum(len(step.get("events", [])) for step in candidate.get("steps", []))
    current_events = sum(len(step.get("events", [])) for step in current.get("steps", []))
    return (candidate_events, -len(candidate.get("steps", []))) > (
        current_events,
        -len(current.get("steps", [])),
    )


def summarize_episode(path: Path, episode: dict[str, Any]) -> list[str]:
    lines = []
    status = "success" if episode.get("success") else "failure"
    lines.append(
        f"- Source: `{path}`; prompt `{episode.get('prompt_condition')}`, "
        f"memory `{episode.get('memory_condition')}`, seed `{episode.get('seed')}`; "
        f"outcome: **{status}**."
    )
    terminal = episode.get("terminal_failure")
    if terminal:
        lines.append(f"- Terminal failure: `{terminal}`.")
    final_metrics = episode.get("final_metrics") or {}
    if final_metrics:
        shown = []
        for key in ["steps_taken", "true_fitness", "estimated_fitness", "variant", "total_violations"]:
            if key in final_metrics:
                shown.append(f"{key}={final_metrics[key]}")
        if shown:
            lines.append(f"- Final metrics: {', '.join(shown)}.")
    lines.append("")
    lines.append("| t | action | before -> after | events |")
    lines.append("|---|---|---|---|")
    for step in episode.get("steps", []):
        events = ", ".join(step.get("events", [])) or "-"
        before = compact_structure(step.get("structure_before", ""))
        after = compact_structure(step.get("structure_after", ""))
        action = compact_structure(step.get("raw_action", ""))
        lines.append(f"| {step.get('t')} | `{action}` | `{before}` -> `{after}` | {events} |")
    return lines


def render_markdown(examples: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "# Representative Trajectory Examples",
        "",
        "These examples are extracted deterministically from stored raw episode logs.",
        "They are included for interpretability and auditability; aggregate claims",
        "remain tied to the JSON summaries checked by `supplementary/verify_artifact.py`.",
        "",
    ]
    for env, label in ENV_LABELS.items():
        lines.append(f"## {label}")
        lines.append("")
        for bucket in ["success", "failure"]:
            item = examples.get(env, {}).get(bucket)
            lines.append(f"### Representative {bucket}")
            lines.append("")
            if item is None:
                lines.append("No example found in the scanned raw logs.")
            else:
                lines.extend(summarize_episode(item["path"], item["episode"]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="results_v4/main_prompt_20260311_223450/raw",
        help="Raw episode log directory to scan.",
    )
    parser.add_argument(
        "--output",
        default="supplementary/TRAJECTORY_EXAMPLES.md",
        help="Markdown file to write.",
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    output = Path(args.output)
    examples = pick_examples(raw_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(examples))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
