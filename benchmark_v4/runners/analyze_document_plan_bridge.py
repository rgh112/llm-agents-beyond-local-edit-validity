#!/usr/bin/env python3
"""Deterministic diagnostics for the document-plan bridge environment.

This script does not query LLM APIs. It checks that the NLP-native bridge task
has a concrete edit space, reachable targets, and measurable local-proxy
behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark_v4.envs.document_plan import (  # noqa: E402
    CONTRADICTIONS,
    ORDER_CONSTRAINTS,
    REQUIRED,
    SECTIONS,
    START_OUTLINES,
    VALID_OUTLINES,
    DocumentPlanEnv,
)


def hamming_to_valid(outline: list[str]) -> int:
    return min(sum(1 for x, y in zip(outline, target) if x != y) for target in VALID_OUTLINES)


def local_proxy_score(outline: list[str]) -> float:
    coverage = len(REQUIRED.intersection(outline))
    duplicate_penalty = len(outline) - len(set(outline))
    order_hits = sum(
        1
        for a, b in ORDER_CONSTRAINTS
        if a in outline and b in outline and outline.index(a) < outline.index(b)
    )
    contradiction_penalty = sum(1 for a, b in CONTRADICTIONS if a in outline and b in outline)
    return coverage + 0.5 * order_hits - duplicate_penalty - contradiction_penalty


def success(outline: list[str]) -> bool:
    env = DocumentPlanEnv()
    return env._outline_success(outline)


def analyze() -> dict:
    sections = list(SECTIONS)
    one_edit_actions = [(pos, section) for pos in range(5) for section in sections]
    start_rows = []
    total_valid_first_edits = 0
    local_improvement_edits = 0
    local_deceptive_edits = 0
    shortest_distances = []

    for start in START_OUTLINES:
        start_distance = hamming_to_valid(start)
        start_score = local_proxy_score(start)
        shortest_distances.append(start_distance)
        valid_first_edits = 0
        deceptive = 0
        improving = 0
        for pos, section in one_edit_actions:
            if start[pos] == section:
                continue
            edited = list(start)
            edited[pos] = section
            valid_first_edits += 1
            score_delta = local_proxy_score(edited) - start_score
            distance_delta = hamming_to_valid(edited) - start_distance
            if score_delta > 0:
                improving += 1
            if score_delta > 0 and distance_delta > 0:
                deceptive += 1

        total_valid_first_edits += valid_first_edits
        local_improvement_edits += improving
        local_deceptive_edits += deceptive
        start_rows.append(
            {
                "start": start,
                "success_initially": success(start),
                "shortest_repair_distance": start_distance,
                "local_proxy_score": start_score,
                "valid_first_edits": valid_first_edits,
                "local_proxy_improving_first_edits": improving,
                "local_proxy_deceptive_first_edits": deceptive,
            }
        )

    return {
        "environment": "document_plan",
        "num_sections": len(sections),
        "outline_slots": 5,
        "edit_action_count": len(one_edit_actions),
        "action_count_with_finalize": len(one_edit_actions) + 1,
        "num_start_outlines": len(START_OUTLINES),
        "num_valid_target_outlines": len(VALID_OUTLINES),
        "required_sections": sorted(REQUIRED),
        "order_constraints": list(ORDER_CONSTRAINTS),
        "contradictions": [list(x) for x in sorted(CONTRADICTIONS)],
        "shortest_repair_distance_min": min(shortest_distances),
        "shortest_repair_distance_max": max(shortest_distances),
        "shortest_repair_distance_mean": sum(shortest_distances) / len(shortest_distances),
        "total_valid_first_edits": total_valid_first_edits,
        "local_proxy_improving_first_edits": local_improvement_edits,
        "local_proxy_deceptive_first_edits": local_deceptive_edits,
        "local_proxy_deceptive_fraction_among_improving": (
            local_deceptive_edits / local_improvement_edits if local_improvement_edits else 0.0
        ),
        "starts": start_rows,
        "used_as_manuscript_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results_v4/document_plan_bridge_stats.json",
        help="Path for the deterministic diagnostic JSON.",
    )
    args = parser.parse_args()
    data = analyze()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {output}")
    print(
        "document_plan: "
        f"{data['num_start_outlines']} starts, "
        f"{data['edit_action_count']} edit actions, "
        f"repair distance {data['shortest_repair_distance_min']}-"
        f"{data['shortest_repair_distance_max']}, "
        f"deceptive local-proxy edits "
        f"{data['local_proxy_deceptive_first_edits']}/"
        f"{data['local_proxy_improving_first_edits']}"
    )


if __name__ == "__main__":
    main()
