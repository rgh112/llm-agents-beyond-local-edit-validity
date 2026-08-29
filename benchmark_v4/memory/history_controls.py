"""History-control memory conditions.

These are diagnostic controls, not proposed agent memories. They test whether
history effects come from useful trajectory information or from anchoring and
format artifacts.
"""
from __future__ import annotations

import hashlib
from typing import List

from benchmark_v4.failure_taxonomy import StepRecord
from benchmark_v4.memory.base_memory import BaseMemory


def _event_names(step: StepRecord) -> str:
    names = [e.name if hasattr(e, "name") else str(e) for e in step.events]
    return ", ".join(names) if names else "none"


def _step_line(step: StepRecord) -> str:
    action = step.parsed_action if step.parsed_action else step.raw_action.strip()
    valid = "valid" if step.valid else "INVALID"
    return (
        f"Step {step.t}: action={action} [{valid}], "
        f"before={step.structure_before}, after={step.structure_after}, "
        f"events={_event_names(step)}"
    )


class RandomizedHistoryMemory(BaseMemory):
    """Expose the same recent history content in a deterministic scrambled order."""

    def __init__(self, window_size: int = 3):
        super().__init__(f"randomized_history_{window_size}")
        self.window_size = window_size

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""
        recent = list(trajectory[-self.window_size:])
        recent.sort(
            key=lambda step: hashlib.sha1(
                f"{step.t}:{step.raw_action}".encode("utf-8")
            ).hexdigest()
        )
        lines = ["Recent edits in scrambled order (history-control condition):"]
        lines.extend(f"  {_step_line(step)}" for step in recent)
        return "\n".join(lines)


class MisleadingHistoryMemory(BaseMemory):
    """Expose recent history with before/after swapped to measure anchoring risk."""

    def __init__(self, window_size: int = 3):
        super().__init__(f"misleading_history_{window_size}")
        self.window_size = window_size

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""
        lines = ["Recent edits with intentionally misleading state transitions:"]
        for step in trajectory[-self.window_size:]:
            action = step.parsed_action if step.parsed_action else step.raw_action.strip()
            valid = "valid" if step.valid else "INVALID"
            lines.append(
                f"  Step {step.t}: action={action} [{valid}], "
                f"before={step.structure_after}, after={step.structure_before}, "
                f"events={_event_names(step)}"
            )
        lines.append("Use the current state above as authoritative.")
        return "\n".join(lines)
