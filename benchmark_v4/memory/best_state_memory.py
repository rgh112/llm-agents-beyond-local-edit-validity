"""Best-state memory for constructive editing experiments."""
from typing import Any, Dict, List, Tuple

from benchmark_v4.failure_taxonomy import StepRecord
from benchmark_v4.memory.base_memory import BaseMemory


def _score_step(step: StepRecord) -> Tuple[float, str]:
    """Heuristic score from visible step information.

    This is deliberately environment-agnostic and uses only logged information.
    It is not an oracle; it gives the model a compact anchor for the best state
    seen under visible rewards and failure signals.
    """
    if step.info.get("success"):
        return 1e6, "terminal success"
    score = float(step.reward)
    score += 1.0 if step.valid else -1.0
    score -= 0.25 * len(step.events)
    if step.info.get("true_fitness") is not None:
        score += float(step.info["true_fitness"])
    if step.info.get("true_uts") is not None:
        score += float(step.info["true_uts"]) / 1000.0
    if step.info.get("true_density") is not None:
        score -= float(step.info["true_density"])
    return score, "visible heuristic"


class BestStateMemory(BaseMemory):
    """Expose current trajectory summary plus the best state seen so far."""

    def __init__(self):
        super().__init__("best_state")

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""

        scored: List[Tuple[float, str, StepRecord]] = []
        for step in trajectory:
            score, reason = _score_step(step)
            scored.append((score, reason, step))
        best_score, reason, best = max(scored, key=lambda x: x[0])

        event_counts: Dict[str, int] = {}
        for step in trajectory:
            for event in step.events:
                name = event.name if hasattr(event, "name") else str(event)
                event_counts[name] = event_counts.get(name, 0) + 1

        lines = [
            f"Trajectory summary: {len(trajectory)} steps taken.",
            f"Best seen state ({reason}, score={best_score:.2f}):",
            f"  after step {best.t}: {best.structure_after}",
        ]
        if event_counts:
            lines.append(
                "Issues so far: "
                + ", ".join(f"{k}={v}" for k, v in sorted(event_counts.items()))
            )
        lines.append("Use this best state as an anchor; avoid repeating known bad moves.")
        return "\n".join(lines)
