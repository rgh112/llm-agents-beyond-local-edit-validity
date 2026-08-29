"""Full trajectory memory for constructive editing experiments."""
from typing import List

from benchmark_v4.failure_taxonomy import StepRecord
from benchmark_v4.memory.base_memory import BaseMemory


class FullHistoryMemory(BaseMemory):
    """Expose the full edit trajectory.

    This condition is intentionally verbose. It tests whether access to the
    complete path helps construction or simply adds distracting local
    commitments.
    """

    def __init__(self):
        super().__init__("full_history")

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""

        lines = ["Full edit history:"]
        for step in trajectory:
            action = step.parsed_action if step.parsed_action else step.raw_action.strip()
            valid = "valid" if step.valid else "INVALID"
            events = [
                e.name if hasattr(e, "name") else str(e)
                for e in step.events
            ]
            event_text = f" | events: {', '.join(events)}" if events else ""
            lines.append(
                f"  Step {step.t}: {action} [{valid}]{event_text}"
            )
            lines.append(f"    before: {step.structure_before}")
            lines.append(f"    after:  {step.structure_after}")

        return "\n".join(lines)
