"""Window memory (M1/M3/M5) - last N edits with results."""
from typing import List
from benchmark_v4.memory.base_memory import BaseMemory
from benchmark_v4.failure_taxonomy import StepRecord


class WindowMemory(BaseMemory):
    def __init__(self, window_size: int = 1):
        super().__init__(f"window_{window_size}")
        self.window_size = window_size

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""

        recent = trajectory[-self.window_size:]
        lines = ["Recent edits:"]
        for step in recent:
            edit_str = step.parsed_action if step.parsed_action else "(invalid)"
            valid_str = "valid" if step.valid else "INVALID"
            events_str = ""
            if step.events:
                event_names = [e.name if hasattr(e, 'name') else str(e) for e in step.events]
                events_str = f" | Events: {', '.join(event_names)}"
            lines.append(
                f"  Step {step.t}: Edit={edit_str} [{valid_str}] "
                f"Reward={step.reward:.2f}{events_str}"
            )
            # Show structure change if available
            if step.structure_before is not None and step.structure_after is not None:
                lines.append(f"    Before: {step.structure_before}")
                lines.append(f"    After:  {step.structure_after}")
            # Include feedback from info
            if step.info:
                feedback = step.info.get("feedback", "")
                if feedback:
                    lines.append(f"    Feedback: {feedback}")

        return "\n".join(lines)
