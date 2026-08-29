"""Summary memory - compressed edit history with patterns."""
from typing import List
from benchmark_v4.memory.base_memory import BaseMemory
from benchmark_v4.failure_taxonomy import StepRecord


class SummaryMemory(BaseMemory):
    def __init__(self):
        super().__init__("summary")

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        if not trajectory:
            return ""

        valid_edits = [s for s in trajectory if s.valid]
        invalid_edits = [s for s in trajectory if not s.valid]
        total = len(trajectory)

        lines = [f"Edit history summary ({total} steps taken):"]
        lines.append(f"  Valid edits: {len(valid_edits)}, Invalid: {len(invalid_edits)}")

        if valid_edits:
            first = valid_edits[0]
            last = valid_edits[-1]
            lines.append(f"  First structure: {first.structure_before}")
            lines.append(f"  Current structure: {last.structure_after}")

        event_counts = {}
        for s in trajectory:
            for e in s.events:
                name = e.value if hasattr(e, "value") else str(e)
                event_counts[name] = event_counts.get(name, 0) + 1
        if event_counts:
            event_str = ", ".join(f"{k}: {v}" for k, v in event_counts.items())
            lines.append(f"  Issues encountered: {event_str}")

        recent = trajectory[-2:]
        if recent:
            lines.append("  Most recent edits:")
            for s in recent:
                edit = s.parsed_action if s.parsed_action else "(invalid)"
                lines.append(f"    Step {s.t}: {edit} -> {s.structure_after}")

        return "\n".join(lines)
