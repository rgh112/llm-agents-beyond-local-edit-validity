"""State-only memory (M0) - no edit history, just current state."""
from typing import List
from benchmark_v4.memory.base_memory import BaseMemory
from benchmark_v4.failure_taxonomy import StepRecord


class StateOnlyMemory(BaseMemory):
    def __init__(self):
        super().__init__("state_only")

    def get_context(self, trajectory: List[StepRecord], current_observation: str) -> str:
        return ""
