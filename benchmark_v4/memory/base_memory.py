"""Base memory module interface."""
from abc import ABC, abstractmethod
from typing import List

from benchmark_v4.failure_taxonomy import StepRecord


class BaseMemory(ABC):
    """Determines what trajectory context to include in prompts.

    Each memory condition (state-only, window, summary) implements this.
    """

    def __init__(self, condition_name: str):
        self.condition_name = condition_name

    @abstractmethod
    def get_context(self, trajectory: List[StepRecord],
                    current_observation: str) -> str:
        """Extract context string from trajectory history.

        Args:
            trajectory: List of past step records
            current_observation: Current state observation

        Returns:
            Context string to be inserted into prompt
        """
        pass

    def reset(self):
        """Reset any internal state."""
        pass
