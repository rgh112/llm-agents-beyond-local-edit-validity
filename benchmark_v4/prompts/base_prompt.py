"""Base prompt builder interface for behavioral intervention conditions.

Each condition shapes the model's editing policy differently:
  - Format support: action grammar, observation reading
  - Strategic priors: domain-specific editing heuristics
  - Self-monitoring: constraint verification rules
"""
from abc import ABC, abstractmethod
from typing import Dict


class BasePromptBuilder(ABC):
    """Builds behavioral intervention prompts for constructive editing tasks.

    Conditions differ not only in formatting support, but in how strongly they
    shape the model's editing policy. Each subclass declares metadata describing
    the intervention type.
    """

    def __init__(self, condition_name: str, metadata: Dict[str, bool] = None):
        self.condition_name = condition_name
        self.metadata = metadata or {}

    @abstractmethod
    def build_system_prompt(self, env_name: str, task_description: str,
                            constraints: str, action_format: str) -> str:
        pass

    @abstractmethod
    def build_step_prompt(self, observation: str, memory_context: str,
                          step: int, budget: int,
                          env_name: str) -> str:
        pass
