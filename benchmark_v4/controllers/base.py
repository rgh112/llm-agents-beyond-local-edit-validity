"""Base controller interfaces for LLM edit selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from benchmark_v4.models.base_model import BaseModel, Message


@dataclass
class ControllerDecision:
    action: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseController:
    """Selects the next environment action from a prompt state."""

    condition_name = "base"

    def select_action(self, *, env, prompt_builder, memory_module, model: BaseModel,
                      messages: List[Message], observation: str, step: int,
                      budget: int, env_name: str) -> ControllerDecision:
        raise NotImplementedError


class GreedyController(BaseController):
    """Original one-call, one-action policy."""

    condition_name = "greedy"

    def __init__(self, condition_name: str = "greedy"):
        self.condition_name = condition_name

    def select_action(self, *, env, prompt_builder, memory_module, model: BaseModel,
                      messages: List[Message], observation: str, step: int,
                      budget: int, env_name: str) -> ControllerDecision:
        response = model.generate(messages)
        return ControllerDecision(
            action=response.raw_text,
            candidates=[{
                "rank": 0,
                "raw_action": response.raw_text,
                "token_usage": response.token_usage,
            }],
            metadata={"token_usage": response.token_usage, "model_calls": 1},
        )
