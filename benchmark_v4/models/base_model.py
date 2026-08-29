"""Base model wrapper interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelResponse:
    raw_text: str
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model_id: str = ""


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


class BaseModel(ABC):
    """Wrapper for LLM API calls."""

    def __init__(self, model_id: str, temperature: float = 0.0,
                 max_tokens: int = 512, top_p: Optional[float] = None):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    @abstractmethod
    def generate(self, messages: List[Message]) -> ModelResponse:
        """Generate a response given message history.

        Args:
            messages: List of Message objects (system, user, assistant)

        Returns:
            ModelResponse with raw text and metadata
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        pass
