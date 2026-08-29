"""Base class for constructive editing environments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EditState:
    """Common state for all constructive editing environments."""
    t: int = 0                                    # current step
    budget_left: int = 0                          # remaining budget
    structure: Any = None                         # current partial structure (env-specific type)
    history: List[Dict[str, Any]] = field(default_factory=list)  # edit history
    latent: Dict[str, Any] = field(default_factory=dict)        # hidden state (recovery cost etc)
    done: bool = False


@dataclass
class StepResult:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    events: List[Any]


class BaseConstructiveEditEnv(ABC):
    """Base class for constructive editing environments.

    All environments follow the same pattern:
    1. Agent sees current partial structure + editing constraints
    2. Agent issues EDIT commands to modify the structure
    3. Agent issues FINALIZE to trigger full evaluation
    4. Path-dependent latent variables affect final success
    """

    def __init__(self, env_name: str, max_steps: int):
        self.env_name = env_name
        self.max_steps = max_steps
        self.state: EditState = EditState()
        self.rng: np.random.Generator = np.random.default_rng(0)
        self._done = False
        self._success = False
        self.trajectory: List[Any] = []  # List[StepRecord]

    # --- Properties for prompt builders ---
    @property
    @abstractmethod
    def task_description(self) -> str: ...

    @property
    @abstractmethod
    def constraints(self) -> str: ...

    @property
    @abstractmethod
    def action_format(self) -> str: ...

    # --- Core interface ---
    @abstractmethod
    def reset(self, seed: int, **kwargs) -> str:
        """Reset and return initial observation string."""
        self.rng = np.random.default_rng(seed)
        self._done = False
        self._success = False
        self.trajectory = []
        return ""

    @abstractmethod
    def step(self, raw_action: str) -> StepResult:
        """Process one action."""
        ...

    @abstractmethod
    def render_observation(self) -> str:
        """Render current state as text for LLM."""
        ...

    @abstractmethod
    def parse_action(self, raw_action: str) -> Dict[str, Any]:
        """Parse raw text into structured action dict.
        Returns dict with at minimum: {"type": "edit"|"finalize"|"invalid", ...}
        """
        ...

    @abstractmethod
    def validate_action(self, parsed: Dict[str, Any]) -> Tuple[bool, List[Any]]:
        """Check if parsed action is valid. Returns (valid, events)."""
        ...

    @abstractmethod
    def apply_edit(self, parsed: Dict[str, Any]) -> List[Any]:
        """Apply edit to state. Returns failure events."""
        ...

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Full evaluation. Returns results dict including 'success' key."""
        ...

    @abstractmethod
    def get_episode_summary(self) -> Dict[str, Any]:
        """Return episode metrics for logging."""
        ...

    # --- Common helpers ---
    @property
    def current_step(self) -> int:
        return self.state.t

    def is_done(self) -> bool:
        return self._done

    def is_success(self) -> bool:
        return self._success

    def get_trajectory(self) -> list:
        return list(self.trajectory)

    def record_step(self, record):
        self.trajectory.append(record)


BaseEnv = BaseConstructiveEditEnv
