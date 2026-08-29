"""Failure taxonomy for the constructive-editing benchmark.

Classifies step-level failure events into four categories and provides
dataclasses for recording steps, edit traces, and episode summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

class FailureCategory(Enum):
    SURFACE = auto()
    EDITING_PROCESS = auto()
    STRUCTURE = auto()
    ENV_SPECIFIC = auto()


# ---------------------------------------------------------------------------
# Failure events
# ---------------------------------------------------------------------------

class FailureEvent(Enum):
    # Surface / syntax failures
    MALFORMED_ACTION = auto()
    ILLEGAL_EDIT = auto()
    INVALID_POSITION = auto()
    INVALID_VALUE = auto()
    REPEATED_EXACT_EDIT = auto()

    # Editing-process failures
    OSCILLATION = auto()
    UNDO_LOOP = auto()
    BUDGET_UNAWARE_ACTION = auto()
    PREMATURE_FINALIZE = auto()

    # Structure-level failures
    LOCAL_OPTIMUM_TRAP = auto()
    GLOBAL_FEASIBILITY_LOSS = auto()
    HARD_CONSTRAINT_VIOLATION = auto()
    OBJECTIVE_TRADEOFF_FAILURE = auto()
    RECOVERY_COST_EXPLOSION = auto()

    # Environment-specific (additional tags)
    INVALID_WORD = auto()
    MASS_BALANCE_VIOLATION = auto()
    MOTIF_BREAK = auto()
    SYNERGY_MISSED = auto()


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

FAILURE_CATEGORY_MAP: Dict[FailureEvent, FailureCategory] = {
    # Surface
    FailureEvent.MALFORMED_ACTION: FailureCategory.SURFACE,
    FailureEvent.ILLEGAL_EDIT: FailureCategory.SURFACE,
    FailureEvent.INVALID_POSITION: FailureCategory.SURFACE,
    FailureEvent.INVALID_VALUE: FailureCategory.SURFACE,
    FailureEvent.REPEATED_EXACT_EDIT: FailureCategory.SURFACE,
    # Editing process
    FailureEvent.OSCILLATION: FailureCategory.EDITING_PROCESS,
    FailureEvent.UNDO_LOOP: FailureCategory.EDITING_PROCESS,
    FailureEvent.BUDGET_UNAWARE_ACTION: FailureCategory.EDITING_PROCESS,
    FailureEvent.PREMATURE_FINALIZE: FailureCategory.EDITING_PROCESS,
    # Structure
    FailureEvent.LOCAL_OPTIMUM_TRAP: FailureCategory.STRUCTURE,
    FailureEvent.GLOBAL_FEASIBILITY_LOSS: FailureCategory.STRUCTURE,
    FailureEvent.HARD_CONSTRAINT_VIOLATION: FailureCategory.STRUCTURE,
    FailureEvent.OBJECTIVE_TRADEOFF_FAILURE: FailureCategory.STRUCTURE,
    FailureEvent.RECOVERY_COST_EXPLOSION: FailureCategory.STRUCTURE,
    # Environment-specific
    FailureEvent.INVALID_WORD: FailureCategory.ENV_SPECIFIC,
    FailureEvent.MASS_BALANCE_VIOLATION: FailureCategory.ENV_SPECIFIC,
    FailureEvent.MOTIF_BREAK: FailureCategory.ENV_SPECIFIC,
    FailureEvent.SYNERGY_MISSED: FailureCategory.ENV_SPECIFIC,
}

# Severity order used by get_dominant_failure (highest first).
_CATEGORY_SEVERITY: List[FailureCategory] = [
    FailureCategory.STRUCTURE,
    FailureCategory.EDITING_PROCESS,
    FailureCategory.SURFACE,
    FailureCategory.ENV_SPECIFIC,
]


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Record for a single environment step."""

    t: int
    observation: Any
    raw_action: str
    parsed_action: Optional[Any]
    valid: bool
    events: List[FailureEvent]
    reward: float
    info: Dict[str, Any]
    structure_before: Any
    structure_after: Any


@dataclass
class EditTrace:
    """Lightweight trace entry focused on the edit itself."""

    step: int
    raw_action: str
    parsed_action: Optional[Any]
    valid: bool
    events: List[FailureEvent]
    estimated_feedback: Optional[float]
    true_feedback: Optional[float]


@dataclass
class EpisodeSummary:
    """Aggregate summary for a full episode."""

    env_name: str
    regime: str
    model: str
    prompt_condition: str
    memory_condition: str
    seed: int
    steps: List[StepRecord]
    success: bool
    terminal_failure: Optional[FailureEvent]
    failure_events: List[FailureEvent]
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    # Detailed failure analysis fields
    event_counts: Dict[str, int] = field(default_factory=dict)
    finalize_step: Optional[int] = None
    first_failure_step: Optional[int] = None
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dominant_failure(events: List[FailureEvent]) -> Optional[FailureEvent]:
    """Return the most severe failure event from *events*.

    Severity order: STRUCTURE > EDITING_PROCESS > SURFACE > ENV_SPECIFIC.
    Within a category the first event encountered wins.  Returns ``None``
    when *events* is empty.
    """
    if not events:
        return None
    for category in _CATEGORY_SEVERITY:
        for event in events:
            if FAILURE_CATEGORY_MAP[event] is category:
                return event
    return None
