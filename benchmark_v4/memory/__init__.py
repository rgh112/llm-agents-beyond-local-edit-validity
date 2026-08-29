"""Memory modules for constructive editing experiments."""

from benchmark_v4.memory.state_only import StateOnlyMemory
from benchmark_v4.memory.window_memory import WindowMemory
from benchmark_v4.memory.summary_memory import SummaryMemory
from benchmark_v4.memory.full_history import FullHistoryMemory
from benchmark_v4.memory.best_state_memory import BestStateMemory
from benchmark_v4.memory.history_controls import (
    MisleadingHistoryMemory,
    RandomizedHistoryMemory,
)


MEMORY_MODULES = {
    "state_only": StateOnlyMemory,
    "M0": StateOnlyMemory,
    "window_1": lambda: WindowMemory(window_size=1),
    "M1": lambda: WindowMemory(window_size=1),
    "window_3": lambda: WindowMemory(window_size=3),
    "M3": lambda: WindowMemory(window_size=3),
    "summary": SummaryMemory,
    "full_history": FullHistoryMemory,
    "Mfull": FullHistoryMemory,
    "best_state": BestStateMemory,
    "Mbest": BestStateMemory,
    "randomized_history": lambda: RandomizedHistoryMemory(window_size=3),
    "randomized_history_3": lambda: RandomizedHistoryMemory(window_size=3),
    "Mrand": lambda: RandomizedHistoryMemory(window_size=3),
    "misleading_history": lambda: MisleadingHistoryMemory(window_size=3),
    "misleading_history_3": lambda: MisleadingHistoryMemory(window_size=3),
    "Mmislead": lambda: MisleadingHistoryMemory(window_size=3),
}


def get_memory_module(name: str):
    if name not in MEMORY_MODULES:
        raise ValueError(
            f"Unknown memory module {name!r}; choose from {sorted(MEMORY_MODULES)}"
        )
    factory = MEMORY_MODULES[name]
    return factory()


__all__ = [
    "StateOnlyMemory",
    "WindowMemory",
    "SummaryMemory",
    "FullHistoryMemory",
    "BestStateMemory",
    "RandomizedHistoryMemory",
    "MisleadingHistoryMemory",
    "get_memory_module",
]
