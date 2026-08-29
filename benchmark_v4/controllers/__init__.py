"""Controller wrappers for constructive-editing experiments."""
from benchmark_v4.controllers.base import BaseController, GreedyController
from benchmark_v4.controllers.sampling import (
    BacktrackingController,
    ReflexionRetryController,
    SelfConsistencyController,
)
from benchmark_v4.controllers.search import LookaheadBeamController


def get_controller(name: str, **kwargs) -> BaseController:
    if name == "greedy":
        return GreedyController()
    if name == "greedy_sampled":
        return GreedyController(condition_name="greedy_sampled")
    if name == "self_consistency":
        return SelfConsistencyController(
            k=int(kwargs.get("k", 5)),
            scorer=str(kwargs.get("scorer", "proxy")),
        )
    if name == "reflexion_retry":
        return ReflexionRetryController(
            scorer=str(kwargs.get("scorer", "text_visible")),
        )
    if name == "backtracking":
        name = "loop_avoidant"
    if name == "loop_avoidant":
        return BacktrackingController(
            k=int(kwargs.get("k", 5)),
            scorer=str(kwargs.get("scorer", "proxy")),
        )
    if name == "beam":
        return LookaheadBeamController(
            width=int(kwargs.get("width", 3)),
            depth=int(kwargs.get("depth", 2)),
            samples_per_node=int(kwargs.get("samples_per_node", kwargs.get("k", 3))),
            scorer=str(kwargs.get("scorer", "proxy")),
        )
    if name == "tot_style_beam":
        return LookaheadBeamController(
            width=int(kwargs.get("width", 2)),
            depth=int(kwargs.get("depth", 2)),
            samples_per_node=int(kwargs.get("samples_per_node", kwargs.get("k", 3))),
            scorer=str(kwargs.get("scorer", "text_visible")),
            condition_name="tot_style_beam",
        )
    raise ValueError(f"Unknown controller: {name}")
