#!/usr/bin/env python3
"""Attempt exact recoverability auditing for visited Alloy trajectory states.

This audit is exact only for the implemented, rounded Alloy environment state
graph. It is not a proof about continuous physical alloy design. States that
exceed the expansion cap are reported as unknown rather than forced into a
binary label.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.alloy import (
    AlloyEnv,
    COARSE_DELTAS,
    DENSITY_TABLE,
    ELEMENTS,
    FINE_DELTAS,
    _GOOD_REGION_CENTER,
)
from benchmark_v4.runners.run_soundness_audits import (
    SURFACE_EVENTS,
    iter_raw_files,
    parse_alloy_comp,
)


CompTuple = Tuple[float, ...]


class ExpansionCapExceeded(RuntimeError):
    pass


class SharedExactOracle:
    """Exact implemented-state recoverability oracle with cross-query memoization.

    The Alloy configuration used in the paper has ``overshoot_penalty=0``.
    Therefore the terminal predicate depends on composition and recovery cost,
    but not on the historical maximum density. This lets us share the dynamic
    program across visited trajectory states. If that environment setting ever
    changes, this oracle should be extended to include max-density history in
    the memo key.
    """

    def __init__(self, env: AlloyEnv, max_expansions_per_state: int):
        if float(env._overshoot_penalty) != 0.0:
            raise ValueError(
                "SharedExactOracle assumes AlloyEnv overshoot_penalty=0; "
                "include max density in the state key before using it otherwise."
            )
        self.env = env
        self.max_expansions_per_state = int(max_expansions_per_state)
        self._active_expansions = 0

        self.element_bounds = tuple(
            env._element_bounds.get(el, (0.0, 100.0)) for el in ELEMENTS
        )
        self.density_weights = tuple(DENSITY_TABLE[el] / 100.0 for el in ELEMENTS)
        self.good_center = tuple(float(_GOOD_REGION_CENTER.get(el, 0.0)) for el in ELEMENTS)
        self.heavy_items = tuple(
            (ELEMENTS.index(el), float(weight))
            for el, weight in env._heavy_weights.items()
            if el in ELEMENTS
        )
        self.heavy_floor = float(env._heavy_floor)
        self.heavy_threshold = float(env._heavy_threshold)
        self.uts_target = float(env._uts_target)
        self.density_target = float(env._density_target)
        self.rc_threshold = float(env._rc_threshold)
        self.rc_rate = float(env._rc_rate)

    @lru_cache(maxsize=None)
    def density_of(self, comp_t: CompTuple) -> float:
        return sum(float(comp_t[i]) * self.density_weights[i] for i in range(len(ELEMENTS)))

    @lru_cache(maxsize=None)
    def distance_from_good(self, comp_t: CompTuple) -> float:
        return sum(
            (float(comp_t[i]) - self.good_center[i]) ** 2
            for i in range(len(ELEMENTS))
        ) ** 0.5

    @lru_cache(maxsize=None)
    def uts_of(self, comp_t: CompTuple) -> float:
        base = float(self.env._uts_model.predict([[float(x) for x in comp_t]])[0])
        heavy_pct = sum(float(comp_t[idx]) * weight for idx, weight in self.heavy_items)
        scale = self.heavy_floor + (1.0 - self.heavy_floor) * min(
            1.0, heavy_pct / self.heavy_threshold
        )
        return base * scale

    @lru_cache(maxsize=None)
    def valid_edits_for(self, comp_t: CompTuple, cur_t: int) -> Tuple[Tuple[int, int, int], ...]:
        deltas = COARSE_DELTAS if cur_t < self.env.max_steps // 2 else FINE_DELTAS
        edits = []
        for inc_idx in range(len(ELEMENTS)):
            for dec_idx in range(len(ELEMENTS)):
                if inc_idx == dec_idx:
                    continue
                inc_lo, inc_hi = self.element_bounds[inc_idx]
                dec_lo, _dec_hi = self.element_bounds[dec_idx]
                for delta in deltas:
                    new_inc = float(comp_t[inc_idx]) + delta
                    new_dec = float(comp_t[dec_idx]) - delta
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    if new_inc <= inc_hi and new_dec >= dec_lo:
                        edits.append((inc_idx, dec_idx, int(delta)))
        return tuple(edits)

    def apply_edit_tuple(
        self, comp_t: CompTuple, rc: float, edit: Tuple[int, int, int]
    ) -> Tuple[CompTuple, float]:
        inc_idx, dec_idx, delta = edit
        nxt = list(comp_t)
        nxt[inc_idx] = round(float(nxt[inc_idx]) + delta, 3)
        nxt[dec_idx] = round(float(nxt[dec_idx]) - delta, 3)
        nxt_t = tuple(nxt)
        prev_distance = self.distance_from_good(comp_t)
        new_distance = self.distance_from_good(nxt_t)
        nxt_rc = float(rc)
        if new_distance > prev_distance:
            nxt_rc += self.rc_rate * (new_distance - prev_distance)
        elif nxt_rc > 0:
            nxt_rc = max(0.0, nxt_rc - self.rc_rate * (prev_distance - new_distance) * 0.3)
        return nxt_t, round(float(nxt_rc), 4)

    def state_success_tuple(self, comp_t: CompTuple, rc: float) -> bool:
        return (
            self.uts_of(comp_t) >= self.uts_target
            and self.density_of(comp_t) <= self.density_target
            and float(rc) < self.rc_threshold
        )

    @lru_cache(maxsize=None)
    def rec(
        self,
        comp_t: CompTuple,
        rc_rounded: float,
        cur_t: int,
        budget_left: int,
    ) -> bool | None:
        self._active_expansions += 1
        if self._active_expansions > self.max_expansions_per_state:
            return None
        if self.state_success_tuple(comp_t, float(rc_rounded)):
            return True
        if budget_left <= 0:
            return False
        any_unknown = False
        for edit in self.valid_edits_for(comp_t, int(cur_t)):
            nxt_comp_t, nxt_rc = self.apply_edit_tuple(comp_t, float(rc_rounded), edit)
            child = self.rec(
                nxt_comp_t,
                round(float(nxt_rc), 4),
                int(cur_t) + 1,
                int(budget_left) - 1,
            )
            if child is True:
                return True
            if child is None:
                any_unknown = True
                if self._active_expansions >= self.max_expansions_per_state:
                    return None
        if any_unknown:
            return None
        return False

    def label(
        self,
        *,
        comp: Dict[str, float],
        recovery_cost: float,
        t: int,
        budget: int,
    ) -> Dict[str, Any]:
        start_comp = comp_tuple(comp)
        before_cache = self.rec.cache_info().currsize
        self._active_expansions = 0
        recoverable = self.rec(
            start_comp,
            round(float(recovery_cost), 4),
            int(t),
            int(budget),
        )
        if recoverable is None:
            return {
                "status": "unknown_cap",
                "recoverable": None,
                "expanded_states": self._active_expansions,
                "memo_states": self.rec.cache_info().currsize,
                "new_memo_states": self.rec.cache_info().currsize - before_cache,
            }
        return {
            "status": "exact",
            "recoverable": bool(recoverable),
            "expanded_states": self._active_expansions,
            "memo_states": self.rec.cache_info().currsize,
            "new_memo_states": self.rec.cache_info().currsize - before_cache,
        }


class FrontierExactOracle:
    """Exact implemented-state oracle using minimum-recovery-cost frontiers.

    Valid Alloy edits depend on composition and phase, not on path history
    except through recovery cost. For a fixed composition and time, lower
    recovery cost weakly dominates higher recovery cost for every continuation.
    The frontier therefore keeps only the minimum recovery cost per composition
    at each depth while preserving exactness over the rounded edit graph.
    """

    def __init__(self, env: AlloyEnv, max_expansions_per_state: int):
        if float(env._overshoot_penalty) != 0.0:
            raise ValueError(
                "FrontierExactOracle assumes AlloyEnv overshoot_penalty=0; "
                "include max density in the state key before using it otherwise."
            )
        self.env = env
        self.max_expansions_per_state = int(max_expansions_per_state)
        self.element_bounds = tuple(
            env._element_bounds.get(el, (0.0, 100.0)) for el in ELEMENTS
        )
        self.density_weights = tuple(DENSITY_TABLE[el] / 100.0 for el in ELEMENTS)
        self.good_center = tuple(float(_GOOD_REGION_CENTER.get(el, 0.0)) for el in ELEMENTS)
        self.heavy_items = tuple(
            (ELEMENTS.index(el), float(weight))
            for el, weight in env._heavy_weights.items()
            if el in ELEMENTS
        )
        self.heavy_floor = float(env._heavy_floor)
        self.heavy_threshold = float(env._heavy_threshold)
        self.uts_target = float(env._uts_target)
        self.density_target = float(env._density_target)
        self.rc_threshold = float(env._rc_threshold)
        self.rc_rate = float(env._rc_rate)

    @lru_cache(maxsize=None)
    def density_of(self, comp_t: CompTuple) -> float:
        return sum(float(comp_t[i]) * self.density_weights[i] for i in range(len(ELEMENTS)))

    @lru_cache(maxsize=None)
    def distance_from_good(self, comp_t: CompTuple) -> float:
        return sum(
            (float(comp_t[i]) - self.good_center[i]) ** 2
            for i in range(len(ELEMENTS))
        ) ** 0.5

    @lru_cache(maxsize=None)
    def uts_of(self, comp_t: CompTuple) -> float:
        base = float(self.env._uts_model.predict([[float(x) for x in comp_t]])[0])
        heavy_pct = sum(float(comp_t[idx]) * weight for idx, weight in self.heavy_items)
        scale = self.heavy_floor + (1.0 - self.heavy_floor) * min(
            1.0, heavy_pct / self.heavy_threshold
        )
        return base * scale

    @lru_cache(maxsize=None)
    def valid_edits_for(self, comp_t: CompTuple, cur_t: int) -> Tuple[Tuple[int, int, int], ...]:
        deltas = COARSE_DELTAS if cur_t < self.env.max_steps // 2 else FINE_DELTAS
        edits = []
        for inc_idx in range(len(ELEMENTS)):
            for dec_idx in range(len(ELEMENTS)):
                if inc_idx == dec_idx:
                    continue
                inc_lo, inc_hi = self.element_bounds[inc_idx]
                dec_lo, _dec_hi = self.element_bounds[dec_idx]
                for delta in deltas:
                    new_inc = float(comp_t[inc_idx]) + delta
                    new_dec = float(comp_t[dec_idx]) - delta
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    if new_inc <= inc_hi and new_dec >= dec_lo:
                        edits.append((inc_idx, dec_idx, int(delta)))
        return tuple(edits)

    def apply_edit_tuple(
        self, comp_t: CompTuple, rc: float, edit: Tuple[int, int, int]
    ) -> Tuple[CompTuple, float]:
        inc_idx, dec_idx, delta = edit
        nxt = list(comp_t)
        nxt[inc_idx] = round(float(nxt[inc_idx]) + delta, 3)
        nxt[dec_idx] = round(float(nxt[dec_idx]) - delta, 3)
        nxt_t = tuple(nxt)
        prev_distance = self.distance_from_good(comp_t)
        new_distance = self.distance_from_good(nxt_t)
        nxt_rc = float(rc)
        if new_distance > prev_distance:
            nxt_rc += self.rc_rate * (new_distance - prev_distance)
        elif nxt_rc > 0:
            nxt_rc = max(0.0, nxt_rc - self.rc_rate * (prev_distance - new_distance) * 0.3)
        return nxt_t, round(float(nxt_rc), 4)

    def state_success_tuple(self, comp_t: CompTuple, rc: float) -> bool:
        return (
            self.uts_of(comp_t) >= self.uts_target
            and self.density_of(comp_t) <= self.density_target
            and float(rc) < self.rc_threshold
        )

    def label(
        self,
        *,
        comp: Dict[str, float],
        recovery_cost: float,
        t: int,
        budget: int,
    ) -> Dict[str, Any]:
        frontier: Dict[CompTuple, float] = {
            comp_tuple(comp): round(float(recovery_cost), 4)
        }
        expanded = 0
        visited_states = 1
        max_frontier = 1

        for depth in range(int(budget) + 1):
            cur_t = int(t) + depth
            for comp_t, rc in frontier.items():
                if self.state_success_tuple(comp_t, float(rc)):
                    return {
                        "status": "exact",
                        "recoverable": True,
                        "expanded_states": expanded,
                        "memo_states": visited_states,
                        "max_frontier": max_frontier,
                    }
            if depth >= int(budget):
                return {
                    "status": "exact",
                    "recoverable": False,
                    "expanded_states": expanded,
                    "memo_states": visited_states,
                    "max_frontier": max_frontier,
                }

            next_frontier: Dict[CompTuple, float] = {}
            for comp_t, rc in frontier.items():
                expanded += 1
                if expanded > self.max_expansions_per_state:
                    return {
                        "status": "unknown_cap",
                        "recoverable": None,
                        "expanded_states": expanded,
                        "memo_states": visited_states,
                        "max_frontier": max_frontier,
                    }
                for edit in self.valid_edits_for(comp_t, cur_t):
                    nxt_comp_t, nxt_rc = self.apply_edit_tuple(comp_t, float(rc), edit)
                    old = next_frontier.get(nxt_comp_t)
                    if old is None or nxt_rc < old:
                        next_frontier[nxt_comp_t] = nxt_rc
            frontier = next_frontier
            visited_states += len(frontier)
            max_frontier = max(max_frontier, len(frontier))

        return {
            "status": "exact",
            "recoverable": False,
            "expanded_states": expanded,
            "memo_states": visited_states,
            "max_frontier": max_frontier,
        }


class ThresholdExactOracle:
    """Exact implemented-state oracle via recovery-cost thresholds.

    For the paper Alloy setting, terminal success depends on composition and
    recovery cost, and edit availability depends on composition and time. The
    recovery-cost update is monotone: if a path succeeds from cost ``r``, the
    same path also succeeds from any lower cost. We can therefore compute, for
    each composition/time/budget triple, the largest rounded recovery-cost
    value from which some successful continuation exists. This avoids a
    separate forward search for each visited trajectory state while preserving
    exactness over the rounded implemented edit graph.
    """

    def __init__(
        self,
        env: AlloyEnv,
        max_expansions_per_state: int,
        *,
        max_rc_units: int = 100000,
    ):
        if float(env._overshoot_penalty) != 0.0:
            raise ValueError(
                "ThresholdExactOracle assumes AlloyEnv overshoot_penalty=0; "
                "include max density in the state key before using it otherwise."
            )
        self.env = env
        self.max_expansions_per_state = int(max_expansions_per_state)
        self.max_rc_units = int(max_rc_units)
        self.element_bounds = tuple(
            env._element_bounds.get(el, (0.0, 100.0)) for el in ELEMENTS
        )
        self.density_weights = tuple(DENSITY_TABLE[el] / 100.0 for el in ELEMENTS)
        self.good_center = tuple(float(_GOOD_REGION_CENTER.get(el, 0.0)) for el in ELEMENTS)
        self.heavy_items = tuple(
            (ELEMENTS.index(el), float(weight))
            for el, weight in env._heavy_weights.items()
            if el in ELEMENTS
        )
        self.heavy_floor = float(env._heavy_floor)
        self.heavy_threshold = float(env._heavy_threshold)
        self.uts_target = float(env._uts_target)
        self.density_target = float(env._density_target)
        self.rc_threshold = float(env._rc_threshold)
        self.rc_rate = float(env._rc_rate)
        self._active_expansions = 0

    @lru_cache(maxsize=None)
    def density_of(self, comp_t: CompTuple) -> float:
        return sum(float(comp_t[i]) * self.density_weights[i] for i in range(len(ELEMENTS)))

    @lru_cache(maxsize=None)
    def distance_from_good(self, comp_t: CompTuple) -> float:
        return sum(
            (float(comp_t[i]) - self.good_center[i]) ** 2
            for i in range(len(ELEMENTS))
        ) ** 0.5

    @lru_cache(maxsize=None)
    def uts_of(self, comp_t: CompTuple) -> float:
        base = float(self.env._uts_model.predict([[float(x) for x in comp_t]])[0])
        heavy_pct = sum(float(comp_t[idx]) * weight for idx, weight in self.heavy_items)
        scale = self.heavy_floor + (1.0 - self.heavy_floor) * min(
            1.0, heavy_pct / self.heavy_threshold
        )
        return base * scale

    @lru_cache(maxsize=None)
    def valid_edits_for(self, comp_t: CompTuple, cur_t: int) -> Tuple[Tuple[int, int, int], ...]:
        deltas = COARSE_DELTAS if cur_t < self.env.max_steps // 2 else FINE_DELTAS
        edits = []
        for inc_idx in range(len(ELEMENTS)):
            for dec_idx in range(len(ELEMENTS)):
                if inc_idx == dec_idx:
                    continue
                inc_lo, inc_hi = self.element_bounds[inc_idx]
                dec_lo, _dec_hi = self.element_bounds[dec_idx]
                for delta in deltas:
                    new_inc = float(comp_t[inc_idx]) + delta
                    new_dec = float(comp_t[dec_idx]) - delta
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    if new_inc <= inc_hi and new_dec >= dec_lo:
                        edits.append((inc_idx, dec_idx, int(delta)))
        return tuple(edits)

    def apply_edit_comp(self, comp_t: CompTuple, edit: Tuple[int, int, int]) -> CompTuple:
        inc_idx, dec_idx, delta = edit
        nxt = list(comp_t)
        nxt[inc_idx] = round(float(nxt[inc_idx]) + delta, 3)
        nxt[dec_idx] = round(float(nxt[dec_idx]) - delta, 3)
        return tuple(nxt)

    def transition_rc_units(self, comp_t: CompTuple, nxt_t: CompTuple, rc_units: int) -> int:
        rc = float(rc_units) / 10000.0
        prev_distance = self.distance_from_good(comp_t)
        new_distance = self.distance_from_good(nxt_t)
        if new_distance > prev_distance:
            rc += self.rc_rate * (new_distance - prev_distance)
        elif rc > 0:
            rc = max(0.0, rc - self.rc_rate * (prev_distance - new_distance) * 0.3)
        return int(round(round(float(rc), 4) * 10000))

    def immediate_success_limit_units(self, comp_t: CompTuple) -> int:
        if not (
            self.uts_of(comp_t) >= self.uts_target
            and self.density_of(comp_t) <= self.density_target
        ):
            return -1
        # Strict condition: recovery_cost < rc_threshold on the rounded graph.
        return min(
            self.max_rc_units,
            max(-1, int(math.ceil(self.rc_threshold * 10000.0 - 1e-9)) - 1),
        )

    @lru_cache(maxsize=None)
    def inverse_transition_limit(
        self,
        comp_t: CompTuple,
        nxt_t: CompTuple,
        child_limit_units: int,
    ) -> int:
        if child_limit_units < 0:
            return -1
        if self.transition_rc_units(comp_t, nxt_t, 0) > child_limit_units:
            return -1
        lo = 0
        hi = self.max_rc_units
        if self.transition_rc_units(comp_t, nxt_t, hi) <= child_limit_units:
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.transition_rc_units(comp_t, nxt_t, mid) <= child_limit_units:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @lru_cache(maxsize=None)
    def limit_units(self, comp_t: CompTuple, cur_t: int, budget_left: int) -> int | None:
        self._active_expansions += 1
        if self._active_expansions > self.max_expansions_per_state:
            return None
        best = self.immediate_success_limit_units(comp_t)
        if int(budget_left) <= 0:
            return best
        any_unknown = False
        for edit in self.valid_edits_for(comp_t, int(cur_t)):
            nxt_t = self.apply_edit_comp(comp_t, edit)
            child_limit = self.limit_units(nxt_t, int(cur_t) + 1, int(budget_left) - 1)
            if child_limit is None:
                any_unknown = True
                if self._active_expansions >= self.max_expansions_per_state:
                    return None
                continue
            best = max(
                best,
                self.inverse_transition_limit(comp_t, nxt_t, int(child_limit)),
            )
            if best >= self.max_rc_units:
                return self.max_rc_units
        if any_unknown and best < 0:
            return None
        return best

    def label(
        self,
        *,
        comp: Dict[str, float],
        recovery_cost: float,
        t: int,
        budget: int,
    ) -> Dict[str, Any]:
        start_comp = comp_tuple(comp)
        before_cache = self.limit_units.cache_info().currsize
        self._active_expansions = 0
        limit = self.limit_units(start_comp, int(t), int(budget))
        if limit is None:
            return {
                "status": "unknown_cap",
                "recoverable": None,
                "expanded_states": self._active_expansions,
                "memo_states": self.limit_units.cache_info().currsize,
                "new_memo_states": self.limit_units.cache_info().currsize - before_cache,
            }
        rc_units = int(round(round(float(recovery_cost), 4) * 10000))
        return {
            "status": "exact",
            "recoverable": bool(rc_units <= int(limit)),
            "expanded_states": self._active_expansions,
            "memo_states": self.limit_units.cache_info().currsize,
            "new_memo_states": self.limit_units.cache_info().currsize - before_cache,
            "max_recoverable_rc": round(float(limit) / 10000.0, 4),
        }


def comp_tuple(comp: Dict[str, float]) -> CompTuple:
    return tuple(round(float(comp.get(el, 0.0)), 3) for el in ELEMENTS)


def comp_dict(comp: CompTuple) -> Dict[str, float]:
    return {el: float(comp[i]) for i, el in enumerate(ELEMENTS)}


def state_key(comp: Dict[str, float], recovery_cost: float, t: int, budget: int) -> Tuple[Any, ...]:
    return comp_tuple(comp) + (round(float(recovery_cost), 4), int(t), int(budget))


def collect_states_fast(
    paths: Iterable[str],
    max_states: int | None,
    *,
    sample_mode: str,
    sample_seed: int,
) -> Tuple[list[dict], list[dict]]:
    states_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    episodes = []
    stop = False
    for path in iter_raw_files(paths):
        if stop:
            break
        try:
            with open(path) as f:
                ep = json.load(f)
        except Exception:
            continue
        if ep.get("env_name") != "alloy":
            continue
        steps = ep.get("steps") or []
        events = [event for step in steps for event in (step.get("events") or [])]
        surface_clean = not any(event in SURFACE_EVENTS for event in events)
        episode_state_keys = []
        for step in steps:
            diag = (step.get("info") or {}).get("diagnostics") or {}
            for side, struct_key, t_offset in [
                ("before", "structure_before", -1),
                ("after", "structure_after", 0),
            ]:
                snap = diag.get(f"recoverability_{side}") or {}
                if not snap or struct_key not in step:
                    continue
                try:
                    comp = parse_alloy_comp(step[struct_key])
                except Exception:
                    continue
                budget = int(snap.get("remaining_budget", 0))
                t = max(0, int(step.get("t", 0)) + t_offset)
                rc = float(snap.get("recovery_cost", 0.0))
                key = state_key(comp, rc, t, budget)
                episode_state_keys.append(key)
                if key not in states_by_key:
                    states_by_key[key] = {
                        "key": key,
                        "comp": comp,
                        "recovery_cost": rc,
                        "t": t,
                        "budget": budget,
                        "probe_recoverable": bool(snap.get("recoverable")),
                        "probe_score": float(snap.get("recoverability_score", 0.0)),
                    }
                    if (
                        sample_mode == "first"
                        and max_states is not None
                        and len(states_by_key) >= max_states
                    ):
                        stop = True
                        break
            if stop:
                break
        episodes.append({
            "path": str(path),
            "success": bool(ep.get("success")),
            "surface_clean": surface_clean,
            "state_keys": episode_state_keys,
        })
    states = list(states_by_key.values())
    if sample_mode == "random" and max_states is not None and len(states) > max_states:
        rng = random.Random(int(sample_seed))
        states = rng.sample(states, int(max_states))
    return states, episodes


def exact_label_for_state(
    env: AlloyEnv,
    *,
    comp: Dict[str, float],
    recovery_cost: float,
    t: int,
    budget: int,
    max_expansions: int,
) -> Dict[str, Any]:
    """Return exact implemented-state recoverability unless the cap is hit."""
    expansions = {"n": 0}
    start_comp = comp_tuple(comp)
    element_bounds = tuple(env._element_bounds.get(el, (0.0, 100.0)) for el in ELEMENTS)
    density_weights = tuple(DENSITY_TABLE[el] / 100.0 for el in ELEMENTS)
    good_center = tuple(float(_GOOD_REGION_CENTER.get(el, 0.0)) for el in ELEMENTS)
    heavy_items = tuple(
        (ELEMENTS.index(el), float(weight))
        for el, weight in env._heavy_weights.items()
        if el in ELEMENTS
    )
    heavy_floor = float(env._heavy_floor)
    heavy_threshold = float(env._heavy_threshold)
    uts_target = float(env._uts_target)
    density_target = float(env._density_target)
    overshoot_penalty = float(env._overshoot_penalty)
    rc_threshold = float(env._rc_threshold)
    rc_rate = float(env._rc_rate)

    @lru_cache(maxsize=None)
    def density_of(comp_t: CompTuple) -> float:
        return sum(float(comp_t[i]) * density_weights[i] for i in range(len(ELEMENTS)))

    @lru_cache(maxsize=None)
    def distance_from_good(comp_t: CompTuple) -> float:
        return sum(
            (float(comp_t[i]) - good_center[i]) ** 2
            for i in range(len(ELEMENTS))
        ) ** 0.5

    @lru_cache(maxsize=None)
    def uts_of(comp_t: CompTuple) -> float:
        base = float(env._uts_model.predict([[float(x) for x in comp_t]])[0])
        heavy_pct = sum(float(comp_t[idx]) * weight for idx, weight in heavy_items)
        scale = heavy_floor + (1.0 - heavy_floor) * min(1.0, heavy_pct / heavy_threshold)
        return base * scale

    @lru_cache(maxsize=None)
    def valid_edits_for(comp_t: CompTuple, cur_t: int) -> Tuple[Tuple[int, int, int], ...]:
        deltas = COARSE_DELTAS if cur_t < env.max_steps // 2 else FINE_DELTAS
        edits = []
        for inc_idx in range(len(ELEMENTS)):
            for dec_idx in range(len(ELEMENTS)):
                if inc_idx == dec_idx:
                    continue
                inc_lo, inc_hi = element_bounds[inc_idx]
                dec_lo, _dec_hi = element_bounds[dec_idx]
                for delta in deltas:
                    new_inc = float(comp_t[inc_idx]) + delta
                    new_dec = float(comp_t[dec_idx]) - delta
                    if new_inc > 100.0 or new_dec < 0.0:
                        continue
                    if new_inc <= inc_hi and new_dec >= dec_lo:
                        edits.append((inc_idx, dec_idx, int(delta)))
        return tuple(edits)

    def apply_edit_tuple(comp_t: CompTuple, rc: float, edit: Tuple[int, int, int]) -> Tuple[CompTuple, float]:
        inc_idx, dec_idx, delta = edit
        nxt = list(comp_t)
        nxt[inc_idx] = round(float(nxt[inc_idx]) + delta, 3)
        nxt[dec_idx] = round(float(nxt[dec_idx]) - delta, 3)
        nxt_t = tuple(nxt)
        prev_distance = distance_from_good(comp_t)
        new_distance = distance_from_good(nxt_t)
        nxt_rc = float(rc)
        if new_distance > prev_distance:
            nxt_rc += rc_rate * (new_distance - prev_distance)
        elif nxt_rc > 0:
            nxt_rc = max(0.0, nxt_rc - rc_rate * (prev_distance - new_distance) * 0.3)
        return nxt_t, round(float(nxt_rc), 4)

    def state_success_tuple(comp_t: CompTuple, rc: float, max_density_seen: float) -> bool:
        true_density = density_of(comp_t)
        max_density = max(float(max_density_seen), true_density)
        overshoot = max(0.0, max_density - density_target)
        effective_uts_target = uts_target * (1.0 + overshoot * overshoot_penalty)
        return (
            uts_of(comp_t) >= effective_uts_target
            and true_density <= density_target
            and float(rc) < rc_threshold
        )

    start_density = round(density_of(start_comp), 4)

    @lru_cache(maxsize=None)
    def rec(
        comp_t: CompTuple,
        rc_rounded: float,
        cur_t: int,
        budget_left: int,
        max_density_seen: float,
    ) -> bool:
        expansions["n"] += 1
        if expansions["n"] > max_expansions:
            raise ExpansionCapExceeded
        cur_max_density = round(max(float(max_density_seen), density_of(comp_t)), 4)
        if state_success_tuple(comp_t, float(rc_rounded), cur_max_density):
            return True
        if budget_left <= 0:
            return False
        for edit in valid_edits_for(comp_t, int(cur_t)):
            nxt_comp_t, nxt_rc = apply_edit_tuple(comp_t, float(rc_rounded), edit)
            nxt_density = max(cur_max_density, density_of(nxt_comp_t))
            if rec(
                nxt_comp_t,
                round(float(nxt_rc), 4),
                int(cur_t) + 1,
                int(budget_left) - 1,
                round(float(nxt_density), 4),
            ):
                return True
        return False

    try:
        recoverable = rec(
            start_comp,
            round(float(recovery_cost), 4),
            int(t),
            int(budget),
            start_density,
        )
        return {
            "status": "exact",
            "recoverable": bool(recoverable),
            "expanded_states": expansions["n"],
            "memo_states": rec.cache_info().currsize,
        }
    except ExpansionCapExceeded:
        return {
            "status": "unknown_cap",
            "recoverable": None,
            "expanded_states": expansions["n"],
            "memo_states": rec.cache_info().currsize,
        }


def audit(
    paths: Iterable[str],
    *,
    max_states: int | None,
    min_budget: int | None,
    max_budget: int | None,
    max_expansions_per_state: int,
    sample_mode: str,
    sample_seed: int,
    shared_cache: bool,
    method: str,
) -> Dict[str, Any]:
    env = AlloyEnv()
    if method == "threshold":
        oracle = ThresholdExactOracle(env, max_expansions_per_state)
    elif method == "frontier":
        oracle = FrontierExactOracle(env, max_expansions_per_state)
    elif shared_cache:
        oracle = SharedExactOracle(env, max_expansions_per_state)
    else:
        oracle = None
    states, episodes = collect_states_fast(
        paths,
        max_states=max_states,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    states_before_budget_filter = len(states)
    if min_budget is not None:
        states = [st for st in states if int(st["budget"]) >= int(min_budget)]
    if max_budget is not None:
        states = [st for st in states if int(st["budget"]) <= int(max_budget)]
    labels: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    exact = 0
    unknown = 0
    recoverable = 0
    agreement = 0

    for idx, st in enumerate(states, start=1):
        key = state_key(st["comp"], st["recovery_cost"], st["t"], st["budget"])
        if oracle is not None:
            result = oracle.label(
                comp=st["comp"],
                recovery_cost=st["recovery_cost"],
                t=st["t"],
                budget=st["budget"],
            )
        else:
            result = exact_label_for_state(
                env,
                comp=st["comp"],
                recovery_cost=st["recovery_cost"],
                t=st["t"],
                budget=st["budget"],
                max_expansions=max_expansions_per_state,
            )
        result.update({
            "index": idx,
            "key": key,
            "probe_recoverable": bool(st["probe_recoverable"]),
            "probe_score": st["probe_score"],
        })
        labels[key] = result
        if result["status"] == "exact":
            exact += 1
            recoverable += int(bool(result["recoverable"]))
            agreement += int(bool(result["recoverable"]) == bool(st["probe_recoverable"]))
        else:
            unknown += 1

    surface_clean_failures = 0
    known_surface_clean_failures = 0
    exact_loss_episodes = 0
    unknown_loss_episodes = 0
    for ep in episodes:
        if not (ep["surface_clean"] and not ep["success"]):
            continue
        surface_clean_failures += 1
        keys = [k for k in ep["state_keys"] if k in labels]
        known_keys = [k for k in keys if labels[k]["status"] == "exact"]
        if len(known_keys) >= 2:
            known_surface_clean_failures += 1
            has_loss = any(
                labels[a]["recoverable"] is True and labels[b]["recoverable"] is False
                for a, b in zip(known_keys, known_keys[1:])
            )
            exact_loss_episodes += int(has_loss)
        elif keys:
            unknown_loss_episodes += 1

    return {
        "note": (
            "Exact labels are over the implemented rounded Alloy edit graph and "
            "stored trajectory states, not continuous-state materials feasibility."
        ),
        "max_states": max_states,
        "min_budget": min_budget,
        "max_budget": max_budget,
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "max_expansions_per_state": max_expansions_per_state,
        "shared_cache": shared_cache,
        "method": method,
        "states_before_budget_filter": states_before_budget_filter,
        "states_audited": len(states),
        "exact_labeled_states": exact,
        "unknown_cap_states": unknown,
        "exact_recoverable_states": recoverable,
        "exact_recoverable_rate_known": recoverable / exact if exact else None,
        "agreement_with_probe_known": agreement / exact if exact else None,
        "surface_clean_failures": surface_clean_failures,
        "known_surface_clean_failures": known_surface_clean_failures,
        "surface_clean_failures_with_exact_loss": exact_loss_episodes,
        "surface_clean_exact_loss_rate_known": (
            exact_loss_episodes / known_surface_clean_failures
            if known_surface_clean_failures else None
        ),
        "surface_clean_failures_with_unknown_labels": unknown_loss_episodes,
        "state_results": list(labels.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Raw JSON file, run directory, or parent directory.")
    ap.add_argument("--output", default="results_v4/alloy_exact_recoverability_audit.json")
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--min-budget", type=int, default=None)
    ap.add_argument("--max-budget", type=int, default=None)
    ap.add_argument("--sample-mode", choices=["first", "random"], default="first")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-expansions-per-state", type=int, default=250000)
    ap.add_argument(
        "--method",
        choices=["frontier", "recursive", "threshold"],
        default="recursive",
        help=(
            "threshold computes maximum recoverable recovery-cost thresholds; "
            "frontier keeps the minimum recovery cost per composition/time; "
            "recursive uses the original depth-first oracle."
        ),
    )
    ap.add_argument(
        "--no-shared-cache",
        action="store_true",
        help="Disable cross-query memoization and use the original per-state oracle.",
    )
    args = ap.parse_args()

    result = audit(
        args.inputs,
        max_states=args.max_states,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        max_expansions_per_state=args.max_expansions_per_state,
        shared_cache=not args.no_shared_cache,
        method=args.method,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "state_results"}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
