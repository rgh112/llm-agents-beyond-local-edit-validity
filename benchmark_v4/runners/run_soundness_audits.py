#!/usr/bin/env python3
"""Soundness audits for the constructive-editing diagnostic benchmark.

Adds three reviewer-facing checks without LLM calls:
  1. Alloy trajectory-state near-exact feasibility audit via beam-size sweep.
  2. Non-LLM visible/privileged beam planner baselines for all environments.
  3. Episode-level stratified bootstrap confidence intervals from raw logs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from benchmark_v4.envs.alloy import AlloyEnv, ELEMENTS
from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv
from benchmark_v4.envs.word_ladder import WordLadderEnv

SURFACE_EVENTS = {
    "MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
    "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT",
}


def iter_raw_files(paths: Iterable[str]) -> Iterable[Path]:
    for item in paths:
        p = Path(item)
        if p.is_file() and p.suffix == ".json":
            yield p
        elif (p / "raw").is_dir():
            yield from sorted((p / "raw").glob("*.json"))
        elif p.is_dir():
            yield from sorted(p.glob("**/raw/*.json"))


def parse_alloy_comp(text: str) -> Dict[str, float]:
    comp: Dict[str, float] = {}
    for el, val in re.findall(r"\b(Fe|Ni|Cr|Co|Mn|Mo):\s*([0-9.]+)%", text):
        comp[el] = float(val)
    missing = [el for el in ELEMENTS if el not in comp]
    if missing:
        raise ValueError(f"could not parse alloy composition from: {text!r}")
    return comp


def alloy_state_key(comp: Dict[str, float], rc: float, t: int, budget: int) -> Tuple[Any, ...]:
    return tuple(round(comp[el], 3) for el in ELEMENTS) + (round(float(rc), 4), int(t), int(budget))


def alloy_beam_feasible(env: AlloyEnv, comp: Dict[str, float], rc: float, t: int, budget: int,
                        beam_width: int, scorer: str = "visible") -> Dict[str, Any]:
    """Beam feasibility over the fixed Alloy edit grid from an arbitrary state."""
    density_traj = [env._calculate_density(comp)]
    if env._state_success(comp, rc, density_traj):
        return {"recoverable": True, "best_score": env._local_proxy_score(comp, rc), "depth_found": 0}
    frontier = [(env._local_proxy_score(comp, rc), dict(comp), float(rc), density_traj, int(t))]
    best_score = frontier[0][0]
    best_success_score = -1e9
    for depth in range(1, min(max(0, int(budget)), 4) + 1):
        candidates = []
        for _, cur_comp, cur_rc, cur_density_traj, cur_t in frontier:
            for edit in env._simulate_valid_edits(cur_comp, cur_t):
                nxt_comp, nxt_rc, nxt_density_traj = env._simulate_edit_state(
                    cur_comp, cur_rc, cur_density_traj, edit
                )
                proxy = env._local_proxy_score(nxt_comp, nxt_rc)
                if scorer == "privileged":
                    uts = env._predict_uts(nxt_comp)
                    dens = env._calculate_density(nxt_comp)
                    score = (
                        min(0.0, (uts - env._uts_target) / env._uts_target)
                        - max(0.0, dens - env._density_target)
                        - 0.25 * max(0.0, nxt_rc - env._rc_threshold)
                        + 0.1 * proxy
                    )
                else:
                    score = proxy
                best_score = max(best_score, score)
                if env._state_success(nxt_comp, nxt_rc, nxt_density_traj):
                    best_success_score = max(best_success_score, score)
                    return {"recoverable": True, "best_score": round(best_score, 4), "depth_found": depth}
                candidates.append((score, nxt_comp, nxt_rc, nxt_density_traj, cur_t + 1))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        frontier = candidates[:beam_width]
    return {"recoverable": False, "best_score": round(max(best_score, best_success_score), 4), "depth_found": None}


def collect_alloy_states(paths: Iterable[str], max_states: int | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    states_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    episodes: List[Dict[str, Any]] = []
    for path in iter_raw_files(paths):
        try:
            ep = json.load(open(path))
        except Exception:
            continue
        if ep.get("env_name") != "alloy":
            continue
        steps = ep.get("steps") or []
        events = [e for st in steps for e in (st.get("events") or [])]
        surface_clean = not any(e in SURFACE_EVENTS for e in events)
        episode_state_keys: List[Tuple[Any, ...]] = []
        for st in steps:
            diag = (st.get("info") or {}).get("diagnostics") or {}
            for side, struct_key, t_offset in [("before", "structure_before", -1), ("after", "structure_after", 0)]:
                snap = diag.get(f"recoverability_{side}") or {}
                if not snap or struct_key not in st:
                    continue
                try:
                    comp = parse_alloy_comp(st[struct_key])
                except Exception:
                    continue
                budget = int(snap.get("remaining_budget", 0))
                t = max(0, int(st.get("t", 0)) + t_offset)
                rc = float(snap.get("recovery_cost", 0.0))
                key = alloy_state_key(comp, rc, t, budget)
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
                if max_states and len(states_by_key) >= max_states:
                    break
            if max_states and len(states_by_key) >= max_states:
                break
        episodes.append({
            "path": str(path),
            "success": bool(ep.get("success")),
            "surface_clean": surface_clean,
            "state_keys": episode_state_keys,
        })
        if max_states and len(states_by_key) >= max_states:
            # Continue reading episodes is not useful once state audit is capped.
            pass
    return list(states_by_key.values()), episodes


def audit_alloy(paths: Iterable[str], widths: List[int], max_states: int | None = None) -> Dict[str, Any]:
    env = AlloyEnv()
    states, episodes = collect_alloy_states(paths, max_states=max_states)
    by_width: Dict[str, Dict[str, Any]] = {}
    labels: Dict[int, Dict[Tuple[Any, ...], bool]] = {}
    for width in widths:
        recoverable = 0
        agreement = 0
        labels[width] = {}
        for st in states:
            res = alloy_beam_feasible(env, st["comp"], st["recovery_cost"], st["t"], st["budget"], width, scorer="visible")
            rec = bool(res["recoverable"])
            labels[width][st["key"]] = rec
            recoverable += int(rec)
            agreement += int(rec == bool(st["probe_recoverable"]))
        by_width[str(width)] = {
            "states_audited": len(states),
            "recoverable_states": recoverable,
            "recoverable_rate": recoverable / len(states) if states else None,
            "agreement_with_original_probe": agreement / len(states) if states else None,
        }
    largest = max(widths)
    smallest = min(widths)
    stable = sum(1 for st in states if labels[smallest].get(st["key"]) == labels[largest].get(st["key"]))
    surface_clean_failures = 0
    surface_clean_failures_with_loss = 0
    loss_episodes = 0
    for ep in episodes:
        has_loss = False
        keys = [k for k in ep["state_keys"] if k in labels[largest]]
        for a, b in zip(keys, keys[1:]):
            if labels[largest].get(a) and not labels[largest].get(b):
                has_loss = True
                break
        loss_episodes += int(has_loss)
        if ep["surface_clean"] and not ep["success"]:
            surface_clean_failures += 1
            surface_clean_failures_with_loss += int(has_loss)
    return {
        "beam_widths": widths,
        "states_audited": len(states),
        "episodes_audited": len(episodes),
        "by_width": by_width,
        "small_large_label_stability": stable / len(states) if states else None,
        "surface_clean_failures": surface_clean_failures,
        "surface_clean_failures_with_large_beam_loss": surface_clean_failures_with_loss,
        "surface_clean_failure_large_beam_loss_rate": (
            surface_clean_failures_with_loss / surface_clean_failures if surface_clean_failures else None
        ),
        "note": "Alloy audit is over unique states visited in supplied raw logs, not the full continuous composition space.",
    }



_ENV_CACHE: Dict[str, Any] = {}


def cached_env(name: str):
    if name not in _ENV_CACHE:
        if name == "word_ladder":
            _ENV_CACHE[name] = WordLadderEnv(data_dir="wordladder_data")
        elif name == "alloy":
            _ENV_CACHE[name] = AlloyEnv()
        elif name == "gb1_sequence":
            _ENV_CACHE[name] = GB1SequenceEnv()
        else:
            raise ValueError(name)
    return _ENV_CACHE[name]


def run_word_planner(seed: int, scorer: str, width: int) -> Dict[str, Any]:
    env = cached_env("word_ladder")
    env.reset(seed)
    start = env.state.structure
    target = env.state.latent["target_word"]
    if scorer == "privileged":
        try:
            path = env.graph.shortest_path(start, target)  # type: ignore[attr-defined]
        except Exception:
            import networkx as nx
            path = nx.shortest_path(env.graph, start, target)
        return {"success": len(path) - 1 <= env.max_steps, "steps": len(path) - 1}
    # Visible greedy beam using target Hamming distance and no-revisit state.
    frontier = [(-sum(a != b for a, b in zip(start, target)), start, frozenset([start]), 0)]
    for _ in range(env.max_steps):
        cand = []
        for _, word, used, steps in frontier:
            if word == target:
                return {"success": True, "steps": steps}
            for pos in range(len(word)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    if c == word[pos]:
                        continue
                    nxt = word[:pos] + c + word[pos+1:]
                    if nxt in env.vocab and nxt not in used:
                        score = -sum(a != b for a, b in zip(nxt, target))
                        cand.append((score, nxt, used | frozenset([nxt]), steps + 1))
        if not cand:
            break
        cand.sort(key=lambda x: x[0], reverse=True)
        frontier = cand[:width]
    return {"success": any(word == target for _, word, _, _ in frontier), "steps": min((st for _, w, _, st in frontier if w == target), default=env.max_steps)}


def run_alloy_planner(seed: int, scorer: str, width: int) -> Dict[str, Any]:
    env = cached_env("alloy")
    env.reset(seed)
    comp0 = dict(env.state.structure)
    dens0 = [env._calculate_density(comp0)]
    frontier = [(env._local_proxy_score(comp0, 0.0), comp0, 0.0, dens0, 0)]
    for depth in range(env.max_steps + 1):
        cand = []
        for _, comp, rc, dens, t in frontier:
            if env._state_success(comp, rc, dens):
                return {"success": True, "steps": depth}
            if depth >= env.max_steps:
                continue
            for edit in env._simulate_valid_edits(comp, t):
                nxt_comp, nxt_rc, nxt_dens = env._simulate_edit_state(comp, rc, dens, edit)
                proxy = env._local_proxy_score(nxt_comp, nxt_rc)
                if scorer == "privileged":
                    # Privileged score uses terminal predicate components, not raw model hidden state.
                    uts = env._predict_uts(nxt_comp)
                    density = env._calculate_density(nxt_comp)
                    score = proxy - 2.0 * max(0.0, (env._uts_target - uts) / env._uts_target) - max(0.0, density - env._density_target) - 0.25 * max(0.0, nxt_rc - env._rc_threshold)
                else:
                    score = proxy
                cand.append((score, nxt_comp, nxt_rc, nxt_dens, t + 1))
        if not cand:
            break
        cand.sort(key=lambda x: x[0], reverse=True)
        frontier = cand[:width]
    return {"success": False, "steps": env.max_steps}


def run_gb1_planner(seed: int, scorer: str, width: int) -> Dict[str, Any]:
    env = cached_env("gb1_sequence")
    env.reset(seed)
    seq0 = tuple(env.state.structure)
    stab0 = float(env.state.latent["stability"])
    def score(seq: Tuple[str, ...], stab: float) -> float:
        seq_l = list(seq)
        if scorer == "privileged":
            return env._true_fitness(seq_l) + 0.5 * stab
        return env._estimated_fitness(seq_l) + 0.5 * stab
    frontier = [(score(seq0, stab0), seq0, stab0, 0)]
    seen = {(seq0, stab0, 0)}
    for depth in range(env.max_steps + 1):
        cand = []
        for _, seq, stab, steps in frontier:
            if env._success_for(list(seq), stab):
                return {"success": True, "steps": steps}
            if depth >= env.max_steps:
                continue
            for pos, aa in env._available_edits_for(list(seq), stab):
                nxt = list(seq)
                nxt[pos] = aa
                nxt_t = tuple(nxt)
                nxt_stab = env._next_stability(list(seq), stab, pos, aa)
                key = (nxt_t, nxt_stab, steps + 1)
                if key in seen:
                    continue
                seen.add(key)
                cand.append((score(nxt_t, nxt_stab), nxt_t, nxt_stab, steps + 1))
        if not cand:
            break
        cand.sort(key=lambda x: x[0], reverse=True)
        frontier = cand[:width]
    return {"success": False, "steps": env.max_steps}


def run_beam_episode(env_name: str, seed: int, scorer: str, beam_width: int) -> Dict[str, Any]:
    if env_name == "word_ladder":
        return run_word_planner(seed, scorer, beam_width)
    if env_name == "alloy":
        return run_alloy_planner(seed, scorer, beam_width)
    if env_name == "gb1_sequence":
        return run_gb1_planner(seed, scorer, beam_width)
    raise ValueError(env_name)


def run_planners(envs: List[str], seeds: List[int], widths: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for env_name in envs:
        out[env_name] = {}
        for scorer in ["visible", "privileged"]:
            out[env_name][scorer] = {}
            for width in widths:
                rows = [run_beam_episode(env_name, seed, scorer, width) for seed in seeds]
                succ = sum(int(r["success"]) for r in rows)
                out[env_name][scorer][str(width)] = {
                    "n": len(rows),
                    "successes": succ,
                    "success_rate": succ / len(rows) if rows else None,
                    "mean_steps": mean([r["steps"] for r in rows]) if rows else None,
                }
    return {"seeds": seeds, "beam_widths": widths, "results": out}

def load_episode_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in iter_raw_files(paths):
        try:
            ep = json.load(open(path))
        except Exception:
            continue
        steps = ep.get("steps") or []
        events = [e for st in steps for e in (st.get("events") or [])]
        rows.append({
            "path": str(path),
            "env_name": ep.get("env_name"),
            "model": ep.get("model"),
            "prompt_condition": ep.get("prompt_condition"),
            "seed": ep.get("seed"),
            "success": int(bool(ep.get("success"))),
            "surface_clean": int(not any(e in SURFACE_EVENTS for e in events)),
        })
    return rows


def bootstrap_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    if len(vals) == 0:
        return (math.nan, math.nan, math.nan)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def stratified_bootstrap(rows: List[Dict[str, Any]], n_boot: int = 5000, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {"n_boot": n_boot, "overall": {}, "by_env": {}, "by_env_prompt": {}}
    arr = np.array([r["success"] for r in rows], dtype=float)
    out["overall"]["success_rate"] = dict(zip(["mean", "lo", "hi"], bootstrap_ci(arr, n_boot, rng)))
    surface = np.array([r["success"] for r in rows if r["surface_clean"]], dtype=float)
    out["overall"]["surface_clean_success_rate"] = dict(zip(["mean", "lo", "hi"], bootstrap_ci(surface, n_boot, rng)))
    for env in sorted(set(r["env_name"] for r in rows)):
        env_rows = [r for r in rows if r["env_name"] == env]
        vals = np.array([r["success"] for r in env_rows], dtype=float)
        surf_vals = np.array([r["success"] for r in env_rows if r["surface_clean"]], dtype=float)
        surf_rate_vals = np.array([r["surface_clean"] for r in env_rows], dtype=float)
        out["by_env"][env] = {
            "n": len(env_rows),
            "success_rate": dict(zip(["mean", "lo", "hi"], bootstrap_ci(vals, n_boot, rng))),
            "surface_clean_rate": dict(zip(["mean", "lo", "hi"], bootstrap_ci(surf_rate_vals, n_boot, rng))),
            "surface_clean_success_rate": dict(zip(["mean", "lo", "hi"], bootstrap_ci(surf_vals, n_boot, rng))),
        }
    for env in sorted(set(r["env_name"] for r in rows)):
        for prompt in sorted(set(r["prompt_condition"] for r in rows if r["env_name"] == env)):
            key = f"{env}|{prompt}"
            cell = [r for r in rows if r["env_name"] == env and r["prompt_condition"] == prompt]
            vals = np.array([r["success"] for r in cell], dtype=float)
            out["by_env_prompt"][key] = {
                "n": len(cell),
                "success_rate": dict(zip(["mean", "lo", "hi"], bootstrap_ci(vals, n_boot, rng))),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Experiment dirs/raw dirs/raw JSON files.")
    ap.add_argument("--output", default="results_v4/soundness_audits/soundness_audits.json")
    ap.add_argument("--planner-seeds", default="0-49")
    ap.add_argument("--planner-widths", default="32,128")
    ap.add_argument("--alloy-widths", default="128,512,2048")
    ap.add_argument("--max-alloy-states", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--planner-envs", default="word_ladder,alloy,gb1_sequence")
    args = ap.parse_args()

    paths = args.paths or ["results_v4/core_breadth_20260509"]
    planner_seeds = parse_seed_spec(args.planner_seeds)
    planner_widths = [int(x) for x in args.planner_widths.split(",") if x]
    alloy_widths = [int(x) for x in args.alloy_widths.split(",") if x]
    rows = load_episode_rows(paths)
    result = {
        "inputs": paths,
        "alloy_near_exact_audit": audit_alloy(paths, alloy_widths, max_states=args.max_alloy_states or None),
        "planner_baselines": run_planners([x for x in args.planner_envs.split(",") if x], planner_seeds, planner_widths),
        "bootstrap_ci": stratified_bootstrap(rows, n_boot=args.bootstrap),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({
        "output": str(out),
        "alloy_states": result["alloy_near_exact_audit"]["states_audited"],
        "alloy_episodes": result["alloy_near_exact_audit"]["episodes_audited"],
        "planner_seeds": len(planner_seeds),
    }, indent=2))


def parse_seed_spec(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


if __name__ == "__main__":
    main()
