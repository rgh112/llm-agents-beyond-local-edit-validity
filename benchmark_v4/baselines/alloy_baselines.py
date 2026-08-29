"""Alloy environment baseline strategies for calibration and comparison.

All non-random baselines are privileged: they access internal environment
functions (_predict_uts, _calculate_density) not available to the LLM agent.

Strategies:
  random                      — uniformly random valid edits
  privileged_greedy_uts       — maximize UTS only, ignores density (oracle access)
  privileged_greedy_balanced  — maximize UTS with density penalty (oracle access)
  privileged_sequence_aware   — balanced + trajectory trend + RC awareness (oracle access)
"""
from collections import Counter
from typing import Dict, List

import numpy as np

from benchmark_v4.envs.alloy import AlloyEnv, ELEMENTS

STRATEGIES = ["random", "privileged_greedy_uts", "privileged_greedy_balanced", "privileged_sequence_aware"]


def _is_valid(env: AlloyEnv, comp: dict, inc: str, dec: str, d: int) -> bool:
    ni, nd = comp[inc] + d, comp[dec] - d
    if ni > 100 or nd < 0:
        return False
    if ni > env._element_bounds.get(inc, (0, 100))[1]:
        return False
    if nd < env._element_bounds.get(dec, (0, 100))[0]:
        return False
    return True


def run_baseline(env: AlloyEnv, strategy: str, seed: int) -> Dict:
    """Run one episode with a baseline strategy. Returns episode summary dict."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000)

    while not env.is_done():
        comp = env.state.structure
        deltas = env._allowed_deltas()

        if strategy == "random":
            action = None
            for _ in range(100):
                inc = rng.choice(ELEMENTS)
                dec = rng.choice([e for e in ELEMENTS if e != inc])
                d = int(rng.choice(deltas))
                if _is_valid(env, comp, inc, dec, d):
                    action = f"EDIT {inc} {dec} {d}"
                    break
            if action is None:
                action = "FINALIZE"

        elif strategy == "privileged_greedy_uts":
            best_action, best_uts = "FINALIZE", -1e9
            for inc in ELEMENTS:
                for dec in ELEMENTS:
                    if inc == dec:
                        continue
                    for d in deltas:
                        if not _is_valid(env, comp, inc, dec, d):
                            continue
                        nc = dict(comp); nc[inc] += d; nc[dec] -= d
                        u = env._predict_uts(nc)
                        if u > best_uts:
                            best_uts = u
                            best_action = f"EDIT {inc} {dec} {d}"
            action = best_action

        elif strategy == "privileged_greedy_balanced":
            best_action, best_score = "FINALIZE", -1e9
            cu = env._predict_uts(comp)
            cd = env._calculate_density(comp)
            for inc in ELEMENTS:
                for dec in ELEMENTS:
                    if inc == dec:
                        continue
                    for d in deltas:
                        if not _is_valid(env, comp, inc, dec, d):
                            continue
                        nc = dict(comp); nc[inc] += d; nc[dec] -= d
                        u = env._predict_uts(nc)
                        dn = env._calculate_density(nc)
                        s = ((u - cu) / env._uts_target
                             - max(0, dn - env._density_target) * 10
                             - max(0, dn - cd) * 2)
                        if s > best_score:
                            best_score = s
                            best_action = f"EDIT {inc} {dec} {d}"
            action = best_action

        elif strategy == "privileged_sequence_aware":
            best_action, best_score = "FINALIZE", -1e9
            cu = env._predict_uts(comp)
            cd = env._calculate_density(comp)
            dt = env.state.latent.get("density_trajectory", [])
            trend = (dt[-1] - dt[-2]) if len(dt) >= 2 else 0
            rc = env.state.latent.get("recovery_cost", 0)
            for inc in ELEMENTS:
                for dec in ELEMENTS:
                    if inc == dec:
                        continue
                    for d in deltas:
                        if not _is_valid(env, comp, inc, dec, d):
                            continue
                        nc = dict(comp); nc[inc] += d; nc[dec] -= d
                        u = env._predict_uts(nc)
                        dn = env._calculate_density(nc)
                        ug = (u - cu) / env._uts_target
                        dp = max(0, dn - env._density_target) * 15
                        dc = dn - cd
                        if dc > 0 and trend > 0:
                            dp += dc * 5
                        rr = 0.4 if inc == "Mo" else (0.15 if inc in ("Co", "Ni") else 0)
                        s = ug - dp - rr - rc * 0.15
                        if s > best_score:
                            best_score = s
                            best_action = f"EDIT {inc} {dec} {d}"
            action = best_action

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        env.step(action)
        if not env.is_done() and env.state.budget_left <= 1:
            env.step("FINALIZE")

    return env.get_episode_summary()


def run_batch(strategy: str, n_episodes: int, **env_kwargs) -> Dict:
    """Run N episodes and return aggregate stats."""
    results = [run_baseline(AlloyEnv(**env_kwargs), strategy, seed)
               for seed in range(n_episodes)]
    sr = sum(1 for r in results if r["success"]) / n_episodes
    evts = Counter()
    for r in results:
        evts.update(r.get("events", []))

    fail_reasons = Counter()
    for r in results:
        if not r["success"]:
            if not r.get("uts_met", True):
                fail_reasons["uts_unmet"] += 1
            if not r.get("density_met", True):
                fail_reasons["density_over"] += 1
            if not r.get("recovery_cost_ok", True):
                fail_reasons["rc_explosion"] += 1

    return {
        "sr": sr,
        "avg_uts": float(np.mean([r["final_uts"] for r in results])),
        "avg_density": float(np.mean([r["final_density"] for r in results])),
        "avg_rc": float(np.mean([r["recovery_cost"] for r in results])),
        "events": dict(evts),
        "fail_reasons": dict(fail_reasons),
    }
