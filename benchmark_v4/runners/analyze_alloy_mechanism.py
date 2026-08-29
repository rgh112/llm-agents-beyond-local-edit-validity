#!/usr/bin/env python3
"""Deterministic diagnostics for the alloy-like composition environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.baselines.alloy_baselines import run_baseline
from benchmark_v4.envs.alloy import AlloyEnv, ELEMENTS


SETTINGS = {
    "default": {},
    "heavy_scaling_off": {"heavy_floor": 1.0},
    "heavy_scaling_strong": {"heavy_floor": 0.15, "heavy_threshold": 24.0},
    "recovery_off": {"rc_rate": 0.0, "rc_threshold": 1e9},
    "recovery_strict": {"rc_threshold": 2.5},
    "easy_combined": {"heavy_floor": 1.0, "rc_rate": 0.0, "rc_threshold": 1e9},
    "hard_combined": {"heavy_floor": 0.15, "heavy_threshold": 24.0, "rc_threshold": 2.5},
}


def _heavy_pct(env: AlloyEnv, comp: dict[str, float]) -> float:
    return float(sum(comp.get(el, 0.0) * w for el, w in env._heavy_weights.items()))


def _summarize(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def one_step_tradeoff(env: AlloyEnv, seeds: list[int]) -> dict:
    rows = []
    start_rows = []
    for seed in seeds:
        env.reset(seed=seed)
        comp = dict(env.state.structure)
        start_rows.append({
            "seed": seed,
            "uts": env._predict_uts(comp),
            "density": env._calculate_density(comp),
            "heavy_pct": _heavy_pct(env, comp),
            "success": env._state_success(
                comp,
                float(env.state.latent["recovery_cost"]),
                list(env.state.latent["density_trajectory"]),
            ),
        })
        prev_uts = env._predict_uts(comp)
        prev_density = env._calculate_density(comp)
        prev_rc = float(env.state.latent["recovery_cost"])
        prev_traj = list(env.state.latent["density_trajectory"])
        for edit in env._simulate_valid_edits(comp, env.state.t):
            nxt_comp, nxt_rc, nxt_traj = env._simulate_edit_state(comp, prev_rc, prev_traj, edit)
            rows.append({
                "seed": seed,
                "edit": edit,
                "delta_uts": env._predict_uts(nxt_comp) - prev_uts,
                "delta_density": env._calculate_density(nxt_comp) - prev_density,
                "delta_heavy_pct": _heavy_pct(env, nxt_comp) - _heavy_pct(env, comp),
                "delta_recovery_cost": nxt_rc - prev_rc,
            })

    uts_up = [r for r in rows if r["delta_uts"] > 0]
    density_down = [r for r in rows if r["delta_density"] < 0]
    tradeoff = [
        r for r in rows
        if (r["delta_uts"] > 0 and r["delta_density"] > 0)
        or (r["delta_uts"] < 0 and r["delta_density"] < 0)
    ]
    deltas_uts = np.array([r["delta_uts"] for r in rows], dtype=float)
    deltas_density = np.array([r["delta_density"] for r in rows], dtype=float)
    corr = float(np.corrcoef(deltas_uts, deltas_density)[0, 1]) if len(rows) > 1 else None

    return {
        "n_start_states": len(start_rows),
        "n_valid_one_step_edits": len(rows),
        "start_success_fraction": sum(r["success"] for r in start_rows) / len(start_rows),
        "start_uts": _summarize([r["uts"] for r in start_rows]),
        "start_density": _summarize([r["density"] for r in start_rows]),
        "start_heavy_pct": _summarize([r["heavy_pct"] for r in start_rows]),
        "delta_uts_density_correlation": corr,
        "tradeoff_edit_fraction": len(tradeoff) / len(rows) if rows else None,
        "uts_improving_edit_fraction": len(uts_up) / len(rows) if rows else None,
        "uts_improving_edits_density_worse_fraction": (
            sum(1 for r in uts_up if r["delta_density"] > 0) / len(uts_up)
            if uts_up else None
        ),
        "density_improving_edit_fraction": len(density_down) / len(rows) if rows else None,
        "density_improving_edits_uts_worse_fraction": (
            sum(1 for r in density_down if r["delta_uts"] < 0) / len(density_down)
            if density_down else None
        ),
        "recovery_cost_increasing_edit_fraction": (
            sum(1 for r in rows if r["delta_recovery_cost"] > 0) / len(rows) if rows else None
        ),
    }


def baseline_sensitivity(seeds: list[int], settings: dict[str, dict]) -> dict:
    out = {}
    strategies = ["random", "privileged_greedy_uts", "privileged_greedy_balanced", "privileged_sequence_aware"]
    for setting, kwargs in settings.items():
        env = AlloyEnv(**kwargs)
        # These diagnostics do not affect transitions or final success, but
        # they are expensive inside step(); disable them for baseline sweeps.
        env._recoverability_probe_depth = 0
        env._recoverability_beam_width = 1
        out[setting] = {}
        for strategy in strategies:
            rows = [run_baseline(env, strategy, seed) for seed in seeds]
            out[setting][strategy] = {
                "n": len(rows),
                "SR": sum(1 for r in rows if r["success"]) / len(rows),
                "avg_uts": float(np.mean([r["final_uts"] for r in rows])),
                "avg_density": float(np.mean([r["final_density"] for r in rows])),
                "avg_recovery_cost": float(np.mean([r["recovery_cost"] for r in rows])),
                "uts_met_rate": sum(1 for r in rows if r.get("uts_met")) / len(rows),
                "density_met_rate": sum(1 for r in rows if r.get("density_met")) / len(rows),
                "recovery_cost_ok_rate": sum(1 for r in rows if r.get("recovery_cost_ok")) / len(rows),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--baseline-seeds", default="0-19")
    ap.add_argument("--output", default="results_v4/alloy_mechanism_stats.json")
    args = ap.parse_args()

    def parse_seed_spec(spec: str) -> list[int]:
        if "-" in spec:
            start, end = [int(x) for x in spec.split("-", 1)]
            return list(range(start, end + 1))
        return [int(x) for x in spec.split(",") if x.strip()]

    seeds = parse_seed_spec(args.seeds)
    baseline_seeds = parse_seed_spec(args.baseline_seeds)

    default_env = AlloyEnv()
    out = {
        "seeds": seeds,
        "baseline_seeds": baseline_seeds,
        "settings": SETTINGS,
        "default_one_step_tradeoff": one_step_tradeoff(default_env, seeds),
        "baseline_sensitivity": baseline_sensitivity(baseline_seeds, SETTINGS),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
