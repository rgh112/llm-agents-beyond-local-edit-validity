#!/usr/bin/env python3
"""Package C: Comprehensive baseline experiment.

Runs all baselines for all 3 environments with enough seeds for stable estimates.
Produces the baseline hierarchy table for the paper.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.word_ladder import WordLadderEnv
from benchmark_v4.envs.alloy import AlloyEnv
from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv

from benchmark_v4.baselines.alloy_baselines import run_baseline as run_alloy_baseline
from benchmark_v4.baselines.gb1_baselines import run_baseline as run_gb1_baseline
from benchmark_v4.baselines.observation_heuristics import (
    run_word_ladder_obs_heuristic,
    run_alloy_obs_heuristic,
    run_gb1_obs_heuristic,
)

N_EPISODES = 100  # Per baseline strategy
GB1_FITNESS_TARGET = float(os.environ.get("GB1_FITNESS_TARGET", "5.0"))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/baselines_{timestamp}"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

results = []
sweep_start = time.time()

# ═══════════════════════════════════════════════════
# Word Ladder baselines
# ═══════════════════════════════════════════════════
print("=" * 60)
print("WORD LADDER BASELINES")
print("=" * 60)

# Random baseline
print("\n  random...")
wl_random_ok = 0
for seed in range(N_EPISODES):
    env = WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 5000)
    while not env.is_done():
        current = env.state.structure
        used = env.state.latent["used_words"]
        neighbors = env._compute_neighbors(current, used)
        if neighbors:
            chosen = neighbors[int(rng.integers(len(neighbors)))]
            # Find the edit
            for pos in range(len(current)):
                if current[pos] != chosen[pos]:
                    env.step(f"EDIT {pos} {chosen[pos]}")
                    break
        else:
            env.step("FINALIZE")
    if env._success:
        wl_random_ok += 1
sr = wl_random_ok / N_EPISODES
print(f"    random: SR={sr:.0%} ({wl_random_ok}/{N_EPISODES})")
results.append({"env": "word_ladder", "strategy": "random", "SR": sr, "n": N_EPISODES})

# Observation heuristic
print("  obs_mismatch_greedy...")
wl_obs_ok = 0
for seed in range(N_EPISODES):
    env = WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6))
    s = run_word_ladder_obs_heuristic(env, seed)
    if s["success"]:
        wl_obs_ok += 1
sr = wl_obs_ok / N_EPISODES
print(f"    obs_mismatch_greedy: SR={sr:.0%} ({wl_obs_ok}/{N_EPISODES})")
results.append({"env": "word_ladder", "strategy": "obs_mismatch_greedy", "SR": sr, "n": N_EPISODES})

# BFS oracle (upper bound)
print("  bfs_oracle...")
import networkx as nx
wl_bfs_ok = 0
wl_bfs_lengths = []
for seed in range(N_EPISODES):
    env = WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6))
    env.reset(seed=seed)
    start = env.state.structure
    target = env.state.latent["target_word"]
    try:
        path = nx.shortest_path(env.graph, start, target)
        # Execute path
        for i in range(1, len(path)):
            prev, curr = path[i-1], path[i]
            for pos in range(len(prev)):
                if prev[pos] != curr[pos]:
                    env.step(f"EDIT {pos} {curr[pos]}")
                    break
        wl_bfs_ok += 1
        wl_bfs_lengths.append(len(path) - 1)
    except nx.NetworkXNoPath:
        pass
sr = wl_bfs_ok / N_EPISODES
avg_len = np.mean(wl_bfs_lengths) if wl_bfs_lengths else 0
print(f"    bfs_oracle: SR={sr:.0%} ({wl_bfs_ok}/{N_EPISODES}), avg_path={avg_len:.1f}")
results.append({"env": "word_ladder", "strategy": "bfs_oracle", "SR": sr, "n": N_EPISODES,
                "avg_path_length": round(float(avg_len), 1)})

# ═══════════════════════════════════════════════════
# Alloy baselines
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALLOY BASELINES")
print("=" * 60)

ALLOY_STRATEGIES = ["random", "privileged_greedy_uts", "privileged_greedy_balanced", "privileged_sequence_aware"]

for strat in ALLOY_STRATEGIES:
    print(f"\n  {strat}...")
    ok = 0
    uts_vals = []
    density_vals = []
    rc_vals = []
    for seed in range(N_EPISODES):
        env = AlloyEnv()
        s = run_alloy_baseline(env, strat, seed)
        if s["success"]:
            ok += 1
        uts_vals.append(s["final_uts"])
        density_vals.append(s["final_density"])
        rc_vals.append(s["recovery_cost"])
    sr = ok / N_EPISODES
    print(f"    {strat}: SR={sr:.0%} ({ok}/{N_EPISODES}) "
          f"avg_uts={np.mean(uts_vals):.0f} avg_density={np.mean(density_vals):.2f} avg_rc={np.mean(rc_vals):.2f}")
    results.append({
        "env": "alloy", "strategy": strat, "SR": sr, "n": N_EPISODES,
        "avg_uts": round(float(np.mean(uts_vals)), 1),
        "avg_density": round(float(np.mean(density_vals)), 3),
        "avg_rc": round(float(np.mean(rc_vals)), 3),
    })

# Alloy observation heuristic
print(f"\n  obs_uts_greedy...")
ok = 0
for seed in range(N_EPISODES):
    env = AlloyEnv()
    s = run_alloy_obs_heuristic(env, seed)
    if s["success"]:
        ok += 1
sr = ok / N_EPISODES
print(f"    obs_uts_greedy: SR={sr:.0%} ({ok}/{N_EPISODES})")
results.append({"env": "alloy", "strategy": "obs_uts_greedy", "SR": sr, "n": N_EPISODES})

# ═══════════════════════════════════════════════════
# GB1 baselines
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GB1 BASELINES")
print("=" * 60)

GB1_STRATEGIES = ["random", "additive_greedy", "privileged_noisy_oracle", "privileged_epistasis"]

for strat in GB1_STRATEGIES:
    print(f"\n  {strat}...")
    ok = 0
    fitness_vals = []
    for seed in range(N_EPISODES):
        env = GB1SequenceEnv(fitness_target=GB1_FITNESS_TARGET)
        s = run_gb1_baseline(env, strat, seed)
        if s["success"]:
            ok += 1
        fitness_vals.append(s["true_fitness"])
    sr = ok / N_EPISODES
    print(f"    {strat}: SR={sr:.0%} ({ok}/{N_EPISODES}) avg_fitness={np.mean(fitness_vals):.2f}")
    results.append({
        "env": "gb1_sequence", "strategy": strat, "SR": sr, "n": N_EPISODES,
        "avg_fitness": round(float(np.mean(fitness_vals)), 3),
    })

# GB1 observation heuristic
print(f"\n  obs_additive_greedy...")
ok = 0
for seed in range(N_EPISODES):
    env = GB1SequenceEnv(fitness_target=GB1_FITNESS_TARGET)
    s = run_gb1_obs_heuristic(env, seed)
    if s["success"]:
        ok += 1
sr = ok / N_EPISODES
print(f"    obs_additive_greedy: SR={sr:.0%} ({ok}/{N_EPISODES})")
results.append({"env": "gb1_sequence", "strategy": "obs_additive_greedy", "SR": sr, "n": N_EPISODES})

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
sweep_time = time.time() - sweep_start

print(f"\n{'='*70}")
print(f"BASELINES COMPLETE in {sweep_time:.0f}s")
print(f"{'='*70}")

for env_name in ["word_ladder", "alloy", "gb1_sequence"]:
    print(f"\n  {env_name.upper()}:")
    env_results = [r for r in results if r["env"] == env_name]
    for r in sorted(env_results, key=lambda x: x["SR"]):
        extras = ""
        if "avg_fitness" in r:
            extras = f" avg_fit={r['avg_fitness']:.2f}"
        elif "avg_uts" in r:
            extras = f" avg_uts={r['avg_uts']:.0f} density={r['avg_density']:.2f}"
        elif "avg_path_length" in r:
            extras = f" avg_path={r['avg_path_length']:.1f}"
        print(f"    {r['strategy']:<30} SR={r['SR']:.0%}{extras}")

# Save
with open(Path(OUTPUT_DIR) / "baselines_summary.json", "w") as f:
    json.dump({
        "config": {"n_episodes": N_EPISODES, "gb1_fitness_target": GB1_FITNESS_TARGET},
        "results": results,
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
