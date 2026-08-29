"""GB1 sequence environment baseline strategies for calibration.

Non-random baselines access internal fitness functions not available to the LLM.

Strategies:
  random                    — uniformly random valid edits
  additive_greedy           — greedy on single-site estimates (observation-level, falls for epistasis)
  privileged_noisy_oracle   — greedy on true fitness + noise (oracle with imperfect knowledge)
  privileged_epistasis      — greedy on true combinatorial fitness (full oracle, one-step lookahead)
"""
from collections import Counter
from typing import Dict

import numpy as np

from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv


STRATEGIES = ["random", "additive_greedy", "privileged_noisy_oracle", "privileged_epistasis"]


def run_baseline(env: GB1SequenceEnv, strategy: str, seed: int,
                 oracle_noise: float = 1.5) -> Dict:
    """Run one episode with a baseline strategy."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 2000)

    while not env.is_done():
        sequence = list(env.state.structure)
        budget = env.state.budget_left

        if budget <= 1:
            env.step("FINALIZE")
            continue

        if strategy == "random":
            options = []
            for pos in range(4):
                for aa in env._allowed[pos]:
                    if aa != sequence[pos]:
                        options.append((pos, aa))
            if options:
                pos, aa = options[int(rng.integers(len(options)))]
                action = f"EDIT {pos} {aa}"
            else:
                action = "FINALIZE"

        elif strategy == "additive_greedy":
            best_action, best_delta = "FINALIZE", -1e9
            for pos in range(4):
                for aa in env._allowed[pos]:
                    if aa == sequence[pos]:
                        continue
                    single = env._single_site[str(pos)].get(aa, 0.0)
                    cur_val = env._single_site[str(pos)].get(sequence[pos], 1.0)
                    delta = single - cur_val
                    if delta > best_delta:
                        best_delta = delta
                        best_action = f"EDIT {pos} {aa}"
            if best_delta <= 0:
                action = "FINALIZE"
            else:
                action = best_action

        elif strategy == "privileged_noisy_oracle":
            # Greedy on true fitness + Gaussian noise
            # Simulates imperfect knowledge of combinatorial landscape
            best_action, best_score = "FINALIZE", -1e9
            cur_fit = env._true_fitness(sequence)
            for pos in range(4):
                for aa in env._allowed[pos]:
                    if aa == sequence[pos]:
                        continue
                    trial = list(sequence)
                    trial[pos] = aa
                    fit = env._true_fitness(trial)
                    noisy_fit = fit + float(rng.normal(0, oracle_noise))
                    delta = noisy_fit - cur_fit
                    if delta > best_score:
                        best_score = delta
                        best_action = f"EDIT {pos} {aa}"
            if best_score <= 0:
                action = "FINALIZE"
            else:
                action = best_action

        elif strategy == "privileged_epistasis":
            best_action, best_fitness = "FINALIZE", env._true_fitness(sequence)
            for pos in range(4):
                for aa in env._allowed[pos]:
                    if aa == sequence[pos]:
                        continue
                    trial = list(sequence)
                    trial[pos] = aa
                    fit = env._true_fitness(trial)
                    if fit > best_fitness:
                        best_fitness = fit
                        best_action = f"EDIT {pos} {aa}"
            action = best_action

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        env.step(action)

    return env.get_episode_summary()


def run_batch(strategy: str, n_episodes: int, oracle_noise: float = 1.5,
              **env_kwargs) -> Dict:
    """Run N episodes and return aggregate stats."""
    results = [run_baseline(GB1SequenceEnv(**env_kwargs), strategy, seed,
                            oracle_noise=oracle_noise)
               for seed in range(n_episodes)]
    sr = sum(1 for r in results if r["success"]) / n_episodes
    avg_fitness = float(np.mean([r["true_fitness"] for r in results]))

    evts = Counter()
    for r in results:
        for e in r.get("events", []):
            evts[e] += 1

    return {
        "sr": sr,
        "avg_true_fitness": avg_fitness,
        "events": dict(evts),
    }
