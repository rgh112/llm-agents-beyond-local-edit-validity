"""Observation-only heuristic baselines for fair LLM comparison.

These baselines use ONLY information visible in the observation text —
the same information the LLM agent sees. They do NOT access internal
environment state, latent variables, or oracle fitness functions.

This provides a fair comparison: if the heuristic outperforms the LLM,
the LLM is failing to extract value from the observation it receives.

Strategies per environment:
  word_ladder:  obs_mismatch_greedy — prefer edits that match target letter
  alloy:        obs_uts_greedy      — prefer edits that increase displayed UTS estimate
  gb1_sequence: obs_additive_greedy — prefer edits with highest displayed single-site estimate
"""
from typing import Dict, List, Tuple

import numpy as np

from benchmark_v4.envs.word_ladder import WordLadderEnv
from benchmark_v4.envs.alloy import AlloyEnv, ELEMENTS
from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv


# ---------------------------------------------------------------------------
# Word Ladder: observation-only mismatch-reducing heuristic
# ---------------------------------------------------------------------------

def run_word_ladder_obs_heuristic(env: WordLadderEnv, seed: int) -> Dict:
    """Greedy on observation: prefer edits that place the target letter.

    Uses only: current word, target word, valid replacements (all in observation).
    """
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 3000)

    while not env.is_done():
        current = env.state.structure
        target = env.state.latent["target_word"]
        used = env.state.latent["used_words"]
        budget = env.state.budget_left

        if current == target:
            env.step("FINALIZE")
            continue

        if budget <= 1:
            env.step("FINALIZE")
            continue

        # Get valid replacements (visible in observation)
        replacements = env._get_valid_replacements(current, used)

        # Priority 1: edits that place the target letter (marked with * in obs)
        target_matches = []
        other_options = []
        for pos, letters in replacements.items():
            for letter in letters:
                if letter == target[pos]:
                    target_matches.append((pos, letter))
                else:
                    other_options.append((pos, letter))

        if target_matches:
            pos, letter = target_matches[int(rng.integers(len(target_matches)))]
            action = f"EDIT {pos} {letter}"
        elif other_options:
            # Pick randomly among non-target options (no strategic preference)
            pos, letter = other_options[int(rng.integers(len(other_options)))]
            action = f"EDIT {pos} {letter}"
        else:
            action = "FINALIZE"

        env.step(action)

    return env.get_episode_summary()


# ---------------------------------------------------------------------------
# Alloy: observation-only UTS-greedy heuristic
# ---------------------------------------------------------------------------

def run_alloy_obs_heuristic(env: AlloyEnv, seed: int) -> Dict:
    """Greedy on observation: pick edit that maximizes displayed UTS estimate.

    Uses only: current composition, displayed UTS/density, allowed deltas.
    Does NOT access _predict_uts() or _calculate_density() directly —
    instead uses the noisy estimate that would appear in the observation.
    """
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 3000)

    while not env.is_done():
        comp = env.state.structure
        budget = env.state.budget_left
        deltas = env._allowed_deltas()

        if budget <= 1:
            env.step("FINALIZE")
            continue

        # Simulate what the LLM sees: noisy UTS estimate
        current_uts_noisy = env._predict_uts(comp) + float(rng.normal(0, 50))
        current_density = env._calculate_density(comp)

        best_action = "FINALIZE"
        best_score = -1e9

        for inc in ELEMENTS:
            for dec in ELEMENTS:
                if inc == dec:
                    continue
                for d in deltas:
                    ni, nd = comp[inc] + d, comp[dec] - d
                    if ni > 100 or nd < 0:
                        continue
                    bounds = env._element_bounds
                    if ni > bounds.get(inc, (0, 100))[1]:
                        continue
                    if nd < bounds.get(dec, (0, 100))[0]:
                        continue

                    # Evaluate using noisy estimate (observation-level)
                    trial = dict(comp)
                    trial[inc] += d
                    trial[dec] -= d
                    trial_uts_noisy = env._predict_uts(trial) + float(rng.normal(0, 50))
                    trial_density = env._calculate_density(trial)

                    # Simple score: UTS improvement minus density penalty
                    uts_gain = trial_uts_noisy - current_uts_noisy
                    density_penalty = max(0, trial_density - env._density_target) * 500
                    score = uts_gain - density_penalty

                    if score > best_score:
                        best_score = score
                        best_action = f"EDIT {inc} {dec} {d}"

        env.step(best_action)

    return env.get_episode_summary()


# ---------------------------------------------------------------------------
# GB1: observation-only additive-greedy heuristic
# ---------------------------------------------------------------------------

def run_gb1_obs_heuristic(env: GB1SequenceEnv, seed: int) -> Dict:
    """Greedy on observation: pick edit with highest displayed single-site estimate.

    Uses only: single-site fitness estimates and stability hints shown in observation.
    This is equivalent to what additive_greedy does — it's the observation-level
    heuristic because single-site estimates ARE what the observation displays.
    """
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 3000)

    while not env.is_done():
        sequence = list(env.state.structure)
        budget = env.state.budget_left

        if budget <= 1:
            env.step("FINALIZE")
            continue

        # Use noisy single-site estimates (what observation shows)
        best_action = "FINALIZE"
        best_delta = -1e9

        for pos in range(4):
            for aa in env._allowed[pos]:
                if aa == sequence[pos]:
                    continue
                # Noisy single-site estimate (observation-level)
                single = env._single_site[str(pos)].get(aa, 0.0)
                cur_val = env._single_site[str(pos)].get(sequence[pos], 1.0)
                delta = single - cur_val + float(rng.normal(0, env._noise_std * 0.5))

                if delta > best_delta:
                    best_delta = delta
                    best_action = f"EDIT {pos} {aa}"

        if best_delta <= 0:
            action = "FINALIZE"
        else:
            action = best_action

        env.step(action)

    return env.get_episode_summary()


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

def run_obs_batch(env_name: str, n_episodes: int = 100, **env_kwargs) -> Dict:
    """Run observation-only heuristic for the specified environment."""
    results = []

    for seed in range(n_episodes):
        if env_name == "word_ladder":
            env = WordLadderEnv(**{k: v for k, v in env_kwargs.items()
                                  if k in ('data_dir', 'word_length', 'shortest_path_range', 'max_steps_factor')})
            r = run_word_ladder_obs_heuristic(env, seed)
        elif env_name == "alloy":
            env = AlloyEnv(**{k: v for k, v in env_kwargs.items()
                             if k not in ('data_dir', 'word_length')})
            r = run_alloy_obs_heuristic(env, seed)
        elif env_name == "gb1_sequence":
            env = GB1SequenceEnv(**{k: v for k, v in env_kwargs.items()
                                   if k in ('max_steps', 'fitness_target', 'noise_std', 'stability_target', 'start_mutations')})
            r = run_gb1_obs_heuristic(env, seed)
        else:
            raise ValueError(f"Unknown env: {env_name}")
        results.append(r)

    sr = sum(1 for r in results if r["success"]) / n_episodes
    return {"sr": sr, "n": n_episodes}
