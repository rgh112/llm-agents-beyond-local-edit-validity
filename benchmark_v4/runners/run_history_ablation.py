#!/usr/bin/env python3
"""Package B: History ablation experiment.

3 envs × 2 prompts (best + zero_shot) × 3 memory (M0, M1, M3) × 40 seeds = 720 episodes.
Tests when edit history access helps vs hurts.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.word_ladder import WordLadderEnv
from benchmark_v4.envs.alloy import AlloyEnv
from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv
from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.runners.single_episode_runner import run_episode
from benchmark_v4.logging_utils.episode_logger import EpisodeLogger

# ── Config ──
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY before running hosted experiments.")

SEEDS = list(range(40))
MEMORY_CONDITIONS = ["state_only", "window_1", "window_3"]  # M0, M1, M3

GB1_FITNESS_TARGET = float(os.environ.get("GB1_FITNESS_TARGET", "5.0"))

# Per-env best prompt (from Package A results) + zero_shot baseline
# These will be updated after Package A completes
ENV_PROMPTS = {
    "word_ladder": ["zero_shot", "scaffold"],
    "alloy": ["zero_shot", "self_check"],
    "gb1_sequence": ["zero_shot", "self_check"],
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/history_ablation_{timestamp}"

# ── Setup ──
model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

envs = {
    "word_ladder": lambda: WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6)),
    "alloy": lambda: AlloyEnv(),
    "gb1_sequence": lambda: GB1SequenceEnv(fitness_target=GB1_FITNESS_TARGET),
}

total = sum(len(prompts) for prompts in ENV_PROMPTS.values()) * len(MEMORY_CONDITIONS) * len(SEEDS)
combo_count = 0
results_table = []

print(f"=== Package B: History Ablation ===")
print(f"  Memory conditions: {MEMORY_CONDITIONS}")
print(f"  Seeds: {len(SEEDS)}")
print(f"  Total: {total} episodes")
print(f"  Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for env_name, env_factory in envs.items():
    prompts_for_env = ENV_PROMPTS[env_name]

    for prompt_name in prompts_for_env:
        for mem_name in MEMORY_CONDITIONS:
            prompt = get_prompt_builder(prompt_name)
            memory = get_memory_module(mem_name)

            successes = 0
            steps_total = 0
            failures = Counter()
            events = Counter()

            for seed in SEEDS:
                combo_count += 1
                env = env_factory()

                try:
                    summary = run_episode(
                        env=env, prompt_builder=prompt, memory_module=memory,
                        model=model, seed=seed, verbose=False,
                    )
                    logger.log_episode(summary)

                    if summary.success:
                        successes += 1
                    steps_total += len(summary.steps)
                    if summary.terminal_failure:
                        failures[summary.terminal_failure] += 1
                    for ename, cnt in summary.event_counts.items():
                        events[ename] += cnt

                    status = "OK" if summary.success else "FAIL"
                    print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|{mem_name}|s{seed} "
                          f"-> {status} ({len(summary.steps)}steps)")

                except Exception as e:
                    print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|{mem_name}|s{seed} -> ERROR: {e}")

            sr = successes / len(SEEDS)
            avg_steps = steps_total / len(SEEDS)

            results_table.append({
                "env": env_name,
                "prompt": prompt_name,
                "memory": mem_name,
                "SR": sr,
                "n_success": successes,
                "n_total": len(SEEDS),
                "avg_steps": round(avg_steps, 1),
                "terminal_failures": dict(failures),
                "all_events": dict(events),
            })

            print(f"  >> {env_name}|{prompt_name}|{mem_name}: SR={sr:.0%} ({successes}/{len(SEEDS)})")
            print()

sweep_time = time.time() - sweep_start

# ── Results ──
print(f"\n{'='*90}")
print(f"HISTORY ABLATION COMPLETE: {combo_count} episodes in {sweep_time:.0f}s")
print(f"{'='*90}")

for env_name in envs:
    print(f"\n{'='*25} {env_name.upper()} {'='*25}")
    env_results = [r for r in results_table if r["env"] == env_name]
    print(f"{'Prompt':<22} {'Memory':<12} {'SR':>8} {'Steps':>6}")
    print("-" * 55)
    for r in env_results:
        print(f"{r['prompt']:<22} {r['memory']:<12} "
              f"{r['n_success']:>2}/{r['n_total']:<3} ({r['SR']:>4.0%}) "
              f"{r['avg_steps']:>5.1f}")

# History sensitivity analysis
print(f"\n{'='*25} HISTORY SENSITIVITY {'='*25}")
for env_name in envs:
    env_results = [r for r in results_table if r["env"] == env_name]
    for prompt_name in ENV_PROMPTS[env_name]:
        prompt_results = [r for r in env_results if r["prompt"] == prompt_name]
        m0 = next((r for r in prompt_results if r["memory"] == "state_only"), None)
        m1 = next((r for r in prompt_results if r["memory"] == "window_1"), None)
        m3 = next((r for r in prompt_results if r["memory"] == "window_3"), None)
        if m0 and m1 and m3:
            print(f"  {env_name}|{prompt_name}:")
            print(f"    M0={m0['SR']:.0%}  M1={m1['SR']:.0%}  M3={m3['SR']:.0%}")
            delta_m1 = m1['SR'] - m0['SR']
            delta_m3 = m3['SR'] - m0['SR']
            print(f"    M1 effect: {delta_m1:+.0%}  M3 effect: {delta_m3:+.0%}")

# Save
summary_path = Path(OUTPUT_DIR) / "history_ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "config": {
            "model": MODEL_ID,
            "seeds": SEEDS,
            "memory_conditions": MEMORY_CONDITIONS,
            "env_prompts": ENV_PROMPTS,
            "gb1_fitness_target": GB1_FITNESS_TARGET,
        },
        "results": results_table,
        "total_episodes": combo_count,
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
