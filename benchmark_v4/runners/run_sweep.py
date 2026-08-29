#!/usr/bin/env python3
"""Lightweight v4 sweep: 3 envs × 4 prompts × 2 memory × 5 seeds = 120 episodes."""
import sys, os, json, time
from datetime import datetime
from pathlib import Path

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

SEEDS = list(range(10))
PROMPT_CONDITIONS = ["zero_shot", "few_shot", "scaffold", "self_check"]
MEMORY_CONDITIONS = ["state_only", "window_1"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/sweep_{timestamp}"

# ── Setup ──
model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

envs = {
    "word_ladder": lambda: WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6)),
    "alloy": lambda: AlloyEnv(),
    "gb1_sequence": lambda: GB1SequenceEnv(),
}

total = len(envs) * len(PROMPT_CONDITIONS) * len(MEMORY_CONDITIONS) * len(SEEDS)
combo_count = 0
all_summaries = []
results_table = []

print(f"=== v4 Sweep: {total} episodes ===")
print(f"Model: {MODEL_ID}")
print(f"Envs: {list(envs.keys())}")
print(f"Prompts: {PROMPT_CONDITIONS}")
print(f"Memory: {MEMORY_CONDITIONS}")
print(f"Seeds: {SEEDS}")
print(f"Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for env_name, env_factory in envs.items():
    for prompt_name in PROMPT_CONDITIONS:
        for memory_name in MEMORY_CONDITIONS:
            prompt = get_prompt_builder(prompt_name)
            memory = get_memory_module(memory_name)

            condition_successes = 0
            condition_steps = 0
            condition_failures = {}

            for seed in SEEDS:
                combo_count += 1
                env = env_factory()

                try:
                    summary = run_episode(
                        env=env,
                        prompt_builder=prompt,
                        memory_module=memory,
                        model=model,
                        seed=seed,
                        verbose=False,
                    )
                    logger.log_episode(summary)
                    all_summaries.append(summary)

                    status = "OK" if summary.success else "FAIL"
                    if summary.success:
                        condition_successes += 1
                    condition_steps += len(summary.steps)

                    if summary.terminal_failure:
                        tf = summary.terminal_failure
                        condition_failures[tf] = condition_failures.get(tf, 0) + 1

                    print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|{memory_name}|s{seed} "
                          f"-> {status} ({len(summary.steps)} steps, {summary.wall_time_s:.1f}s)")

                except Exception as e:
                    print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|{memory_name}|s{seed} "
                          f"-> ERROR: {e}")

            sr = condition_successes / len(SEEDS)
            avg_steps = condition_steps / len(SEEDS)
            results_table.append({
                "env": env_name,
                "prompt": prompt_name,
                "memory": memory_name,
                "SR": f"{sr:.0%}",
                "avg_steps": f"{avg_steps:.1f}",
                "failures": condition_failures,
            })
            print(f"  >> {env_name}|{prompt_name}|{memory_name}: SR={sr:.0%} "
                  f"avg_steps={avg_steps:.1f} failures={condition_failures}")
            print()

sweep_time = time.time() - sweep_start

# ── Summary ──
print()
print("=" * 70)
print(f"SWEEP COMPLETE: {len(all_summaries)} episodes in {sweep_time:.0f}s")
print("=" * 70)
print()

header = f"{'Env':<14} {'Prompt':<12} {'Memory':<12} {'SR':>5} {'Steps':>6}  Failures"
print(header)
print("-" * len(header) + "-" * 30)
for r in results_table:
    fail_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["failures"].items()))
    print(f"{r['env']:<14} {r['prompt']:<12} {r['memory']:<12} {r['SR']:>5} {r['avg_steps']:>6}  {fail_str}")

summary_path = Path(OUTPUT_DIR) / "sweep_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "config": {
            "model": MODEL_ID,
            "seeds": SEEDS,
            "prompt_conditions": PROMPT_CONDITIONS,
            "memory_conditions": MEMORY_CONDITIONS,
            "environments": list(envs.keys()),
        },
        "results": results_table,
        "total_episodes": len(all_summaries),
        "total_successes": sum(1 for s in all_summaries if s.success),
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
