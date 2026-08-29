#!/usr/bin/env python3
"""Alloy LLM pilot: where does the LLM fall in the baseline hierarchy?"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.alloy import AlloyEnv
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

SEEDS = [0, 1, 2, 3, 4]

CONDITIONS = [
    ("zero_shot", "state_only"),
    ("few_shot", "state_only"),
    ("scaffold", "state_only"),
    ("zero_shot", "window_1"),
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/alloy_pilot_{timestamp}"

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

total = len(CONDITIONS) * len(SEEDS)
count = 0
results_table = []

print(f"=== Alloy LLM Pilot: {total} episodes ===")
print(f"Model: {MODEL_ID}")
print(f"Conditions: {CONDITIONS}")
print(f"Seeds: {SEEDS}")
print(f"Output: {OUTPUT_DIR}")
print()

print("Reference baselines (N=100):")
print("  random:           SR=4%")
print("  greedy_uts:        SR=15%")
print("  greedy_balanced:   SR=86%")
print("  sequence_aware:    SR=93%")
print()

sweep_start = time.time()

for prompt_name, memory_name in CONDITIONS:
    prompt = get_prompt_builder(prompt_name)
    memory = get_memory_module(memory_name)

    successes = 0
    steps_total = 0
    failures = Counter()

    for seed in SEEDS:
        count += 1
        env = AlloyEnv()

        try:
            summary = run_episode(
                env=env, prompt_builder=prompt, memory_module=memory,
                model=model, seed=seed, verbose=False,
            )
            logger.log_episode(summary)

            status = "OK" if summary.success else "FAIL"
            if summary.success:
                successes += 1
            steps_total += len(summary.steps)
            if summary.terminal_failure:
                failures[summary.terminal_failure] += 1

            fm = summary.final_metrics or {}
            uts_str = f"UTS={fm.get('final_uts', fm.get('true_uts', '?')):.0f}" if isinstance(fm.get('final_uts', fm.get('true_uts')), (int, float)) else ""
            dens_str = f"d={fm.get('final_density', fm.get('true_density', '?')):.2f}" if isinstance(fm.get('final_density', fm.get('true_density')), (int, float)) else ""

            print(f"  [{count}/{total}] {prompt_name}|{memory_name}|s{seed} "
                  f"-> {status} ({len(summary.steps)}steps, {summary.wall_time_s:.1f}s) {uts_str} {dens_str}")

        except Exception as e:
            print(f"  [{count}/{total}] {prompt_name}|{memory_name}|s{seed} -> ERROR: {e}")

    sr = successes / len(SEEDS)
    avg_steps = steps_total / len(SEEDS)
    results_table.append({
        "prompt": prompt_name, "memory": memory_name,
        "SR": sr, "avg_steps": avg_steps, "failures": dict(failures),
    })
    print(f"  >> {prompt_name}|{memory_name}: SR={sr:.0%} avg_steps={avg_steps:.1f} failures={dict(failures)}")
    print()

sweep_time = time.time() - sweep_start

print("=" * 70)
print(f"PILOT COMPLETE: {count} episodes in {sweep_time:.0f}s")
print("=" * 70)
print()

print(f"{'Prompt':<12} {'Memory':<12} {'SR':>5} {'Steps':>6}  Failures")
print("-" * 65)
for r in results_table:
    fail_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["failures"].items()))
    print(f"{r['prompt']:<12} {r['memory']:<12} {r['SR']:>4.0%} {r['avg_steps']:>6.1f}  {fail_str}")

print()
print("Comparison with baselines:")
print(f"  {'random':>20}: SR=4%")
print(f"  {'greedy_uts':>20}: SR=15%")
for r in results_table:
    label = f"LLM {r['prompt']}+{r['memory']}"
    print(f"  {label:>20}: SR={r['SR']:.0%}")
print(f"  {'greedy_balanced':>20}: SR=86%")
print(f"  {'sequence_aware':>20}: SR=93%")

summary_path = Path(OUTPUT_DIR) / "pilot_summary.json"
with open(summary_path, "w") as f:
    json.dump({"config": {"model": MODEL_ID, "seeds": SEEDS, "conditions": CONDITIONS},
               "results": results_table, "wall_time_s": sweep_time}, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT_DIR}/")
