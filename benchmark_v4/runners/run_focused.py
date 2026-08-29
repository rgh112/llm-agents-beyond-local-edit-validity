#!/usr/bin/env python3
"""Focused experiment: key conditions × 30 seeds for robust results."""
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

SEEDS = list(range(30))

# Key conditions per environment (from pilot analysis)
CONDITIONS = {
    "word_ladder": [
        ("zero_shot", "state_only"),
        ("few_shot", "state_only"),
        ("few_shot", "window_1"),
        ("self_check", "state_only"),
    ],
    "alloy": [
        ("zero_shot", "state_only"),
        ("few_shot", "state_only"),
        ("self_check", "state_only"),
        ("zero_shot", "window_1"),
    ],
    "gb1_sequence": [
        ("zero_shot", "state_only"),
        ("few_shot", "state_only"),
        ("self_check", "state_only"),
        ("zero_shot", "window_1"),
    ],
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/focused_{timestamp}"

# ── Setup ──
model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

envs = {
    "word_ladder": lambda: WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6)),
    "alloy": lambda: AlloyEnv(),
    "gb1_sequence": lambda: GB1SequenceEnv(),
}

total = sum(len(conds) for conds in CONDITIONS.values()) * len(SEEDS)
combo_count = 0
all_summaries = []
results_table = []

print(f"=== Focused Experiment: {total} episodes ===")
print(f"Model: {MODEL_ID}")
print(f"Seeds: {len(SEEDS)}")
print(f"Output: {OUTPUT_DIR}")
for env_name, conds in CONDITIONS.items():
    print(f"  {env_name}: {len(conds)} conditions × {len(SEEDS)} seeds = {len(conds)*len(SEEDS)}")
print()

sweep_start = time.time()

for env_name, conds in CONDITIONS.items():
    for prompt_name, memory_name in conds:
        prompt = get_prompt_builder(prompt_name)
        memory = get_memory_module(memory_name)

        condition_successes = 0
        condition_steps = 0
        condition_failures = Counter()
        condition_events = Counter()

        for seed in SEEDS:
            combo_count += 1
            env = envs[env_name]()

            try:
                summary = run_episode(
                    env=env, prompt_builder=prompt, memory_module=memory,
                    model=model, seed=seed, verbose=False,
                )
                logger.log_episode(summary)
                all_summaries.append(summary)

                status = "OK" if summary.success else "FAIL"
                if summary.success:
                    condition_successes += 1
                condition_steps += len(summary.steps)

                if summary.terminal_failure:
                    condition_failures[summary.terminal_failure] += 1

                # Collect all events
                for step in summary.steps:
                    if hasattr(step, 'events'):
                        for e in step.events:
                            ename = e.name if hasattr(e, 'name') else str(e)
                            condition_events[ename] += 1

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
            "SR": sr,
            "SR_pct": f"{sr:.0%}",
            "avg_steps": round(avg_steps, 1),
            "n_success": condition_successes,
            "n_total": len(SEEDS),
            "terminal_failures": dict(condition_failures),
            "all_events": dict(condition_events),
        })
        print(f"  >> {env_name}|{prompt_name}|{memory_name}: SR={sr:.0%} "
              f"({condition_successes}/{len(SEEDS)}) avg_steps={avg_steps:.1f}")
        print(f"     failures={dict(condition_failures)}")
        print(f"     events={dict(condition_events)}")
        print()

sweep_time = time.time() - sweep_start

# ── Summary ──
print()
print("=" * 80)
print(f"FOCUSED EXPERIMENT COMPLETE: {len(all_summaries)} episodes in {sweep_time:.0f}s")
print("=" * 80)
print()

# Results by environment
for env_name in CONDITIONS:
    print(f"\n{'='*20} {env_name.upper()} {'='*20}")
    env_results = [r for r in results_table if r["env"] == env_name]
    print(f"{'Prompt':<12} {'Memory':<12} {'SR':>8} {'Steps':>6}  Top Failures")
    print("-" * 75)
    for r in env_results:
        top_fail = ", ".join(f"{k}:{v}" for k, v in
                            sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3])
        print(f"{r['prompt']:<12} {r['memory']:<12} "
              f"{r['n_success']:>2}/{r['n_total']:<3} ({r['SR_pct']:>4}) "
              f"{r['avg_steps']:>5.1f}  {top_fail}")

# Failure composition table
print(f"\n{'='*20} FAILURE COMPOSITION {'='*20}")
for env_name in CONDITIONS:
    print(f"\n{env_name}:")
    env_results = [r for r in results_table if r["env"] == env_name]
    # Collect all event types
    all_event_types = set()
    for r in env_results:
        all_event_types.update(r["all_events"].keys())
    # Print header
    event_types = sorted(all_event_types)
    header = f"{'Condition':<25}" + "".join(f"{e[:15]:>16}" for e in event_types)
    print(header)
    print("-" * len(header))
    for r in env_results:
        label = f"{r['prompt']}+{r['memory'][:5]}"
        vals = "".join(f"{r['all_events'].get(e, 0):>16}" for e in event_types)
        print(f"{label:<25}{vals}")

# Save
summary_path = Path(OUTPUT_DIR) / "focused_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "config": {
            "model": MODEL_ID,
            "seeds": SEEDS,
            "conditions": {k: v for k, v in CONDITIONS.items()},
        },
        "results": results_table,
        "total_episodes": len(all_summaries),
        "total_successes": sum(1 for s in all_summaries if s.success),
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
