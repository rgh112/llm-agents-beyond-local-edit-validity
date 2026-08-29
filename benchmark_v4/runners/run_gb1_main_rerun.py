#!/usr/bin/env python3
"""GB1 main prompt re-run with final FINALIZE precheck.

gb1_sequence × 6 prompts × 50 seeds = 300 episodes.
Replaces GB1 portion of Package A.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv
from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.runners.single_episode_runner import run_episode
from benchmark_v4.logging_utils.episode_logger import EpisodeLogger

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY before running hosted experiments.")

SEEDS = list(range(50))
MEMORY = "state_only"

PROMPT_CONDITIONS = [
    "zero_shot",
    "few_shot_format",
    "few_shot_strategy",
    "scaffold",
    "self_check_generic",
    "self_check",
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/gb1_main_rerun_{timestamp}"

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

total = len(PROMPT_CONDITIONS) * len(SEEDS)
combo_count = 0
results_table = []

print(f"=== GB1 Main Prompt Re-run (final FINALIZE precheck) ===")
print(f"  6 prompts × {len(SEEDS)} seeds = {total} episodes")
print(f"  fitness_target: 5.0 (default)")
print(f"  Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for prompt_name in PROMPT_CONDITIONS:
    prompt = get_prompt_builder(prompt_name)
    memory = get_memory_module(MEMORY)

    successes = 0
    steps_total = 0
    failures = Counter()
    events = Counter()
    finalize_steps = []
    first_failures = []

    for seed in SEEDS:
        combo_count += 1
        env = GB1SequenceEnv()

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
            if summary.finalize_step is not None:
                finalize_steps.append(summary.finalize_step)
            if summary.first_failure_step is not None:
                first_failures.append(summary.first_failure_step)

            status = "OK" if summary.success else "FAIL"
            print(f"  [{combo_count}/{total}] {prompt_name}|s{seed} "
                  f"-> {status} ({len(summary.steps)}steps, {summary.wall_time_s:.1f}s)")

        except Exception as e:
            print(f"  [{combo_count}/{total}] {prompt_name}|s{seed} -> ERROR: {e}")

    sr = successes / len(SEEDS)
    avg_steps = steps_total / len(SEEDS)
    meta = prompt.metadata

    surface_events = {"MALFORMED_ACTION", "INVALID_POSITION", "INVALID_VALUE",
                      "INVALID_WORD", "ILLEGAL_EDIT", "REPEATED_EXACT_EDIT"}
    structural_events = {"BUDGET_UNAWARE_ACTION", "LOCAL_OPTIMUM_TRAP",
                         "OBJECTIVE_TRADEOFF_FAILURE", "PREMATURE_FINALIZE",
                         "HARD_CONSTRAINT_VIOLATION", "RECOVERY_COST_EXPLOSION",
                         "GLOBAL_FEASIBILITY_LOSS", "OSCILLATION"}

    surface_count = sum(events.get(e, 0) for e in surface_events)
    structural_count = sum(events.get(e, 0) for e in structural_events)

    results_table.append({
        "env": "gb1_sequence",
        "prompt": prompt_name,
        "memory": MEMORY,
        "SR": sr,
        "n_success": successes,
        "n_total": len(SEEDS),
        "avg_steps": round(avg_steps, 1),
        "terminal_failures": dict(failures),
        "all_events": dict(events),
        "surface_failure_count": surface_count,
        "structural_failure_count": structural_count,
        "avg_finalize_step": round(sum(finalize_steps) / len(finalize_steps), 1) if finalize_steps else None,
        "avg_first_failure": round(sum(first_failures) / len(first_failures), 1) if first_failures else None,
        "prompt_metadata": meta,
    })

    print(f"  >> {prompt_name}: SR={sr:.0%} ({successes}/{len(SEEDS)}) "
          f"avg_steps={avg_steps:.1f} surface={surface_count} structural={structural_count}")
    print()

sweep_time = time.time() - sweep_start

print(f"\n{'='*70}")
print(f"GB1 MAIN RE-RUN COMPLETE: {combo_count} episodes in {sweep_time:.0f}s")
print(f"{'='*70}")

print(f"\n{'Prompt':<22} {'Strat':>5} {'Task':>5} {'SR':>8} {'Steps':>6} "
      f"{'Surf':>5} {'Struct':>6}  Top Failures")
print("-" * 100)
for r in results_table:
    m = r["prompt_metadata"]
    top_fail = ", ".join(f"{k}:{v}" for k, v in
                        sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3])
    print(f"{r['prompt']:<22} "
          f"{'Y' if m.get('strategy_bearing') else 'N':>5} "
          f"{'Y' if m.get('task_specific') else 'N':>5} "
          f"{r['n_success']:>2}/{r['n_total']:<3} ({r['SR']:>4.0%}) "
          f"{r['avg_steps']:>5.1f} "
          f"{r['surface_failure_count']:>5} {r['structural_failure_count']:>6}  {top_fail}")

# Save
with open(Path(OUTPUT_DIR) / "gb1_main_rerun_summary.json", "w") as f:
    json.dump({
        "config": {
            "model": MODEL_ID,
            "seeds": SEEDS,
            "memory": MEMORY,
            "prompt_conditions": PROMPT_CONDITIONS,
            "fitness_target": 5.0,
            "note": "Re-run with final FINALIZE precheck (stability-only gate)",
        },
        "results": results_table,
        "total_episodes": combo_count,
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
