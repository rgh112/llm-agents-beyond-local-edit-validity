#!/usr/bin/env python3
"""Package A: Prompt disentanglement experiment.

Isolates format vs strategy effects of few-shot and self-check interventions.
3 envs × 6 conditions × 30 seeds = 540 episodes.
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

SEEDS = list(range(30))

# All conditions use M0 (state_only) to isolate prompt effect
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
OUTPUT_DIR = f"results_v4/disentangle_{timestamp}"

# ── Setup ──
model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

envs = {
    "word_ladder": lambda: WordLadderEnv(data_dir="wordladder_data", word_length=4, shortest_path_range=(3, 6)),
    "alloy": lambda: AlloyEnv(),
    "gb1_sequence": lambda: GB1SequenceEnv(),
}

total = len(envs) * len(PROMPT_CONDITIONS) * len(SEEDS)
combo_count = 0
results_table = []

print(f"=== Package A: Prompt Disentanglement ===")
print(f"  {len(envs)} envs × {len(PROMPT_CONDITIONS)} prompts × {len(SEEDS)} seeds = {total} episodes")
print(f"  Model: {MODEL_ID}")
print(f"  Memory: {MEMORY} (fixed)")
print(f"  Prompts: {PROMPT_CONDITIONS}")
print(f"  Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for env_name, env_factory in envs.items():
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

                if summary.finalize_step is not None:
                    finalize_steps.append(summary.finalize_step)
                if summary.first_failure_step is not None:
                    first_failures.append(summary.first_failure_step)

                status = "OK" if summary.success else "FAIL"
                print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|s{seed} "
                      f"-> {status} ({len(summary.steps)}steps, {summary.wall_time_s:.1f}s)")

            except Exception as e:
                print(f"  [{combo_count}/{total}] {env_name}|{prompt_name}|s{seed} -> ERROR: {e}")

        sr = successes / len(SEEDS)
        avg_steps = steps_total / len(SEEDS)
        meta = prompt.metadata

        results_table.append({
            "env": env_name,
            "prompt": prompt_name,
            "memory": MEMORY,
            "SR": sr,
            "n_success": successes,
            "n_total": len(SEEDS),
            "avg_steps": round(avg_steps, 1),
            "terminal_failures": dict(failures),
            "all_events": dict(events),
            "avg_finalize_step": round(sum(finalize_steps) / len(finalize_steps), 1) if finalize_steps else None,
            "avg_first_failure": round(sum(first_failures) / len(first_failures), 1) if first_failures else None,
            "prompt_metadata": meta,
        })

        print(f"  >> {env_name}|{prompt_name}: SR={sr:.0%} ({successes}/{len(SEEDS)}) "
              f"avg_steps={avg_steps:.1f}")
        print(f"     strategy_bearing={meta.get('strategy_bearing')} "
              f"task_specific={meta.get('task_specific')}")
        print(f"     failures={dict(failures)}")
        print()

sweep_time = time.time() - sweep_start

# ── Results ──
print()
print("=" * 90)
print(f"DISENTANGLEMENT COMPLETE: {combo_count} episodes in {sweep_time:.0f}s")
print("=" * 90)

for env_name in envs:
    print(f"\n{'='*25} {env_name.upper()} {'='*25}")
    env_results = [r for r in results_table if r["env"] == env_name]
    print(f"{'Prompt':<22} {'Strat':>5} {'Task':>5} {'SR':>8} {'Steps':>6} {'Fin':>5} {'1stF':>5}  Top Failures")
    print("-" * 95)
    for r in env_results:
        m = r["prompt_metadata"]
        top_fail = ", ".join(f"{k}:{v}" for k, v in
                            sorted(r["terminal_failures"].items(), key=lambda x: -x[1])[:3])
        fin = f"{r['avg_finalize_step']:.0f}" if r['avg_finalize_step'] else "-"
        ff = f"{r['avg_first_failure']:.0f}" if r['avg_first_failure'] else "-"
        print(f"{r['prompt']:<22} "
              f"{'Y' if m.get('strategy_bearing') else 'N':>5} "
              f"{'Y' if m.get('task_specific') else 'N':>5} "
              f"{r['n_success']:>2}/{r['n_total']:<3} ({r['SR']:>4.0%}) "
              f"{r['avg_steps']:>5.1f} "
              f"{fin:>5} {ff:>5}  {top_fail}")

# Key comparison
print(f"\n{'='*25} KEY COMPARISONS {'='*25}")
for env_name in envs:
    env_r = {r["prompt"]: r for r in results_table if r["env"] == env_name}
    zs = env_r.get("zero_shot", {}).get("SR", 0)
    ff = env_r.get("few_shot_format", {}).get("SR", 0)
    fs = env_r.get("few_shot_strategy", {}).get("SR", 0)
    sg = env_r.get("self_check_generic", {}).get("SR", 0)
    st = env_r.get("self_check", {}).get("SR", 0)
    print(f"\n{env_name}:")
    print(f"  Format effect:   few_shot_format({ff:.0%}) - zero_shot({zs:.0%}) = {ff-zs:+.0%}")
    print(f"  Strategy effect:  few_shot_strategy({fs:.0%}) - few_shot_format({ff:.0%}) = {fs-ff:+.0%}")
    print(f"  Generic check:   self_check_generic({sg:.0%}) - zero_shot({zs:.0%}) = {sg-zs:+.0%}")
    print(f"  Task rules:      self_check({st:.0%}) - self_check_generic({sg:.0%}) = {st-sg:+.0%}")

# Save
summary_path = Path(OUTPUT_DIR) / "disentangle_summary.json"
with open(summary_path, "w") as f:
    json.dump({
        "config": {
            "model": MODEL_ID,
            "seeds": SEEDS,
            "memory": MEMORY,
            "prompt_conditions": PROMPT_CONDITIONS,
            "environments": list(envs.keys()),
        },
        "results": results_table,
        "total_episodes": combo_count,
        "wall_time_s": sweep_time,
    }, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
