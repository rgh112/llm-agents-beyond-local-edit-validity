#!/usr/bin/env python3
"""GB1 calibration mini-run: find operating point for fitness_target.

Tests fitness_target = {4.7, 4.8, 5.0} × 3 prompts × 10 seeds.
Goal: zero_shot 5-15%, few_shot_strategy 10-25%, self_check 5-20%.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv
from benchmark_v4.prompts import get_prompt_builder
from benchmark_v4.memory import get_memory_module
from benchmark_v4.models.api_model import APIModel
from benchmark_v4.runners.single_episode_runner import run_episode
from benchmark_v4.baselines.gb1_baselines import run_baseline as run_gb1_baseline

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY before running hosted experiments.")

FITNESS_TARGETS = [4.7, 4.8, 5.0]
PROMPTS = ["zero_shot", "few_shot_strategy", "self_check"]
SEEDS = list(range(10))
MEMORY = "state_only"

# Also run baselines for reference
BASELINE_STRATEGIES = ["random", "additive_greedy", "privileged_noisy_oracle", "privileged_epistasis"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/gb1_calibration_{timestamp}"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
memory = get_memory_module(MEMORY)

results = []
sweep_start = time.time()

for ft in FITNESS_TARGETS:
    print(f"\n{'='*60}")
    print(f"FITNESS TARGET = {ft}")
    print(f"{'='*60}")

    # --- Baselines first (fast) ---
    print(f"\n  Baselines (20 seeds each):")
    for strat in BASELINE_STRATEGIES:
        successes = 0
        for seed in range(20):
            env = GB1SequenceEnv(fitness_target=ft)
            summary = run_gb1_baseline(env, strat, seed)
            if summary["success"]:
                successes += 1
        sr = successes / 20
        print(f"    {strat:<25} SR={sr:.0%} ({successes}/20)")
        results.append({
            "type": "baseline", "fitness_target": ft,
            "strategy": strat, "SR": sr, "n": 20,
        })

    # --- LLM conditions ---
    for prompt_name in PROMPTS:
        prompt = get_prompt_builder(prompt_name)
        memory = get_memory_module(MEMORY)
        successes = 0
        total_steps = 0

        for seed in SEEDS:
            env = GB1SequenceEnv(fitness_target=ft)
            try:
                summary = run_episode(
                    env=env, prompt_builder=prompt, memory_module=memory,
                    model=model, seed=seed, verbose=False,
                )
                if summary.success:
                    successes += 1
                total_steps += len(summary.steps)
                status = "OK" if summary.success else "FAIL"
                print(f"    ft={ft}|{prompt_name}|s{seed} -> {status} "
                      f"({len(summary.steps)}steps, {summary.wall_time_s:.1f}s)")
            except Exception as e:
                print(f"    ft={ft}|{prompt_name}|s{seed} -> ERROR: {e}")

        sr = successes / len(SEEDS)
        avg_steps = total_steps / len(SEEDS)
        print(f"  >> ft={ft}|{prompt_name}: SR={sr:.0%} ({successes}/{len(SEEDS)}) "
              f"avg_steps={avg_steps:.1f}")
        results.append({
            "type": "llm", "fitness_target": ft,
            "prompt": prompt_name, "SR": sr,
            "n_success": successes, "n_total": len(SEEDS),
            "avg_steps": round(avg_steps, 1),
        })

sweep_time = time.time() - sweep_start

# Summary
print(f"\n{'='*70}")
print(f"GB1 CALIBRATION COMPLETE in {sweep_time:.0f}s")
print(f"{'='*70}")

for ft in FITNESS_TARGETS:
    print(f"\n  fitness_target = {ft}:")
    ft_results = [r for r in results if r["fitness_target"] == ft]
    for r in ft_results:
        if r["type"] == "baseline":
            print(f"    [baseline] {r['strategy']:<25} SR={r['SR']:.0%}")
        else:
            print(f"    [LLM]      {r['prompt']:<25} SR={r['SR']:.0%}")

# Save
with open(Path(OUTPUT_DIR) / "calibration_results.json", "w") as f:
    json.dump({"results": results, "wall_time_s": sweep_time}, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
