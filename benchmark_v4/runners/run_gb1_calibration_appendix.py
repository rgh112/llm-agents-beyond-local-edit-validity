#!/usr/bin/env python3
"""Package E: GB1 calibration appendix experiment.

Sweeps fitness_target × policies to justify operating point choice.
3 thresholds × 4 policies × 20 seeds = 240 episodes.
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
from benchmark_v4.baselines.observation_heuristics import run_gb1_obs_heuristic

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-Turbo"
BASE_URL = "https://api.together.xyz/v1"
API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY before running hosted experiments.")

FITNESS_TARGETS = [4.7, 4.8, 5.0]
SEEDS = list(range(20))
MEMORY = "state_only"

# LLM conditions
LLM_PROMPTS = ["zero_shot", "few_shot_strategy"]

# Baseline strategies (fast, no API)
BASELINE_STRATEGIES = ["additive_greedy", "privileged_noisy_oracle"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/gb1_calibration_appendix_{timestamp}"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)

results = []
sweep_start = time.time()

for ft in FITNESS_TARGETS:
    print(f"\n{'='*50}")
    print(f"  FITNESS TARGET = {ft}")
    print(f"{'='*50}")

    # Baselines
    for strat in BASELINE_STRATEGIES:
        ok = 0
        for seed in SEEDS:
            env = GB1SequenceEnv(fitness_target=ft)
            s = run_gb1_baseline(env, strat, seed)
            if s["success"]:
                ok += 1
        sr = ok / len(SEEDS)
        print(f"  [baseline] {strat:<25} ft={ft} SR={sr:.0%}")
        results.append({"type": "baseline", "fitness_target": ft, "policy": strat,
                        "SR": sr, "n": len(SEEDS)})

    # Observation heuristic
    ok = 0
    for seed in SEEDS:
        env = GB1SequenceEnv(fitness_target=ft)
        s = run_gb1_obs_heuristic(env, seed)
        if s["success"]:
            ok += 1
    sr = ok / len(SEEDS)
    print(f"  [obs_heur] obs_additive_greedy     ft={ft} SR={sr:.0%}")
    results.append({"type": "obs_heuristic", "fitness_target": ft, "policy": "obs_additive_greedy",
                    "SR": sr, "n": len(SEEDS)})

    # LLM
    for prompt_name in LLM_PROMPTS:
        prompt = get_prompt_builder(prompt_name)
        memory = get_memory_module(MEMORY)
        ok = 0
        for seed in SEEDS:
            env = GB1SequenceEnv(fitness_target=ft)
            try:
                summary = run_episode(
                    env=env, prompt_builder=prompt, memory_module=memory,
                    model=model, seed=seed, verbose=False,
                )
                if summary.success:
                    ok += 1
                print(f"    ft={ft}|{prompt_name}|s{seed} -> {'OK' if summary.success else 'FAIL'}")
            except Exception as e:
                print(f"    ft={ft}|{prompt_name}|s{seed} -> ERROR: {e}")

        sr = ok / len(SEEDS)
        print(f"  [LLM]      {prompt_name:<25} ft={ft} SR={sr:.0%}")
        results.append({"type": "llm", "fitness_target": ft, "policy": prompt_name,
                        "SR": sr, "n": len(SEEDS)})

sweep_time = time.time() - sweep_start

# Summary table
print(f"\n{'='*70}")
print(f"GB1 CALIBRATION APPENDIX COMPLETE in {sweep_time:.0f}s")
print(f"{'='*70}")
print(f"\n{'Policy':<30} {'ft=4.7':>8} {'ft=4.8':>8} {'ft=5.0':>8}")
print("-" * 60)
policies = sorted(set(r["policy"] for r in results))
for pol in policies:
    row = []
    for ft in FITNESS_TARGETS:
        r = next((x for x in results if x["policy"] == pol and x["fitness_target"] == ft), None)
        row.append(f"{r['SR']:.0%}" if r else "-")
    print(f"  {pol:<28} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

with open(Path(OUTPUT_DIR) / "calibration_appendix_summary.json", "w") as f:
    json.dump({"results": results, "wall_time_s": sweep_time}, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
