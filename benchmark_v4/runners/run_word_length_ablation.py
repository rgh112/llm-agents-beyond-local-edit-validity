#!/usr/bin/env python3
"""Package F: Word Ladder length ablation (appendix).

3 lengths × 2 prompts × 30 seeds = 180 episodes.
Justifies 4-letter as main operating point.
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.word_ladder import WordLadderEnv
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

WORD_LENGTHS = [3, 4, 5]
PROMPTS = ["zero_shot", "scaffold"]
SEEDS = list(range(30))
MEMORY = "state_only"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_v4/word_length_ablation_{timestamp}"

model = APIModel(model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY)
logger = EpisodeLogger(OUTPUT_DIR)

results_table = []
combo_count = 0
total = len(WORD_LENGTHS) * len(PROMPTS) * len(SEEDS)

print(f"=== Package F: Word Length Ablation ===")
print(f"  {len(WORD_LENGTHS)} lengths × {len(PROMPTS)} prompts × {len(SEEDS)} seeds = {total} episodes")
print(f"  Output: {OUTPUT_DIR}")
print()

sweep_start = time.time()

for wl in WORD_LENGTHS:
    # Check if pairs file exists
    pairs_file = f"wordladder_data/pairs_500_len{wl}.csv"
    if not os.path.exists(pairs_file):
        print(f"  WARNING: {pairs_file} not found, skipping length={wl}")
        continue

    for prompt_name in PROMPTS:
        prompt = get_prompt_builder(prompt_name)
        memory = get_memory_module(MEMORY)

        successes = 0
        steps_total = 0
        failures = Counter()

        for seed in SEEDS:
            combo_count += 1
            try:
                env = WordLadderEnv(data_dir="wordladder_data", word_length=wl,
                                    shortest_path_range=(3, 6))
            except Exception as e:
                print(f"  [{combo_count}/{total}] len={wl}|{prompt_name}|s{seed} -> ENV ERROR: {e}")
                continue

            if not env.pairs:
                print(f"  [{combo_count}/{total}] len={wl}|{prompt_name}|s{seed} -> NO PAIRS")
                continue

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

                status = "OK" if summary.success else "FAIL"
                print(f"  [{combo_count}/{total}] len={wl}|{prompt_name}|s{seed} "
                      f"-> {status} ({len(summary.steps)}steps)")

            except Exception as e:
                print(f"  [{combo_count}/{total}] len={wl}|{prompt_name}|s{seed} -> ERROR: {e}")

        n = len(SEEDS)
        sr = successes / n if n > 0 else 0
        avg_steps = steps_total / n if n > 0 else 0

        results_table.append({
            "word_length": wl,
            "prompt": prompt_name,
            "SR": sr,
            "n_success": successes,
            "n_total": n,
            "avg_steps": round(avg_steps, 1),
            "terminal_failures": dict(failures),
        })

        print(f"  >> len={wl}|{prompt_name}: SR={sr:.0%} ({successes}/{n}) avg_steps={avg_steps:.1f}")
        print()

sweep_time = time.time() - sweep_start

print(f"\n{'='*60}")
print(f"WORD LENGTH ABLATION COMPLETE in {sweep_time:.0f}s")
print(f"{'='*60}")

print(f"\n{'Length':>6} {'Prompt':<15} {'SR':>8} {'Steps':>6}")
print("-" * 40)
for r in results_table:
    print(f"  {r['word_length']:>4}   {r['prompt']:<15} "
          f"{r['n_success']:>2}/{r['n_total']:<3} ({r['SR']:>4.0%}) {r['avg_steps']:>5.1f}")

with open(Path(OUTPUT_DIR) / "word_length_ablation_summary.json", "w") as f:
    json.dump({"results": results_table, "wall_time_s": sweep_time}, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/")
