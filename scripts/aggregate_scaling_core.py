#!/usr/bin/env python3
"""Aggregate the May-9 core-breadth scaling tables into a single normalized JSON.

Reads `results_v4/core_breadth_20260509/rebuttal_table_*.json` and emits
`supplementary/scaling_core_summary.json` with one row per
(family, model, size_b, env, prompt) and selected metrics. The de-duplicated
union of the four blocks gives 13 distinct models in three families plus the
Llama scaling ladder.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "results_v4" / "core_breadth_20260509"
OUT = ROOT / "supplementary" / "scaling_core_summary.json"

# Map model id -> (family, size_b, scale_bin)
MODEL_META = {
    # Qwen
    "qwen3-8b": ("Qwen3", 8, "small_7_9B"),
    "qwen3-14b": ("Qwen3", 14, "medium_12_15B"),
    "qwen3-32b": ("Qwen3", 32, "large_24_32B"),
    # Mistral / Ministral
    "ministral-3b-2512": ("Ministral", 3, "tiny_3_4B"),
    "ministral-8b-2512": ("Ministral", 8, "small_7_9B"),
    "ministral-14b-2512": ("Ministral", 14, "medium_12_15B"),
    "mistral-small-3.2-24b-instruct": ("Mistral", 24, "large_24_32B"),
    # Gemma
    "gemma-3-4b-it": ("Gemma3", 4, "tiny_3_4B"),
    "gemma-3-12b-it": ("Gemma3", 12, "medium_12_15B"),
    "gemma-3-27b-it": ("Gemma3", 27, "large_24_32B"),
    # Llama
    "llama-3.2-3b-instruct": ("Llama", 3, "tiny_3_4B"),
    "llama-3.1-8b-instruct": ("Llama", 8, "small_7_9B"),
    "llama-3.3-70b-instruct": ("Llama", 70, "xl_70B"),
}

ENV_SHORT = {"word_ladder": "Word Ladder", "alloy": "Alloy", "gb1_sequence": "GB1"}
PROMPT_SHORT = {"zero_shot": "Zero-shot", "scaffold": "Scaffold", "self_check": "Self-check", "few_shot_format": "Few-shot"}


def _short_model(m: str) -> str:
    return m.split("/")[-1]


def load_table(path: Path) -> list:
    d = json.load(open(path))
    rows = []
    for r in d["rows"]:
        model = _short_model(r["model"])
        if model not in MODEL_META:
            continue
        family, size_b, scale_bin = MODEL_META[model]
        rows.append({
            "table": path.stem.replace("rebuttal_table_", ""),
            "model": model,
            "family": family,
            "size_b": size_b,
            "scale_bin": scale_bin,
            "env": ENV_SHORT.get(r["env"], r["env"]),
            "prompt": PROMPT_SHORT.get(r["prompt"], r["prompt"]),
            "n": r["n"],
            "n_success": r["n_success"],
            "SR": r["SR"],
            "SR_ci95": r.get("SR_ci95"),
            "local_valid": r.get("local_valid_edit_rate"),
            "surface_clean_success": r.get("SR_given_surface_clean"),
        })
    return rows


def main():
    blocks = ["tiny_plus_small_core", "qmg_scaling_core", "large_block_core", "llama_scaling_core"]
    all_rows = []
    for b in blocks:
        all_rows.extend(load_table(CORE / f"rebuttal_table_{b}.json"))
    # de-duplicate on (model, env, prompt), preferring later (qmg / large overrides tiny_plus_small for shared 8B cells).
    by_key = {}
    block_order = {b: i for i, b in enumerate(blocks)}
    for r in all_rows:
        key = (r["model"], r["env"], r["prompt"])
        if key not in by_key or block_order[r["table"]] >= block_order[by_key[key]["table"]]:
            by_key[key] = r
    rows = sorted(by_key.values(), key=lambda r: (r["family"], r["size_b"], r["env"], r["prompt"]))

    payload = {
        "blocks": blocks,
        "models": sorted({r["model"] for r in rows}),
        "families": sorted({r["family"] for r in rows}),
        "scale_bins": sorted({r["scale_bin"] for r in rows}),
        "n_rows": len(rows),
        "n_episodes": sum(r["n"] for r in rows),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    print(f"  {len(rows)} cells, {payload['n_episodes']} episodes, "
          f"{len(payload['models'])} models, {len(payload['families'])} families")


if __name__ == "__main__":
    main()
