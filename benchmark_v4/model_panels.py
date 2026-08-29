"""Pre-registered model panels for robust cross-model experiments."""
from __future__ import annotations

from typing import Iterable, List


MODEL_PANELS = {
    # Main fair family comparison: hold scale approximately fixed.
    "small_block": [
        "openrouter:qwen/qwen3-8b",
        "openrouter:mistralai/ministral-8b-2512",
        "openrouter:meta-llama/llama-3.1-8b-instruct",
    ],
    # Main scaling comparison across families with available matched bins.
    "scaling_qmg": [
        "openrouter:qwen/qwen3-8b",
        "openrouter:mistralai/ministral-8b-2512",
        "openrouter:google/gemma-3-4b-it",
        "openrouter:qwen/qwen3-14b",
        "openrouter:mistralai/ministral-14b-2512",
        "openrouter:google/gemma-3-12b-it",
        "openrouter:qwen/qwen3-32b",
        "openrouter:mistralai/mistral-small-3.2-24b-instruct",
        "openrouter:google/gemma-3-27b-it",
    ],
    # Full current OpenRouter balanced-incomplete design for the main prompt grid.
    "current_balanced": [
        "openrouter:qwen/qwen3-8b",
        "openrouter:mistralai/ministral-8b-2512",
        "openrouter:meta-llama/llama-3.1-8b-instruct",
        "openrouter:google/gemma-3-4b-it",
        "openrouter:qwen/qwen3-14b",
        "openrouter:mistralai/ministral-14b-2512",
        "openrouter:google/gemma-3-12b-it",
        "openrouter:qwen/qwen3-32b",
        "openrouter:mistralai/mistral-small-3.2-24b-instruct",
        "openrouter:google/gemma-3-27b-it",
    ],
    # Llama has a different scale ladder, so keep this separate.
    "llama_scaling": [
        "openrouter:meta-llama/llama-3.2-3b-instruct",
        "openrouter:meta-llama/llama-3.1-8b-instruct",
        "openrouter:meta-llama/llama-3.3-70b-instruct",
    ],
    # Tiny block is useful as a low-capability boundary check.
    "tiny_block": [
        "openrouter:mistralai/ministral-3b-2512",
        "openrouter:google/gemma-3-4b-it",
        "openrouter:meta-llama/llama-3.2-3b-instruct",
    ],
    # XL/MoE/closed models are boundary checks, not parameter-matched evidence.
    "xl_boundary": [
        "openrouter:qwen/qwen3-235b-a22b",
        "openrouter:meta-llama/llama-3.3-70b-instruct",
    ],
    "closed_boundary": [
        "openrouter:openai/gpt-4o-mini",
        "openrouter:openai/gpt-4o",
    ],
    # Provider-checked smoke panel: every currently selected model used by the
    # robust design, excluding the optional expensive XL/MoE boundary.
    "provider_checked_smoke": [
        "openrouter:mistralai/ministral-3b-2512",
        "openrouter:google/gemma-3-4b-it",
        "openrouter:meta-llama/llama-3.2-3b-instruct",
        "openrouter:qwen/qwen3-8b",
        "openrouter:mistralai/ministral-8b-2512",
        "openrouter:meta-llama/llama-3.1-8b-instruct",
        "openrouter:qwen/qwen3-14b",
        "openrouter:mistralai/ministral-14b-2512",
        "openrouter:google/gemma-3-12b-it",
        "openrouter:qwen/qwen3-32b",
        "openrouter:mistralai/mistral-small-3.2-24b-instruct",
        "openrouter:google/gemma-3-27b-it",
        "openrouter:meta-llama/llama-3.3-70b-instruct",
        "openrouter:openai/gpt-4o-mini",
        "openrouter:openai/gpt-4o",
    ],
}


def available_panels() -> List[str]:
    return sorted(MODEL_PANELS)


def expand_model_selection(models: Iterable[str] | None, panel: str | None) -> List[str]:
    selected: List[str] = []
    if panel:
        if panel not in MODEL_PANELS:
            raise ValueError(f"Unknown model panel {panel!r}; choose from {available_panels()}")
        selected.extend(MODEL_PANELS[panel])
    if models:
        selected.extend(models)
    if not selected:
        raise ValueError("Provide --models or --model-panel.")
    out: List[str] = []
    seen = set()
    for model_id in selected:
        if model_id not in seen:
            out.append(model_id)
            seen.add(model_id)
    return out
