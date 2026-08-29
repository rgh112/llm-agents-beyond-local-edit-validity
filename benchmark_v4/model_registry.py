"""Model metadata used for family/scale reporting.

The benchmark runners accept arbitrary OpenAI-compatible model IDs. This
registry only annotates common IDs so analysis tables can report model family
and parameter scale explicitly.
"""
from __future__ import annotations

from typing import Dict


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "qwen/qwen3-8b": {
        "family": "Qwen3",
        "scale": "8B",
        "params_b": "8",
        "scale_bin": "small_7_9B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "current-generation small matched",
        "panel": "current_balanced",
    },
    "qwen/qwen3-14b": {
        "family": "Qwen3",
        "scale": "14B",
        "params_b": "14",
        "scale_bin": "medium_12_14B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "current-generation medium matched; uses /no_think compatibility",
        "panel": "current_balanced",
    },
    "qwen/qwen3-32b": {
        "family": "Qwen3",
        "scale": "32B",
        "params_b": "32",
        "scale_bin": "large_24_32B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "current-generation large matched; uses /no_think compatibility",
        "panel": "current_balanced",
    },
    "qwen/qwen3-30b-a3b-instruct-2507": {
        "family": "Qwen3",
        "scale": "30B-A3B",
        "params_b": "30",
        "scale_bin": "large_24_32B_moe",
        "openness": "open-weight/MoE",
        "provider": "OpenRouter",
        "role": "working Qwen large-bin MoE replacement",
        "panel": "current_balanced",
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507-Lora": {
        "family": "Qwen3",
        "scale": "30B-A3B",
        "params_b": "30",
        "scale_bin": "large_24_32B_moe",
        "openness": "open-weight/MoE",
        "provider": "Together",
        "role": "Qwen large-bin MoE fallback",
        "panel": "current_balanced",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "family": "Qwen2.5",
        "scale": "14B",
        "params_b": "14",
        "scale_bin": "medium_12_14B",
        "openness": "open-weight",
        "provider": "Together",
        "role": "Qwen medium-bin fallback",
        "panel": "current_balanced",
    },
    "qwen/qwen3-235b-a22b": {
        "family": "Qwen3",
        "scale": "235B-A22B",
        "params_b": "235",
        "scale_bin": "xl_70B_plus",
        "openness": "open-weight/MoE",
        "provider": "OpenRouter",
        "role": "current-generation XL focused",
        "panel": "xl_extension",
    },
    "qwen/qwen3.5-9b": {
        "family": "Qwen3.5",
        "scale": "9B",
        "params_b": "9",
        "scale_bin": "small_7_9B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "newer-generation small check",
        "panel": "newer_extension",
    },
    "qwen/qwen3.5-27b": {
        "family": "Qwen3.5",
        "scale": "27B",
        "params_b": "27",
        "scale_bin": "large_24_32B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "newer-generation large check",
        "panel": "newer_extension",
    },
    "qwen/qwen-2.5-7b-instruct": {
        "family": "Qwen2.5",
        "scale": "7B",
        "params_b": "7",
        "scale_bin": "small_7_9B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "matched small open-weight",
        "panel": "matched_small_large",
    },
    "meta-llama/llama-3.2-1b-instruct": {
        "family": "Llama3.2",
        "scale": "1B",
        "params_b": "1",
        "scale_bin": "tiny_1_4B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "tiny scale extension",
        "panel": "tiny_extension",
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "family": "Llama3.2",
        "scale": "3B",
        "params_b": "3",
        "scale_bin": "tiny_1_4B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "tiny scale extension",
        "panel": "tiny_extension",
    },
    "qwen/qwen-2.5-32b-instruct": {
        "family": "Qwen2.5",
        "scale": "32B",
        "params_b": "32",
        "scale_bin": "mid_24_32B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "mid-scale extension",
        "panel": "mid_extension",
    },
    "qwen/qwen-2.5-72b-instruct": {
        "family": "Qwen2.5",
        "scale": "72B",
        "params_b": "72",
        "scale_bin": "large_70_72B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "matched large open-weight",
        "panel": "matched_small_large",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "family": "Llama3.1",
        "scale": "8B",
        "params_b": "8",
        "scale_bin": "small_7_9B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "matched small open-weight",
        "panel": "matched_small_large",
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "family": "Llama3.3",
        "scale": "70B",
        "params_b": "70",
        "scale_bin": "large_70_72B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "matched large open-weight",
        "panel": "matched_small_large",
    },
    "mistralai/ministral-8b-2512": {
        "family": "Mistral",
        "scale": "8B",
        "params_b": "8",
        "scale_bin": "small_7_9B",
        "openness": "open-weight/API",
        "provider": "OpenRouter",
        "role": "small-family extension",
        "panel": "small_family_extension",
    },
    "mistralai/ministral-14b-2512": {
        "family": "Mistral",
        "scale": "14B",
        "params_b": "14",
        "scale_bin": "medium_12_14B",
        "openness": "open-weight/API",
        "provider": "OpenRouter",
        "role": "current-generation medium matched",
        "panel": "current_balanced",
    },
    "mistralai/ministral-3b-2512": {
        "family": "Mistral",
        "scale": "3B",
        "params_b": "3",
        "scale_bin": "tiny_1_4B",
        "openness": "open-weight/API",
        "provider": "OpenRouter",
        "role": "tiny scale extension",
        "panel": "tiny_extension",
    },
    "mistralai/mistral-small-3.2-24b-instruct": {
        "family": "Mistral",
        "scale": "24B",
        "params_b": "24",
        "scale_bin": "mid_24_32B",
        "openness": "open-weight/API",
        "provider": "OpenRouter",
        "role": "mid-scale extension",
        "panel": "mid_extension",
    },
    "mistralai/mistral-large-2411": {
        "family": "Mistral",
        "scale": "undisclosed-large",
        "params_b": "unknown",
        "scale_bin": "undisclosed_large",
        "openness": "closed/API",
        "provider": "OpenRouter",
        "role": "closed large sanity check",
        "panel": "closed_focused",
    },
    "google/gemma-2-9b-it": {
        "family": "Gemma2",
        "scale": "9B",
        "params_b": "9",
        "scale_bin": "small_7_9B",
        "openness": "open-weight",
        "provider": "Together",
        "role": "Gemma small-bin Together fallback",
        "panel": "current_balanced",
    },
    "google/gemma-3-4b-it": {
        "family": "Gemma3",
        "scale": "4B",
        "params_b": "4",
        "scale_bin": "tiny_1_4B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "tiny scale extension",
        "panel": "tiny_extension",
    },
    "google/gemma-3-12b-it": {
        "family": "Gemma3",
        "scale": "12B",
        "params_b": "12",
        "scale_bin": "medium_12_14B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "current-generation medium matched",
        "panel": "current_balanced",
    },
    "google/gemma-3-27b-it": {
        "family": "Gemma3",
        "scale": "27B",
        "params_b": "27",
        "scale_bin": "large_24_32B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "current-generation large matched",
        "panel": "current_balanced",
    },
    "google/gemma-2-27b-it": {
        "family": "Gemma2",
        "scale": "27B",
        "params_b": "27",
        "scale_bin": "mid_24_32B",
        "openness": "open-weight",
        "provider": "OpenRouter",
        "role": "mid-scale extension",
        "panel": "mid_extension",
    },
    "openai/gpt-4o-mini": {
        "family": "OpenAI",
        "scale": "undisclosed-mini",
        "params_b": "unknown",
        "scale_bin": "closed_mini",
        "openness": "closed/API",
        "provider": "OpenRouter/OpenAI",
        "role": "closed-family small-cost check",
        "panel": "closed_focused",
    },
    "openai/gpt-4o": {
        "family": "OpenAI",
        "scale": "undisclosed-frontier",
        "params_b": "unknown",
        "scale_bin": "closed_frontier",
        "openness": "closed/API",
        "provider": "OpenRouter/OpenAI",
        "role": "strong closed focused check",
        "panel": "closed_focused",
    },
    "openai/gpt-4.1-mini": {
        "family": "OpenAI",
        "scale": "undisclosed-mini",
        "params_b": "unknown",
        "scale_bin": "closed_mini",
        "openness": "closed/API",
        "provider": "OpenRouter/OpenAI",
        "role": "stronger mini focused check",
        "panel": "closed_focused",
    },
    "openai/gpt-4.1": {
        "family": "OpenAI",
        "scale": "undisclosed-frontier",
        "params_b": "unknown",
        "scale_bin": "closed_frontier",
        "openness": "closed/API",
        "provider": "OpenRouter/OpenAI",
        "role": "frontier focused check",
        "panel": "closed_focused",
    },
}


def get_model_metadata(model_id: str) -> Dict[str, str]:
    provider_override = ""
    lookup_id = model_id
    if ":" in model_id and model_id.split(":", 1)[0] in {"openrouter", "together"}:
        provider_override, lookup_id = model_id.split(":", 1)
    meta = MODEL_REGISTRY.get(lookup_id) or MODEL_REGISTRY.get(model_id)
    if meta:
        out = dict(meta)
        if provider_override:
            out["provider"] = provider_override
        return out
    lower = lookup_id.lower()
    if "qwen" in lower:
        family = "Qwen"
    elif "llama" in lower:
        family = "Llama"
    elif "mistral" in lower or "ministral" in lower:
        family = "Mistral"
    elif "gpt" in lower or "openai" in lower:
        family = "OpenAI"
    else:
        family = "unknown"
    return {
        "family": family,
        "scale": "unknown",
        "params_b": "unknown",
        "scale_bin": "unknown",
        "openness": "unknown",
        "provider": "unknown",
        "role": "unregistered",
        "panel": "unregistered",
    }
