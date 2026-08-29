"""LLM API wrapper with retry, parsing, and token logging."""
import json
import time
import os
import ssl
import urllib.error
import urllib.request
from typing import List, Optional
from benchmark_v4.models.base_model import BaseModel, ModelResponse, Message

class APIModel(BaseModel):
    """Generic API model wrapper supporting OpenAI-compatible endpoints.

    Supports: Together API, OpenAI, Anthropic (via openai-compatible endpoint), etc.
    """

    def __init__(self, model_id: str, temperature: float = 0.0,
                 max_tokens: int = 512, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, max_retries: int = 3,
                 timeout: float = 60.0, top_p: Optional[float] = None):
        provider_prefix = None
        actual_model_id = model_id
        if model_id.startswith("together:"):
            provider_prefix = "together"
            actual_model_id = model_id.split(":", 1)[1]
        elif model_id.startswith("openrouter:"):
            provider_prefix = "openrouter"
            actual_model_id = model_id.split(":", 1)[1]
        super().__init__(actual_model_id, temperature, max_tokens, top_p=top_p)
        self.requested_model_id = model_id
        self.provider_prefix = provider_prefix
        self.max_retries = max_retries
        self.timeout = timeout

        # Auto-detect provider from model_id or use explicit base_url
        if provider_prefix == "together":
            self.base_url = "https://api.together.xyz/v1"
        elif provider_prefix == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        elif base_url:
            self.base_url = base_url
        elif "gpt" in actual_model_id.lower() or "o1" in actual_model_id.lower() or "o3" in actual_model_id.lower():
            self.base_url = "https://api.openai.com/v1"
        else:
            # Default to Together API
            self.base_url = "https://api.together.xyz/v1"

        if provider_prefix == "together":
            self.api_key = os.environ.get("TOGETHER_API_KEY") or api_key or ""
        elif provider_prefix == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or api_key or ""
        else:
            self.api_key = (
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("TOGETHER_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            )

        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _generate_via_http(self, msg_dicts) -> ModelResponse:
        url = self.base_url.rstrip("/") + "/chat/completions"
        msg_dicts = _compat_messages(self.model_id, self.base_url, msg_dicts)
        payload = {
            "model": self.model_id,
            "messages": msg_dicts,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        extra = _model_extra_body(self.model_id, self.base_url)
        if extra:
            payload.update(extra)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.0",
            },
            method="POST",
        )
        start = time.time()
        context = _ssl_context()
        with urllib.request.urlopen(req, timeout=self.timeout, context=context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.time() - start) * 1000

        usage_data = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_data.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_data.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_data.get("total_tokens", 0) or 0),
        }
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        return ModelResponse(
            raw_text=text,
            token_usage=usage,
            latency_ms=latency,
            model_id=self.model_id,
        )

    def generate(self, messages: List[Message]) -> ModelResponse:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        for attempt in range(self.max_retries):
            try:
                return self._generate_via_http(msg_dicts)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                        retry_after = e.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                wait = max(wait, 15.0 * (attempt + 1))
                        else:
                            wait = max(wait, 15.0 * (attempt + 1))
                    print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API error (final attempt): {e}")
                    return ModelResponse(
                        raw_text=f"[API_ERROR: {e}]",
                        token_usage={},
                        latency_ms=0,
                        model_id=self.model_id,
                    )

    def get_model_name(self) -> str:
        return self.model_id.split("/")[-1]


def _model_extra_body(model_id: str, base_url: str):
    """Provider/model-specific request compatibility settings."""
    return {}


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _compat_messages(model_id: str, base_url: str, msg_dicts):
    """Normalize provider/model output channels without changing task prompts."""
    if "openrouter.ai" in base_url and model_id in {
        "qwen/qwen3-8b",
        "qwen/qwen3-14b",
        "qwen/qwen3-32b",
    }:
        out = [dict(m) for m in msg_dicts]
        for msg in out:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if not content.lstrip().startswith("/no_think"):
                    msg["content"] = "/no_think\n" + content
                break
        return out
    return msg_dicts
