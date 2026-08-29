"""Sampling-based controller wrappers."""
from __future__ import annotations

from typing import List

from benchmark_v4.controllers.base import BaseController, ControllerDecision
from benchmark_v4.controllers.scoring import simulate_one_step
from benchmark_v4.models.base_model import BaseModel, Message


class SelfConsistencyController(BaseController):
    """Sample K candidate actions and choose the best visible simulated result."""

    condition_name = "self_consistency"

    def __init__(self, k: int = 5, scorer: str = "proxy"):
        self.k = k
        self.scorer = scorer

    def select_action(self, *, env, prompt_builder, memory_module, model: BaseModel,
                      messages: List[Message], observation: str, step: int,
                      budget: int, env_name: str) -> ControllerDecision:
        candidates = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        seen = set()
        for i in range(self.k):
            response = model.generate(messages)
            raw = response.raw_text.strip()
            for key in usage:
                usage[key] += response.token_usage.get(key, 0)
            if raw in seen:
                continue
            seen.add(raw)
            _, info = simulate_one_step(env, raw, scorer=self.scorer)
            info["rank"] = i
            candidates.append(info)

        if not candidates:
            response = model.generate(messages)
            raw = response.raw_text
            for key in usage:
                usage[key] += response.token_usage.get(key, 0)
            candidates.append({"rank": 0, "raw_action": raw, "score": -999.0})

        best = max(candidates, key=lambda c: c.get("score", -999.0))
        return ControllerDecision(
            action=best["raw_action"],
            candidates=sorted(candidates, key=lambda c: c.get("score", -999.0), reverse=True),
            metadata={"token_usage": usage, "model_calls": self.k, "scorer": self.scorer},
        )


class BacktrackingController(SelfConsistencyController):
    """Candidate sampler with penalties for immediate loops and known bad moves.

    The environment has no explicit ROLLBACK action, so this is a conservative
    action-selection wrapper: it samples alternatives and downranks candidates
    that repeat exact failed actions or immediately reverse the last edit.
    """

    condition_name = "loop_avoidant"

    def select_action(self, **kwargs) -> ControllerDecision:
        decision = super().select_action(**kwargs)
        env = kwargs["env"]
        trajectory = env.get_trajectory()
        if not trajectory:
            return decision

        last = trajectory[-1]
        last_raw = (last.raw_action or "").strip().lower()
        last_parsed = last.parsed_action or {}
        rescored = []
        for cand in decision.candidates:
            raw = cand.get("raw_action", "").strip().lower()
            score = float(cand.get("score", -999.0))
            parsed = cand.get("parsed_action") or {}
            if raw == last_raw:
                score -= 3.0
            if _is_reverse(last_parsed, parsed):
                score -= 2.0
            if not last.valid and raw == last_raw:
                score -= 5.0
            new = dict(cand)
            new["score"] = score
            rescored.append(new)

        best = max(rescored, key=lambda c: c.get("score", -999.0))
        return ControllerDecision(
            action=best["raw_action"],
            candidates=sorted(rescored, key=lambda c: c.get("score", -999.0), reverse=True),
            metadata={**decision.metadata, "loop_penalty": True},
        )


def _is_reverse(a, b) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("type") != "edit" or b.get("type") != "edit":
        return False
    if "inc_element" in a:
        return (
            a.get("inc_element") == b.get("dec_element")
            and a.get("dec_element") == b.get("inc_element")
            and a.get("delta") == b.get("delta")
        )
    if "position" in a:
        return a.get("position") == b.get("position")
    if "pos" in a:
        return a.get("pos") == b.get("pos")
    return False


class ReflexionRetryController(BaseController):
    """One-step Reflexion-style retry after a visible simulator critique.

    This is an adaptation of reflection/self-refinement ideas to the benchmark's
    action-only interface: the real environment still receives exactly one
    EDIT/FINALIZE action, but the controller may ask the model for one revised
    action when the first proposal is invalid or visibly harmful under the
    non-oracle one-step simulator.
    """

    condition_name = "reflexion_retry"

    def __init__(self, scorer: str = "text_visible"):
        self.scorer = scorer

    def select_action(self, *, env, prompt_builder, memory_module, model: BaseModel,
                      messages: List[Message], observation: str, step: int,
                      budget: int, env_name: str) -> ControllerDecision:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        first_response = model.generate(messages)
        first_raw = first_response.raw_text.strip()
        for key in usage:
            usage[key] += first_response.token_usage.get(key, 0)
        _, first_info = simulate_one_step(env, first_raw, scorer=self.scorer)
        first_info["rank"] = 0
        first_info["reflection_attempt"] = False

        if not self._needs_retry(first_info):
            return ControllerDecision(
                action=first_raw,
                candidates=[first_info],
                metadata={
                    "token_usage": usage,
                    "model_calls": 1,
                    "scorer": self.scorer,
                    "retried": False,
                },
            )

        critique = self._build_reflection_prompt(
            raw_action=first_raw,
            info=first_info,
            step=step,
            budget=budget,
            env_name=env_name,
        )
        retry_messages = list(messages) + [
            Message(role="assistant", content=first_raw),
            Message(role="user", content=critique),
        ]
        retry_response = model.generate(retry_messages)
        retry_raw = retry_response.raw_text.strip()
        for key in usage:
            usage[key] += retry_response.token_usage.get(key, 0)
        _, retry_info = simulate_one_step(env, retry_raw, scorer=self.scorer)
        retry_info["rank"] = 1
        retry_info["reflection_attempt"] = True

        candidates = [first_info, retry_info]
        best = max(candidates, key=lambda row: row.get("score", -999.0))
        return ControllerDecision(
            action=best["raw_action"],
            candidates=sorted(candidates, key=lambda row: row.get("score", -999.0), reverse=True),
            metadata={
                "token_usage": usage,
                "model_calls": 2,
                "scorer": self.scorer,
                "retried": True,
                "selected_reflection_attempt": bool(best.get("reflection_attempt")),
            },
        )

    @staticmethod
    def _needs_retry(info) -> bool:
        if not info.get("valid", False):
            return True
        if info.get("done") and not info.get("success"):
            return True
        if info.get("score", 0.0) < info.get("progress_before", 0.0):
            return True
        parsed = info.get("parsed_action") or {}
        if parsed.get("type") == "finalize" and not info.get("success"):
            return True
        return False

    @staticmethod
    def _build_reflection_prompt(*, raw_action: str, info, step: int, budget: int,
                                 env_name: str) -> str:
        events = ", ".join(info.get("events") or []) or "none"
        valid = "valid" if info.get("valid") else "invalid"
        progress_before = info.get("progress_before")
        progress_after = info.get("progress_after")
        return (
            "Revise your previous action using the same action format. "
            "Return exactly one action and no explanation.\n"
            f"Environment: {env_name}\n"
            f"Step: {step}; remaining budget before action: {budget}\n"
            f"Previous action: {raw_action}\n"
            f"Visible simulator assessment: {valid}; events: {events}; "
            f"visible progress before={progress_before}, after={progress_after}.\n"
            "Choose an action that is locally admissible and preserves progress "
            "toward the delayed objective."
        )
