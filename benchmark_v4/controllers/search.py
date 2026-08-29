"""Shallow lookahead beam controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from benchmark_v4.controllers.base import BaseController, ControllerDecision
from benchmark_v4.controllers.scoring import progress_score, simulate_one_step
from benchmark_v4.models.base_model import BaseModel, Message


@dataclass
class _Node:
    env: Any
    first_action: str
    path: List[str]
    score: float


class LookaheadBeamController(BaseController):
    """Depth-limited tree search over LLM-proposed edits."""

    condition_name = "beam"

    def __init__(self, width: int = 3, depth: int = 2, samples_per_node: int = 3,
                 scorer: str = "proxy", condition_name: str = "beam"):
        self.width = width
        self.depth = depth
        self.samples_per_node = samples_per_node
        self.scorer = scorer
        self.condition_name = condition_name

    def _messages_for_env(self, *, env, prompt_builder, memory_module,
                          system_prompt: str) -> List[Message]:
        obs = env.render_observation()
        memory_context = memory_module.get_context(env.get_trajectory(), obs)
        step_prompt = prompt_builder.build_step_prompt(
            observation=obs,
            memory_context=memory_context,
            step=env.current_step + 1,
            budget=env.max_steps - env.current_step,
            env_name=env.env_name,
        )
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=step_prompt),
        ]

    def select_action(self, *, env, prompt_builder, memory_module, model: BaseModel,
                      messages: List[Message], observation: str, step: int,
                      budget: int, env_name: str) -> ControllerDecision:
        system_prompt = messages[0].content
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_calls = 0
        beam: List[_Node] = [
            _Node(env=env, first_action="", path=[], score=progress_score(env, scorer=self.scorer))
        ]
        all_candidates = []

        for depth_idx in range(self.depth):
            expanded: List[_Node] = []
            for node in beam:
                if node.env.is_done():
                    expanded.append(node)
                    continue
                node_messages = (
                    messages if depth_idx == 0
                    else self._messages_for_env(
                        env=node.env,
                        prompt_builder=prompt_builder,
                        memory_module=memory_module,
                        system_prompt=system_prompt,
                    )
                )
                seen = set()
                for i in range(self.samples_per_node):
                    response = model.generate(node_messages)
                    model_calls += 1
                    raw = response.raw_text.strip()
                    for key in usage:
                        usage[key] += response.token_usage.get(key, 0)
                    if raw in seen:
                        continue
                    seen.add(raw)
                    sim, info = simulate_one_step(node.env, raw, scorer=self.scorer)
                    first = node.first_action or raw
                    path = node.path + [raw]
                    score = info["score"] + 0.05 * depth_idx
                    expanded.append(_Node(env=sim, first_action=first, path=path, score=score))
                    if depth_idx == 0:
                        row = dict(info)
                        row["path"] = path
                        all_candidates.append(row)

            if not expanded:
                break
            expanded.sort(key=lambda n: n.score, reverse=True)
            beam = expanded[:self.width]

        best = max(beam, key=lambda n: n.score)
        if not all_candidates:
            response = model.generate(messages)
            model_calls += 1
            raw = response.raw_text
            for key in usage:
                usage[key] += response.token_usage.get(key, 0)
            return ControllerDecision(
                action=raw,
                candidates=[{"raw_action": raw, "score": -999.0}],
                metadata={"token_usage": usage, "model_calls": model_calls},
            )

        return ControllerDecision(
            action=best.first_action,
            candidates=sorted(all_candidates, key=lambda c: c.get("score", -999.0), reverse=True),
            metadata={
                "token_usage": usage,
                "model_calls": model_calls,
                "scorer": self.scorer,
                "best_path": best.path,
                "best_score": best.score,
                "method_family": "tree_of_thoughts_style" if self.condition_name == "tot_style_beam" else "beam_search",
            },
        )
