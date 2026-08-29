"""Single episode runner for constructive editing benchmark."""
import time
from collections import Counter
from typing import Optional

from benchmark_v4.envs.base_env import BaseConstructiveEditEnv as BaseEnv
from benchmark_v4.prompts.base_prompt import BasePromptBuilder
from benchmark_v4.memory.base_memory import BaseMemory
from benchmark_v4.models.base_model import BaseModel, Message
from benchmark_v4.failure_taxonomy import (
    EpisodeSummary, FailureEvent, get_dominant_failure
)


def run_episode(
    env: BaseEnv,
    prompt_builder: BasePromptBuilder,
    memory_module: BaseMemory,
    model: BaseModel,
    seed: int,
    verbose: bool = False,
    max_episode_steps: Optional[int] = None,
    **reset_kwargs,
) -> EpisodeSummary:
    start_time = time.time()
    memory_module.reset()

    obs = env.reset(seed=seed, **reset_kwargs)
    if max_episode_steps is not None:
        env.max_steps = min(env.max_steps, max_episode_steps)

    system_prompt = prompt_builder.build_system_prompt(
        env_name=env.env_name,
        task_description=env.task_description,
        constraints=env.constraints,
        action_format=env.action_format,
    )

    messages = [Message(role="system", content=system_prompt)]
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    while not env.is_done():
        memory_context = memory_module.get_context(env.get_trajectory(), obs)

        remaining = env.max_steps - env.current_step
        step_prompt = prompt_builder.build_step_prompt(
            observation=obs,
            memory_context=memory_context,
            step=env.current_step + 1,
            budget=remaining,
            env_name=env.env_name,
        )

        messages.append(Message(role="user", content=step_prompt))

        response = model.generate(messages)
        raw_output = response.raw_text

        for k in total_tokens:
            total_tokens[k] += response.token_usage.get(k, 0)

        if verbose:
            print(f"  Step {env.current_step + 1}: {raw_output.strip()[:80]}")

        result = env.step(raw_output)
        obs = result.observation

        messages.append(Message(role="assistant", content=raw_output))

        if env.current_step > env.max_steps + 5:
            break

    wall_time = time.time() - start_time

    # Collect all events and compute detailed analysis
    all_events = []
    event_counts = Counter()
    first_failure_step = None
    finalize_step = None

    for step in env.get_trajectory():
        all_events.extend(step.events)
        for e in step.events:
            ename = e.name if isinstance(e, FailureEvent) else str(e)
            event_counts[ename] += 1
            if first_failure_step is None:
                first_failure_step = step.t
        # Detect finalize
        if step.parsed_action and step.parsed_action.get("type") == "finalize":
            finalize_step = step.t

    dominant = get_dominant_failure(all_events)
    terminal_failure = dominant.name if dominant else None

    summary = EpisodeSummary(
        env_name=env.env_name,
        regime="constructive_editing",
        model=model.get_model_name(),
        prompt_condition=prompt_builder.condition_name,
        memory_condition=memory_module.condition_name,
        seed=seed,
        steps=env.get_trajectory(),
        success=env.is_success(),
        terminal_failure=terminal_failure if not env.is_success() else None,
        failure_events=[
            e.name if isinstance(e, FailureEvent) else str(e)
            for e in all_events
        ],
        final_metrics=env.get_episode_summary(),
        token_usage=total_tokens,
        wall_time_s=wall_time,
        event_counts=dict(event_counts),
        finalize_step=finalize_step,
        first_failure_step=first_failure_step,
        prompt_metadata=getattr(prompt_builder, 'metadata', {}),
    )

    return summary


def run_episode_with_controller(
    env: BaseEnv,
    prompt_builder: BasePromptBuilder,
    memory_module: BaseMemory,
    model: BaseModel,
    controller,
    seed: int,
    verbose: bool = False,
    max_episode_steps: Optional[int] = None,
    **reset_kwargs,
) -> EpisodeSummary:
    """Run one episode using a controller wrapper.

    The environment interface remains action-only. The controller may make
    multiple model calls internally and then returns a single EDIT/FINALIZE
    action to apply to the real environment.
    """
    start_time = time.time()
    memory_module.reset()

    obs = env.reset(seed=seed, **reset_kwargs)
    if max_episode_steps is not None:
        env.max_steps = min(env.max_steps, max_episode_steps)

    system_prompt = prompt_builder.build_system_prompt(
        env_name=env.env_name,
        task_description=env.task_description,
        constraints=env.constraints,
        action_format=env.action_format,
    )

    messages = [Message(role="system", content=system_prompt)]
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    controller_calls = 0

    while not env.is_done():
        memory_context = memory_module.get_context(env.get_trajectory(), obs)
        remaining = env.max_steps - env.current_step
        step_prompt = prompt_builder.build_step_prompt(
            observation=obs,
            memory_context=memory_context,
            step=env.current_step + 1,
            budget=remaining,
            env_name=env.env_name,
        )
        messages.append(Message(role="user", content=step_prompt))

        decision = controller.select_action(
            env=env,
            prompt_builder=prompt_builder,
            memory_module=memory_module,
            model=model,
            messages=messages,
            observation=obs,
            step=env.current_step + 1,
            budget=remaining,
            env_name=env.env_name,
        )
        raw_output = decision.action
        usage = decision.metadata.get("token_usage", {})
        for k in total_tokens:
            total_tokens[k] += usage.get(k, 0)
        controller_calls += int(decision.metadata.get("model_calls", 1))

        if verbose:
            print(f"  Step {env.current_step + 1}: {raw_output.strip()[:80]}")

        result = env.step(raw_output)
        obs = result.observation

        if env.trajectory:
            env.trajectory[-1].info["controller"] = controller.condition_name
            env.trajectory[-1].info["controller_candidates"] = decision.candidates[:10]
            env.trajectory[-1].info["controller_metadata"] = decision.metadata

        messages.append(Message(role="assistant", content=raw_output))

        if env.current_step > env.max_steps + 5:
            break

    wall_time = time.time() - start_time

    all_events = []
    event_counts = Counter()
    first_failure_step = None
    finalize_step = None

    for step in env.get_trajectory():
        all_events.extend(step.events)
        for e in step.events:
            ename = e.name if isinstance(e, FailureEvent) else str(e)
            event_counts[ename] += 1
            if first_failure_step is None:
                first_failure_step = step.t
        if step.parsed_action and step.parsed_action.get("type") == "finalize":
            finalize_step = step.t

    dominant = get_dominant_failure(all_events)
    terminal_failure = dominant.name if dominant else None

    return EpisodeSummary(
        env_name=env.env_name,
        regime=f"constructive_editing:{controller.condition_name}",
        model=model.get_model_name(),
        prompt_condition=prompt_builder.condition_name,
        memory_condition=memory_module.condition_name,
        seed=seed,
        steps=env.get_trajectory(),
        success=env.is_success(),
        terminal_failure=terminal_failure if not env.is_success() else None,
        failure_events=[
            e.name if isinstance(e, FailureEvent) else str(e)
            for e in all_events
        ],
        final_metrics=env.get_episode_summary(),
        token_usage={**total_tokens, "controller_model_calls": controller_calls},
        wall_time_s=wall_time,
        event_counts=dict(event_counts),
        finalize_step=finalize_step,
        first_failure_step=first_failure_step,
        prompt_metadata=getattr(prompt_builder, 'metadata', {}),
    )
