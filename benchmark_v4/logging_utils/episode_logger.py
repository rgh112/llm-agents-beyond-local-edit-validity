"""Episode logging utilities for structured experiment output."""
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_v4.failure_taxonomy import (
    EpisodeSummary, StepRecord, FailureEvent,
    FAILURE_CATEGORY_MAP, get_dominant_failure
)


class EpisodeLogger:
    """Logs episode data in structured JSON format."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.summary_dir = self.output_dir / "summaries"
        self.summary_dir.mkdir(exist_ok=True)

    def log_episode(self, summary: EpisodeSummary) -> str:
        """Save episode summary to JSON. Returns the file path."""
        regime = _safe_name(summary.regime)
        model = _safe_name(summary.model)
        setting = summary.final_metrics.get("sensitivity_setting")
        setting_part = f"_{_safe_name(setting)}" if setting else ""
        filename = (
            f"{summary.env_name}_{summary.prompt_condition}_"
            f"{summary.memory_condition}{setting_part}_{regime}_{model}_"
            f"seed{summary.seed}.json"
        )
        filepath = self.raw_dir / filename

        data = self._serialize_summary(summary)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath)

    def log_batch_summary(self, summaries: List[EpisodeSummary],
                          config: Dict) -> str:
        """Save batch summary with aggregate metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{timestamp}.json"
        filepath = self.summary_dir / filename

        batch_data = {
            "config": config,
            "timestamp": timestamp,
            "n_episodes": len(summaries),
            "aggregate": self._compute_aggregate(summaries),
            "episodes": [self._serialize_summary(s) for s in summaries]
        }

        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)

        return str(filepath)

    def _serialize_summary(self, summary: EpisodeSummary) -> Dict:
        """Convert EpisodeSummary to serializable dict."""
        steps_data = []
        for step in summary.steps:
            step_dict = {
                "t": step.t,
                "observation": step.observation,
                "raw_action": step.raw_action,
                "parsed_action": step.parsed_action,
                "valid": step.valid,
                "events": [e.name if isinstance(e, FailureEvent) else str(e) for e in step.events],
                "reward": step.reward,
                "info": step.info,
                "structure_before": step.structure_before,
                "structure_after": step.structure_after,
            }
            steps_data.append(step_dict)

        return {
            "env_name": summary.env_name,
            "regime": summary.regime,
            "model": summary.model,
            "prompt_condition": summary.prompt_condition,
            "memory_condition": summary.memory_condition,
            "seed": summary.seed,
            "steps": steps_data,
            "success": summary.success,
            "terminal_failure": summary.terminal_failure,
            "failure_events": [e if isinstance(e, str) else (e.name if isinstance(e, FailureEvent) else str(e)) for e in summary.failure_events],
            "final_metrics": summary.final_metrics,
            "token_usage": summary.token_usage,
            "wall_time_s": summary.wall_time_s
        }

    def _compute_aggregate(self, summaries: List[EpisodeSummary]) -> Dict:
        """Compute aggregate metrics from episode summaries."""
        if not summaries:
            return {}

        n = len(summaries)
        successes = sum(1 for s in summaries if s.success)
        total_steps = sum(len(s.steps) for s in summaries)

        # Failure type frequency
        failure_counts = {}
        all_events = []
        for s in summaries:
            for step in s.steps:
                for e in step.events:
                    event_name = e.name if isinstance(e, FailureEvent) else str(e)
                    failure_counts[event_name] = failure_counts.get(event_name, 0) + 1
                    all_events.append(e)

        # Terminal failure distribution
        terminal_failures = {}
        for s in summaries:
            if s.terminal_failure:
                tf = s.terminal_failure
                terminal_failures[tf] = terminal_failures.get(tf, 0) + 1

        return {
            "success_rate": successes / n,
            "n_episodes": n,
            "avg_steps": total_steps / n,
            "failure_event_counts": failure_counts,
            "terminal_failure_distribution": terminal_failures,
            "total_tokens": sum(
                s.token_usage.get("total_tokens", 0) for s in summaries
            ),
            "avg_wall_time_s": sum(s.wall_time_s for s in summaries) / n
        }


def _safe_name(value) -> str:
    return str(value).replace("/", "__").replace(":", "_").replace(" ", "_")
