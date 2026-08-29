#!/usr/bin/env python3
"""MCTS-style non-LLM search baselines for constructive editing.

These runs are diagnostic boundary checks, not LLM-agent leaderboard results.
They use the environment transition functions to enumerate valid edits and
compare visible-proxy rollouts against privileged terminal-objective rollouts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.alloy import AlloyEnv, ELEMENTS
from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv


Action = Hashable
State = Hashable


def parse_seed_spec(spec: str) -> List[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class MCTSNode:
    state: State
    parent: Optional["MCTSNode"] = None
    action: Optional[Action] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Action, "MCTSNode"] = field(default_factory=dict)
    untried: Optional[List[Action]] = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class DomainAdapter:
    env_name: str
    max_steps: int

    def reset(self, seed: int) -> State:
        raise NotImplementedError

    def actions(self, state: State) -> List[Action]:
        raise NotImplementedError

    def transition(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def is_success(self, state: State) -> bool:
        raise NotImplementedError

    def t(self, state: State) -> int:
        raise NotImplementedError

    def value(self, state: State, mode: str) -> float:
        raise NotImplementedError

    def action_text(self, action: Action) -> str:
        raise NotImplementedError

    def terminal_failure(self, state: State) -> str:
        raise NotImplementedError

    def final_metrics(self, state: State) -> Dict[str, Any]:
        raise NotImplementedError


class GB1Adapter(DomainAdapter):
    env_name = "gb1_sequence"

    def __init__(self):
        self.env = GB1SequenceEnv()
        self.max_steps = self.env.max_steps
        self._action_cache: Dict[State, List[Action]] = {}
        self._value_cache: Dict[Tuple[State, str], float] = {}

    def reset(self, seed: int) -> State:
        self.env.reset(seed)
        seq = tuple(self.env.state.structure)
        stability = round(float(self.env.state.latent["stability"]), 4)
        return (seq, stability, 0)

    def actions(self, state: State) -> List[Action]:
        if state in self._action_cache:
            return self._action_cache[state]
        seq, stability, t = state
        if self.is_success(state) or int(t) >= self.max_steps:
            out: List[Action] = []
        else:
            out = list(self.env._available_edits_for(list(seq), float(stability)))
        self._action_cache[state] = out
        return out

    def transition(self, state: State, action: Action) -> State:
        seq, stability, t = state
        pos, aa = action
        nxt = list(seq)
        nxt[pos] = aa
        nxt_stability = self.env._next_stability(list(seq), float(stability), int(pos), str(aa))
        return (tuple(nxt), round(float(nxt_stability), 4), int(t) + 1)

    def is_success(self, state: State) -> bool:
        seq, stability, _ = state
        return self.env._success_for(list(seq), float(stability))

    def t(self, state: State) -> int:
        return int(state[2])

    def value(self, state: State, mode: str) -> float:
        key = (state, mode)
        if key in self._value_cache:
            return self._value_cache[key]
        seq, stability, _ = state
        seq_l = list(seq)
        if mode == "privileged":
            if self.is_success(state):
                val = 1.0
            else:
                val = 0.75 * (self.env._true_fitness(seq_l) / max(self.env._fitness_target, 1.0))
                val += 0.25 * float(stability)
        else:
            proxy = self.env._local_proxy_score(seq_l, float(stability))
            val = 0.5 + 0.18 * proxy
        out = clamp01(val)
        self._value_cache[key] = out
        return out

    def action_text(self, action: Action) -> str:
        pos, aa = action
        return f"EDIT {pos} {aa}"

    def terminal_failure(self, state: State) -> str:
        seq, stability, _ = state
        if float(stability) < self.env._stability_target:
            return "HARD_CONSTRAINT_VIOLATION"
        true_fit = self.env._true_fitness(list(seq))
        est_fit = self.env._estimated_fitness(list(seq))
        if est_fit >= self.env._fitness_target or est_fit > true_fit:
            return "LOCAL_OPTIMUM_TRAP"
        return "GLOBAL_FEASIBILITY_LOSS"

    def final_metrics(self, state: State) -> Dict[str, Any]:
        seq, stability, t = state
        seq_l = list(seq)
        return {
            "variant": "".join(seq_l),
            "true_fitness": round(self.env._true_fitness(seq_l), 4),
            "estimated_fitness": round(self.env._estimated_fitness(seq_l), 4),
            "stability": round(float(stability), 4),
            "steps": int(t),
        }


class AlloyAdapter(DomainAdapter):
    env_name = "alloy"

    def __init__(self):
        self.env = AlloyEnv()
        self.max_steps = self.env.max_steps
        self._action_cache: Dict[State, List[Action]] = {}
        self._value_cache: Dict[Tuple[State, str], float] = {}
        self._transition_cache: Dict[Tuple[State, Action], State] = {}
        self._uts_cache: Dict[Tuple[float, ...], float] = {}
        self._density_cache: Dict[Tuple[float, ...], float] = {}
        self._distance_cache: Dict[Tuple[float, ...], float] = {}

    def reset(self, seed: int) -> State:
        self.env.reset(seed)
        comp = tuple(round(float(self.env.state.structure[el]), 3) for el in ELEMENTS)
        density = self.env._calculate_density(dict(self.env.state.structure))
        return (comp, 0.0, (round(float(density), 4),), 0)

    def _comp_dict(self, state: State) -> Dict[str, float]:
        comp, _, _, _ = state
        return {el: float(comp[i]) for i, el in enumerate(ELEMENTS)}

    def _dict_from_tuple(self, comp: Tuple[float, ...]) -> Dict[str, float]:
        return {el: float(comp[i]) for i, el in enumerate(ELEMENTS)}

    def _predict_uts_tuple(self, comp: Tuple[float, ...]) -> float:
        if comp not in self._uts_cache:
            self._uts_cache[comp] = self.env._predict_uts(self._dict_from_tuple(comp))
        return self._uts_cache[comp]

    def _density_tuple(self, comp: Tuple[float, ...]) -> float:
        if comp not in self._density_cache:
            self._density_cache[comp] = self.env._calculate_density(self._dict_from_tuple(comp))
        return self._density_cache[comp]

    def _distance_tuple(self, comp: Tuple[float, ...]) -> float:
        if comp not in self._distance_cache:
            self._distance_cache[comp] = self.env._distance_from_good_region(self._dict_from_tuple(comp))
        return self._distance_cache[comp]

    def _local_proxy_tuple(self, comp: Tuple[float, ...], recovery_cost: float) -> float:
        uts_margin = (self._predict_uts_tuple(comp) - self.env._uts_target) / max(self.env._uts_target, 1.0)
        density_margin = (self.env._density_target - self._density_tuple(comp)) / max(self.env._density_target, 1.0)
        rc_margin = (self.env._rc_threshold - recovery_cost) / max(self.env._rc_threshold, 1.0)
        return uts_margin + density_margin + 0.25 * rc_margin

    def actions(self, state: State) -> List[Action]:
        if state in self._action_cache:
            return self._action_cache[state]
        comp, _, _, t = state
        if self.is_success(state) or int(t) >= self.max_steps:
            out: List[Action] = []
        else:
            out = list(self.env._simulate_valid_edits({el: float(comp[i]) for i, el in enumerate(ELEMENTS)}, int(t)))
        self._action_cache[state] = out
        return out

    def transition(self, state: State, action: Action) -> State:
        key = (state, action)
        if key in self._transition_cache:
            return self._transition_cache[key]
        comp, rc, density_traj, t = state
        inc_el, dec_el, delta = action
        idx_inc = ELEMENTS.index(inc_el)
        idx_dec = ELEMENTS.index(dec_el)
        nxt = list(comp)
        old_comp = tuple(float(x) for x in comp)
        nxt[idx_inc] = round(float(nxt[idx_inc]) + float(delta), 3)
        nxt[idx_dec] = round(float(nxt[idx_dec]) - float(delta), 3)
        nxt_comp = tuple(float(x) for x in nxt)
        prev_distance = self._distance_tuple(old_comp)
        new_distance = self._distance_tuple(nxt_comp)
        nxt_rc = float(rc)
        if new_distance > prev_distance:
            nxt_rc += self.env._rc_rate * (new_distance - prev_distance)
        elif nxt_rc > 0:
            nxt_rc = max(0.0, nxt_rc - self.env._rc_rate * (prev_distance - new_distance) * 0.3)
        density_tuple = tuple(list(density_traj) + [round(self._density_tuple(nxt_comp), 4)])
        out = (nxt_comp, round(float(nxt_rc), 4), density_tuple, int(t) + 1)
        self._transition_cache[key] = out
        return out

    def is_success(self, state: State) -> bool:
        comp, rc, density_traj, _ = state
        true_uts = self._predict_uts_tuple(tuple(comp))
        true_density = self._density_tuple(tuple(comp))
        max_density_seen = max(density_traj) if density_traj else true_density
        overshoot = max(0, max_density_seen - self.env._density_target)
        effective_uts_target = self.env._uts_target * (1.0 + overshoot * self.env._overshoot_penalty)
        return (
            true_uts >= effective_uts_target
            and true_density <= self.env._density_target
            and float(rc) < self.env._rc_threshold
        )

    def t(self, state: State) -> int:
        return int(state[3])

    def value(self, state: State, mode: str) -> float:
        key = (state, mode)
        if key in self._value_cache:
            return self._value_cache[key]
        comp, rc, density_traj, _ = state
        comp_tuple = tuple(comp)
        proxy = self._local_proxy_tuple(comp_tuple, float(rc))
        if mode == "privileged":
            if self.is_success(state):
                val = 1.0
            else:
                uts = self._predict_uts_tuple(comp_tuple)
                density = self._density_tuple(comp_tuple)
                uts_ratio = uts / max(self.env._uts_target, 1.0)
                density_ok = 1.0 - max(0.0, density - self.env._density_target) / max(self.env._density_target, 1.0)
                rc_ok = 1.0 - max(0.0, float(rc) - self.env._rc_threshold) / max(self.env._rc_threshold, 1.0)
                overshoot = max(0.0, max(density_traj) - self.env._density_target)
                val = 0.45 * min(1.2, uts_ratio) + 0.25 * density_ok + 0.20 * rc_ok + 0.10 * proxy
                val -= 0.15 * overshoot
        else:
            val = 0.5 + 0.18 * proxy
        out = clamp01(val)
        self._value_cache[key] = out
        return out

    def action_text(self, action: Action) -> str:
        inc, dec, delta = action
        return f"EDIT {inc} {dec} {delta}"

    def terminal_failure(self, state: State) -> str:
        comp, rc, density_traj, _ = state
        comp_tuple = tuple(comp)
        uts = self._predict_uts_tuple(comp_tuple)
        density = self._density_tuple(comp_tuple)
        if float(rc) >= self.env._rc_threshold:
            return "RECOVERY_COST_EXPLOSION"
        if density > self.env._density_target or max(density_traj) > self.env._density_target:
            return "HARD_CONSTRAINT_VIOLATION"
        if uts < self.env._uts_target:
            return "OBJECTIVE_TRADEOFF_FAILURE"
        return "GLOBAL_FEASIBILITY_LOSS"

    def final_metrics(self, state: State) -> Dict[str, Any]:
        comp, rc, density_traj, t = state
        comp_tuple = tuple(comp)
        comp_dict = self._dict_from_tuple(comp_tuple)
        return {
            "composition": {el: round(comp_dict[el], 3) for el in ELEMENTS},
            "uts": round(self._predict_uts_tuple(comp_tuple), 4),
            "density": round(self._density_tuple(comp_tuple), 4),
            "max_density_seen": round(max(density_traj), 4),
            "recovery_cost": round(float(rc), 4),
            "steps": int(t),
        }


def make_adapter(env_name: str) -> DomainAdapter:
    if env_name == "gb1_sequence":
        return GB1Adapter()
    if env_name == "alloy":
        return AlloyAdapter()
    raise ValueError(f"unsupported MCTS env: {env_name}")


class MCTSPlanner:
    def __init__(
        self,
        adapter: DomainAdapter,
        mode: str,
        simulations: int,
        exploration: float,
        rollout_policy: str,
        rollout_top_k: int,
        rng: random.Random,
    ):
        self.adapter = adapter
        self.mode = mode
        self.simulations = simulations
        self.exploration = exploration
        self.rollout_policy = rollout_policy
        self.rollout_top_k = rollout_top_k
        self.rng = rng
        self.expanded_nodes = 0
        self.rollout_steps = 0

    def plan(self, root_state: State) -> Tuple[Optional[Action], Dict[str, Any]]:
        root = MCTSNode(root_state)
        root.untried = list(self.adapter.actions(root_state))
        if not root.untried:
            return None, {"root_actions": 0, "root_visits": 0, "expanded_nodes": 0}
        for _ in range(self.simulations):
            node = self._select_expand(root)
            reward = self._rollout(node.state)
            self._backpropagate(node, reward)
        if not root.children:
            return None, {"root_actions": len(root.untried), "root_visits": root.visits, "expanded_nodes": self.expanded_nodes}
        # Execute the most visited action; tie-break by value.
        best_action, best_child = max(
            root.children.items(), key=lambda item: (item[1].visits, item[1].value)
        )
        return best_action, {
            "root_actions": len(root.children) + len(root.untried or []),
            "root_visits": root.visits,
            "best_action_visits": best_child.visits,
            "best_action_value": round(best_child.value, 4),
            "expanded_nodes": self.expanded_nodes,
            "rollout_steps": self.rollout_steps,
        }

    def _select_expand(self, node: MCTSNode) -> MCTSNode:
        while True:
            if self.adapter.is_success(node.state) or self.adapter.t(node.state) >= self.adapter.max_steps:
                return node
            if node.untried is None:
                node.untried = list(self.adapter.actions(node.state))
            if node.untried:
                idx = self.rng.randrange(len(node.untried))
                action = node.untried.pop(idx)
                child_state = self.adapter.transition(node.state, action)
                child = MCTSNode(child_state, parent=node, action=action)
                child.untried = list(self.adapter.actions(child_state))
                node.children[action] = child
                self.expanded_nodes += 1
                return child
            if not node.children:
                return node
            node = self._uct_child(node)

    def _uct_child(self, node: MCTSNode) -> MCTSNode:
        log_parent = math.log(max(2, node.visits))

        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            return child.value + self.exploration * math.sqrt(log_parent / child.visits)

        return max(node.children.values(), key=score)

    def _rollout(self, state: State) -> float:
        cur = state
        while not self.adapter.is_success(cur) and self.adapter.t(cur) < self.adapter.max_steps:
            actions = self.adapter.actions(cur)
            if not actions:
                break
            action = self._rollout_action(cur, actions)
            cur = self.adapter.transition(cur, action)
            self.rollout_steps += 1
        return self.adapter.value(cur, self.mode)

    def _rollout_action(self, state: State, actions: Sequence[Action]) -> Action:
        if self.rollout_policy == "random":
            return self.rng.choice(list(actions))
        ranked = sorted(
            actions,
            key=lambda a: self.adapter.value(self.adapter.transition(state, a), self.mode),
            reverse=True,
        )
        top = ranked[: max(1, min(self.rollout_top_k, len(ranked)))]
        if self.rollout_policy == "best":
            return top[0]
        return self.rng.choice(top)

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent


def run_episode(
    adapter: DomainAdapter,
    seed: int,
    mode: str,
    simulations: int,
    exploration: float,
    rollout_policy: str,
    rollout_top_k: int,
) -> Dict[str, Any]:
    rng_seed = (seed + 1) * 1000003 + simulations * 9176 + (0 if mode == "visible" else 1)
    rng = random.Random(rng_seed)
    state = adapter.reset(seed)
    trace: List[Dict[str, Any]] = []
    total_expanded = 0
    total_rollout_steps = 0
    for _ in range(adapter.max_steps):
        if adapter.is_success(state):
            break
        planner = MCTSPlanner(
            adapter=adapter,
            mode=mode,
            simulations=simulations,
            exploration=exploration,
            rollout_policy=rollout_policy,
            rollout_top_k=rollout_top_k,
            rng=rng,
        )
        action, info = planner.plan(state)
        total_expanded += int(info.get("expanded_nodes", 0))
        total_rollout_steps += int(info.get("rollout_steps", 0))
        if action is None:
            trace.append({"t": adapter.t(state), "action": "NOOP", "planner": info})
            break
        state = adapter.transition(state, action)
        trace.append({
            "t": adapter.t(state),
            "action": adapter.action_text(action),
            "planner": info,
            "state_value": round(adapter.value(state, mode), 4),
        })
    success = adapter.is_success(state)
    return {
        "env": adapter.env_name,
        "seed": seed,
        "mode": mode,
        "simulations_per_decision": simulations,
        "success": bool(success),
        "steps": adapter.t(state),
        "terminal_failure": None if success else adapter.terminal_failure(state),
        "total_expanded_nodes": total_expanded,
        "total_rollout_steps": total_rollout_steps,
        "trace": trace,
        "final_metrics": adapter.final_metrics(state),
    }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["mode"], int(row["simulations_per_decision"]))].append(row)
    out = []
    for (env, mode, sims), cell in sorted(grouped.items()):
        n = len(cell)
        succ = sum(1 for r in cell if r["success"])
        failures = Counter(r["terminal_failure"] for r in cell if not r["success"])
        out.append({
            "env": env,
            "mode": mode,
            "simulations_per_decision": sims,
            "n": n,
            "successes": succ,
            "success_rate": succ / n if n else None,
            "mean_steps": mean([r["steps"] for r in cell]) if cell else None,
            "mean_expanded_nodes": mean([r["total_expanded_nodes"] for r in cell]) if cell else None,
            "mean_rollout_steps": mean([r["total_rollout_steps"] for r in cell]) if cell else None,
            "terminal_failures": dict(failures),
        })
    return out


def write_markdown(path: Path, summary: List[Dict[str, Any]]) -> None:
    lines = [
        "# MCTS-style Diagnostic Baselines",
        "",
        "These rows are non-LLM boundary checks. Visible mode uses visible proxy value; privileged mode uses terminal-objective information and is an upper bound.",
        "",
        "| Environment | Mode | Sims/decision | Success | Mean steps | Expanded nodes | Terminal failures |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        failures = ", ".join(f"{k}: {v}" for k, v in sorted(row["terminal_failures"].items())) or "-"
        lines.append(
            f"| {row['env']} | {row['mode']} | {row['simulations_per_decision']} | "
            f"{row['successes']}/{row['n']} ({pct(row['success_rate'])}) | "
            f"{row['mean_steps']:.2f} | {row['mean_expanded_nodes']:.1f} | {failures} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["alloy", "gb1_sequence"])
    ap.add_argument("--modes", nargs="+", default=["visible", "privileged"], choices=["visible", "privileged"])
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--simulations", default="256",
                    help="Comma-separated simulation budgets per decision.")
    ap.add_argument("--exploration", type=float, default=1.2)
    ap.add_argument("--rollout-policy", choices=["random", "topk", "best"], default="topk")
    ap.add_argument("--rollout-top-k", type=int, default=8)
    ap.add_argument("--output-root", default="results_v4/mcts_baselines")
    ap.add_argument("--output-suffix", default="mcts_diagnostic")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = parse_seed_spec(args.seeds)
    sim_budgets = [int(x) for x in args.simulations.split(",") if x.strip()]
    output_dir = Path(args.output_root) / f"{args.output_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    total = len(args.envs) * len(args.modes) * len(seeds) * len(sim_budgets)
    print("=== MCTS Diagnostic Baselines ===")
    print(f"Envs:        {args.envs}")
    print(f"Modes:       {args.modes}")
    print(f"Seeds:       {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Sim budgets: {sim_budgets}")
    print(f"Episodes:    {total}")
    print(f"Output:      {output_dir}")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    adapters = {env_name: make_adapter(env_name) for env_name in args.envs}
    done = 0
    for env_name in args.envs:
        adapter = adapters[env_name]
        for mode in args.modes:
            for sims in sim_budgets:
                for seed in seeds:
                    row = run_episode(
                        adapter=adapter,
                        seed=seed,
                        mode=mode,
                        simulations=sims,
                        exploration=args.exploration,
                        rollout_policy=args.rollout_policy,
                        rollout_top_k=args.rollout_top_k,
                    )
                    rows.append(row)
                    done += 1
                    print(
                        f"[{done}/{total}] {env_name}|{mode}|sims={sims}|seed={seed} -> "
                        f"{'PASS' if row['success'] else 'FAIL'}"
                    )

    summary = summarize(rows)
    result = {
        "config": {
            "envs": args.envs,
            "modes": args.modes,
            "seeds": args.seeds,
            "simulations": sim_budgets,
            "exploration": args.exploration,
            "rollout_policy": args.rollout_policy,
            "rollout_top_k": args.rollout_top_k,
            "note": "Non-LLM MCTS-style diagnostic boundary check; privileged mode is not an attainable LLM policy.",
        },
        "summary": summary,
        "episodes": rows,
    }
    json_path = output_dir / "mcts_baseline_summary.json"
    md_path = output_dir / "mcts_baseline_summary.md"
    json_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    write_markdown(md_path, summary)
    print(json.dumps({"output": str(json_path), "episodes": len(rows), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
