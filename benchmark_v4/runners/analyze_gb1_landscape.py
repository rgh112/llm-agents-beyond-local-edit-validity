#!/usr/bin/env python3
"""Analyze additive-proxy vs epistatic GB1 landscape structure."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from benchmark_v4.envs.gb1_sequence import GB1SequenceEnv


def spearman(x, y):
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitness-target", type=float, default=5.0)
    ap.add_argument("--output", default="results_v4/gb1_landscape_stats.json")
    args = ap.parse_args()

    env = GB1SequenceEnv(fitness_target=args.fitness_target, start_mutations=0)
    allowed = [env._allowed[i] for i in range(4)]

    rows = []
    for seq_tuple in itertools.product(*allowed):
        seq = list(seq_tuple)
        variant = "".join(seq)
        true_fit = env._true_fitness(seq)
        add_fit = env._estimated_fitness(seq)
        rows.append({
            "variant": variant,
            "true_fitness": true_fit,
            "additive_fitness": add_fit,
            "success": true_fit >= args.fitness_target,
        })

    true_vals = np.array([r["true_fitness"] for r in rows], dtype=float)
    add_vals = np.array([r["additive_fitness"] for r in rows], dtype=float)
    pearson = float(np.corrcoef(true_vals, add_vals)[0, 1])
    rho = spearman(true_vals, add_vals)

    sorted_true = sorted(rows, key=lambda r: r["true_fitness"], reverse=True)
    sorted_add = sorted(rows, key=lambda r: r["additive_fitness"], reverse=True)
    topk = {}
    for k in [5, 10, 25, 50]:
        t = {r["variant"] for r in sorted_true[:k]}
        a = {r["variant"] for r in sorted_add[:k]}
        topk[str(k)] = len(t & a) / k

    local_optima = 0
    for row in rows:
        seq = list(row["variant"])
        fit = row["true_fitness"]
        better_neighbor = False
        for pos in range(4):
            for aa in env._allowed[pos]:
                if aa == seq[pos]:
                    continue
                neighbor = list(seq)
                neighbor[pos] = aa
                if env._true_fitness(neighbor) > fit:
                    better_neighbor = True
                    break
            if better_neighbor:
                break
        if not better_neighbor:
            local_optima += 1

    success_rows = [r for r in rows if r["success"]]
    requires_pos2_mutation = sum(1 for r in success_rows if r["variant"][2] != env._wildtype[2])

    out = {
        "fitness_target": args.fitness_target,
        "n_variants": len(rows),
        "n_success_variants": len(success_rows),
        "success_fraction": len(success_rows) / len(rows),
        "pearson_additive_true": pearson,
        "spearman_additive_true": rho,
        "topk_overlap": topk,
        "n_local_optima": local_optima,
        "local_optima_fraction": local_optima / len(rows),
        "success_variants_requiring_pos2_mutation": requires_pos2_mutation,
        "success_variants_requiring_pos2_mutation_fraction": (
            requires_pos2_mutation / len(success_rows) if success_rows else None
        ),
        "top_true": sorted_true[:20],
        "top_additive": sorted_add[:20],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
