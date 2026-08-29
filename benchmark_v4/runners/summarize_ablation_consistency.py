#!/usr/bin/env python3
"""Summarize directional consistency of ablation effects."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_pct(text):
    text = text.strip()
    if text == "-":
        return None
    m = re.match(r"(-?\d+(?:\.\d+)?)%", text)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("effect_table", help="Markdown table from analyze_ablation_effects.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Treat absolute deltas <= epsilon as neutral.")
    args = ap.parse_args()

    groups = defaultdict(lambda: {"pos": 0, "neg": 0, "zero": 0, "n": 0})
    with open(args.effect_table) as f:
        for line in f:
            if not line.startswith("| ") or line.startswith("| ---") or "ablation" in line:
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 14:
                continue
            ablation = parts[0]
            delta = parse_pct(parts[13])
            if delta is None:
                continue
            family = ablation.split(":", 1)[0]
            key = ablation
            for bucket in {key, family}:
                groups[bucket]["n"] += 1
                if delta > args.epsilon:
                    groups[bucket]["pos"] += 1
                elif delta < -args.epsilon:
                    groups[bucket]["neg"] += 1
                else:
                    groups[bucket]["zero"] += 1

    headers = ["effect", "N", "positive", "negative", "neutral", "pos fraction", "neg fraction"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key, val in sorted(groups.items()):
        n = val["n"]
        lines.append("| " + " | ".join([
            key,
            str(n),
            str(val["pos"]),
            str(val["neg"]),
            str(val["zero"]),
            f"{val['pos'] / n:.2f}" if n else "-",
            f"{val['neg'] / n:.2f}" if n else "-",
        ]) + " |")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
