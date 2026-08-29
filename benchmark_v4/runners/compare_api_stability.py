#!/usr/bin/env python3
"""Compare two date-separated API stability runs."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


KEYS = ["env", "model", "prompt", "memory", "controller", "scorer", "sensitivity_setting"]


def load_rows(pattern: str):
    rows = {}
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
        for row in data.get("results", []):
            key = tuple(str(row.get(k, "")) for k in KEYS)
            rows[key] = row
    return rows


def pct(x):
    return "-" if x is None else f"{100 * float(x):.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-a", required=True, help="Glob for first *_summary.json set.")
    ap.add_argument("--pass-b", required=True, help="Glob for second *_summary.json set.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    a = load_rows(args.pass_a)
    b = load_rows(args.pass_b)
    all_keys = sorted(set(a) | set(b))

    headers = ["env", "model", "prompt", "memory", "controller", "scorer", "setting",
               "N A", "SR A", "N B", "SR B", "Delta"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key in all_keys:
        ra = a.get(key, {})
        rb = b.get(key, {})
        sra = ra.get("SR")
        srb = rb.get("SR")
        delta = None if sra is None or srb is None else float(srb) - float(sra)
        lines.append("| " + " | ".join([
            *key,
            str(ra.get("n_total", "")),
            pct(sra),
            str(rb.get("n_total", "")),
            pct(srb),
            pct(delta),
        ]) + " |")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
