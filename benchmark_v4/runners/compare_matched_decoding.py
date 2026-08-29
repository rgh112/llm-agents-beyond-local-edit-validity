#!/usr/bin/env python3
"""Compare matched deterministic vs sampled decoding runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = ["env", "model", "prompt", "memory"]


def pct(value):
    return "-" if value is None else f"{100 * float(value):.0f}%"


def load_summary(path: Path):
    with path.open() as f:
        data = json.load(f)
    rows = {}
    for row in data.get("results", []):
        key = tuple(str(row.get(k, "")) for k in KEYS)
        rows[key] = row
    return data, rows


def load_raw_successes(raw_dir: Path):
    rows = {}
    for path in raw_dir.glob("*.json"):
        with path.open() as f:
            data = json.load(f)
        key = (
            str(data.get("env_name", "")),
            str(data.get("model", "")),
            str(data.get("prompt_condition", "")),
            str(data.get("memory_condition", "")),
            int(data.get("seed")),
        )
        rows[key] = bool(data.get("success"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-summary", required=True)
    ap.add_argument("--sampled-summary", required=True)
    ap.add_argument("--det-raw", required=True)
    ap.add_argument("--sampled-raw", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    det_summary, det_rows = load_summary(Path(args.det_summary))
    sampled_summary, sampled_rows = load_summary(Path(args.sampled_summary))
    det_raw = load_raw_successes(Path(args.det_raw))
    sampled_raw = load_raw_successes(Path(args.sampled_raw))

    det_temp = det_summary.get("config", {}).get("temperature")
    det_top_p = det_summary.get("config", {}).get("top_p")
    sampled_temp = sampled_summary.get("config", {}).get("temperature")
    sampled_top_p = sampled_summary.get("config", {}).get("top_p")

    headers = [
        "env",
        "model",
        "prompt",
        "memory",
        "det SR",
        "sampled SR",
        "delta",
        "both success",
        "lost",
        "gained",
        "both fail",
        "det LV",
        "sampled LV",
    ]
    lines = [
        "# Matched Decoding Comparison",
        "",
        f"Deterministic config: temperature={det_temp}, top_p={det_top_p}",
        f"Sampled config: temperature={sampled_temp}, top_p={sampled_top_p}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for key in sorted(set(det_rows) | set(sampled_rows)):
        det = det_rows.get(key, {})
        sampled = sampled_rows.get(key, {})
        srd = det.get("SR")
        srs = sampled.get("SR")
        delta = None if srd is None or srs is None else float(srs) - float(srd)
        seeds = sorted(
            seed_key[-1]
            for seed_key in set(det_raw) | set(sampled_raw)
            if seed_key[:-1] == key
        )
        both_success = lost = gained = both_fail = 0
        for seed in seeds:
            seed_key = (*key, seed)
            a = det_raw.get(seed_key)
            b = sampled_raw.get(seed_key)
            if a is True and b is True:
                both_success += 1
            elif a is True and b is False:
                lost += 1
            elif a is False and b is True:
                gained += 1
            elif a is False and b is False:
                both_fail += 1
        lines.append("| " + " | ".join([
            *key,
            pct(srd),
            pct(srs),
            pct(delta),
            str(both_success),
            str(lost),
            str(gained),
            str(both_fail),
            pct(det.get("local_valid_edit_rate")),
            pct(sampled.get("local_valid_edit_rate")),
        ]) + " |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
