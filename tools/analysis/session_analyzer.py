#!/usr/bin/env python3
"""
Analyze coach session JSONL files and export per-episode CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def analyze_sessions(input_dir: str, output: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_steps: dict[tuple[str, int], list[dict]] = defaultdict(list)
    episode_end_rows: dict[tuple[str, int], dict] = {}

    for path in sorted(input_path.glob("session_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)
                key = (str(row.get("session", path.stem)), int(row.get("episode", 0)))
                if row.get("type") == "episode_end":
                    episode_end_rows[key] = row
                else:
                    episode_steps[key].append(row)

    fieldnames = [
        "session",
        "episode",
        "levels_cleared",
        "died",
        "food_remaining",
        "total_steps",
        "followed_steps",
        "compliance_rate",
    ]

    rows: list[dict] = []
    for key in sorted(set(episode_steps) | set(episode_end_rows)):
        steps = episode_steps.get(key, [])
        end_row = episode_end_rows.get(key, {})
        total_steps = len(steps)
        followed_steps = sum(1 for row in steps if row.get("followed"))
        compliance = (followed_steps / total_steps) if total_steps else 0.0

        rows.append(
            {
                "session": key[0],
                "episode": key[1],
                "levels_cleared": int(end_row.get("levels_cleared", 0)),
                "died": bool(end_row.get("died", False)),
                "food_remaining": int(end_row.get("food_remaining", 0)),
                "total_steps": total_steps,
                "followed_steps": followed_steps,
                "compliance_rate": round(compliance, 4),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} episode rows to {output_path}")
    if not rows:
        return

    avg_compliance = sum(row["compliance_rate"] for row in rows) / len(rows)
    avg_levels = sum(row["levels_cleared"] for row in rows) / len(rows)
    print(f"Avg compliance: {avg_compliance:.1%}")
    print(f"Avg levels cleared: {avg_levels:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coach session JSONL files.")
    parser.add_argument("--input-dir", default="coach_sessions", help="Folder with session_*.jsonl files")
    parser.add_argument("--output", default="results/coach_analysis.csv", help="Output CSV path")
    args = parser.parse_args()
    analyze_sessions(args.input_dir, args.output)


if __name__ == "__main__":
    main()
