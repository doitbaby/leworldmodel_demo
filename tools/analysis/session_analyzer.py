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
    malformed_lines = 0

    for path in sorted(input_path.glob("session_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    key = (
                        str(row.get("session", path.stem)),
                        int(row.get("episode", 0)),
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    malformed_lines += 1
                    print(f"Warning: skipped malformed line {path}:{line_number}: {exc}")
                    continue

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
        died = parse_bool(end_row.get("died", False))

        rows.append(
            {
                "session": key[0],
                "episode": key[1],
                "levels_cleared": int(end_row.get("levels_cleared", 0)),
                "died": died,
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

    print(f"Output CSV: {output_path}")
    print(f"Wrote {len(rows)} episode rows to {output_path}")
    print(f"Episodes analyzed: {len(rows)}")
    if malformed_lines:
        print(f"Malformed JSONL lines skipped: {malformed_lines}")

    row_count = len(rows)
    avg_compliance = (
        sum(row["compliance_rate"] for row in rows) / row_count if row_count else 0.0
    )
    avg_levels = (
        sum(row["levels_cleared"] for row in rows) / row_count if row_count else 0.0
    )
    avg_food = (
        sum(row["food_remaining"] for row in rows) / row_count if row_count else 0.0
    )
    death_rate = sum(1 for row in rows if row["died"]) / row_count if row_count else 0.0
    high_compliance = [row for row in rows if row["compliance_rate"] >= 0.75]
    low_compliance = [row for row in rows if row["compliance_rate"] < 0.75]

    print(f"Avg food remaining: {avg_food:.2f}")
    print(f"Death rate: {death_rate:.1%}")
    print(f"Avg compliance: {avg_compliance:.1%}")
    print(f"Avg levels cleared: {avg_levels:.2f}")
    print(format_compliance_split("High compliance >= 0.75", high_compliance))
    print(format_compliance_split("Low compliance < 0.75", low_compliance))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def format_compliance_split(label: str, rows: list[dict]) -> str:
    if not rows:
        return f"{label}: 0 episodes"

    avg_food = sum(row["food_remaining"] for row in rows) / len(rows)
    death_rate = sum(1 for row in rows if row["died"]) / len(rows)
    avg_levels = sum(row["levels_cleared"] for row in rows) / len(rows)
    return (
        f"{label}: {len(rows)} episodes | "
        f"avg food {avg_food:.2f} | "
        f"death rate {death_rate:.1%} | "
        f"avg levels {avg_levels:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coach session JSONL files.")
    parser.add_argument(
        "--input-dir",
        default="coach_sessions",
        help="Folder with session_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        default="results/coach_analysis.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    analyze_sessions(args.input_dir, args.output)


if __name__ == "__main__":
    main()
