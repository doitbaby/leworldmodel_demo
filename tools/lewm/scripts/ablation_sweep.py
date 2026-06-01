"""Run the four-way LeWM ablation sweep for the FISAT rogue case study.

Each ablation isolates one architectural ingredient so the paper can
report a clean component-importance table:

* ``full``         -- baseline LeWM (SIGReg + one-hot actions).
* ``no_sigreg``    -- SIGReg disabled (``--sigreg-weight 0``).
* ``no_actions``   -- action-conditioning ablated via
                       ``--zero-actions`` (predictor sees no action
                       information).
* ``encoder_only`` -- both above disabled (encoder + predictor still
                       train on pixel reconstruction, but no SIGReg
                       and no action context).

The script does *not* invent its own training loop -- it shells out
to ``python -m tools.lewm.train`` with the appropriate flags, mirroring
exactly what a user would type on Kaggle. Use ``--dry-run`` to print
the planned commands without executing them.

After all runs finish, the script aggregates each run's
``metrics.json`` into a single ``summary.json`` + ``summary.csv`` so
the paper has a 1:1 row-per-ablation result table.

Usage::

    python -m tools.lewm.scripts.ablation_sweep \\
        --jsonl data/rogue_transitions_v3.jsonl \\
        --output-dir results/lewm/ablations/ \\
        --epochs 60 --batch-size 128 --device cuda

To preview without training::

    python -m tools.lewm.scripts.ablation_sweep \\
        --jsonl data/rogue_transitions_v3.jsonl \\
        --output-dir results/lewm/ablations/ \\
        --dry-run

Exit 0 on success.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ABLATIONS", "AblationSpec", "build_train_command", "main"]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    # CLI fragments appended to the shared train invocation. Use lists
    # of (flag, value) so we can serialise them into ``summary.json``.
    extra_args: tuple[str, ...] = field(default_factory=tuple)


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        name="full",
        description="Baseline LeWM (SIGReg ON, actions ON).",
    ),
    AblationSpec(
        name="no_sigreg",
        description="SIGReg weight set to 0; encoder + predictor + actions intact.",
        extra_args=("--sigreg-weight", "0"),
    ),
    AblationSpec(
        name="no_actions",
        description="Action conditioning zeroed via --zero-actions; SIGReg ON.",
        extra_args=("--zero-actions",),
    ),
    AblationSpec(
        name="encoder_only",
        description="Both SIGReg and actions disabled (pure encoder/predictor).",
        extra_args=("--sigreg-weight", "0", "--zero-actions"),
    ),
)


def build_train_command(
    *,
    spec: AblationSpec,
    jsonl: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    image_size: int,
    val_split: float,
    lr_schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
    sigreg_weight: float,
    device: str,
) -> list[str]:
    """Compose the full ``python -m tools.lewm.train`` argv for one ablation."""
    run_dir = output_dir / spec.name
    base = [
        sys.executable,
        "-m",
        "tools.lewm.train",
        "--observation-mode",
        "board-jsonl",
        "--jsonl-path",
        str(jsonl),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--image-size",
        str(image_size),
        "--val-split",
        str(val_split),
        "--lr-schedule",
        lr_schedule,
        "--warmup-steps",
        str(warmup_steps),
        "--min-lr-ratio",
        str(min_lr_ratio),
        "--sigreg-weight",
        str(sigreg_weight),
        "--device",
        device,
        "--output",
        str(run_dir / "checkpoint.pt"),
        "--best-output",
        str(run_dir / "best.pt"),
        "--metrics-csv",
        str(run_dir / "metrics.csv"),
    ]
    return base + list(spec.extra_args)


def _maybe_load_metrics(run_dir: Path) -> dict:
    """Load the per-epoch metrics.json that ``train.main`` writes alongside the CSV."""
    candidate = run_dir / "metrics.metrics.json"
    if candidate.exists():
        return json.loads(candidate.read_text())
    # Fallback: legacy path.
    legacy = run_dir / "metrics.json"
    if legacy.exists():
        return json.loads(legacy.read_text())
    return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the four-way LeWM ablation sweep.")
    p.add_argument("--jsonl", required=True, help="rogue.transition.v3 JSONL.")
    p.add_argument("--output-dir", required=True, help="Sweep output root.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--lr-schedule", default="cosine")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument(
        "--sigreg-weight",
        type=float,
        default=0.09,
        help="SIGReg weight used by the ``full`` and ``no_actions`` ablations. "
        "The ``no_sigreg`` and ``encoder_only`` rows always override this to 0.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--only",
        nargs="+",
        choices=[a.name for a in ABLATIONS],
        help="Run only the named ablations (default: all four).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl)
    if not jsonl.exists() and not args.dry_run:
        raise FileNotFoundError(f"jsonl not found: {jsonl}")

    selected = [a for a in ABLATIONS if (args.only is None or a.name in args.only)]
    summary: list[dict] = []
    for spec in selected:
        run_dir = output_dir / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(
            spec=spec,
            jsonl=jsonl,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            image_size=args.image_size,
            val_split=args.val_split,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            sigreg_weight=args.sigreg_weight,
            device=args.device,
        )
        printable = " ".join(cmd)
        print(f"\n=== ablation [{spec.name}] ===")
        print(f"desc: {spec.description}")
        print(f"cmd:  {printable}")
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ablation {spec.name!r} train exited with code {proc.returncode}")

        metrics = _maybe_load_metrics(run_dir)
        entry: dict = {
            "name": spec.name,
            "description": spec.description,
            "extra_args": list(spec.extra_args),
            "metrics_path": str(run_dir / "metrics.metrics.json"),
            "checkpoint_path": str(run_dir / "checkpoint.pt"),
            "best_checkpoint_path": str(run_dir / "best.pt"),
        }
        epochs_log = metrics.get("epochs", [])
        if epochs_log:
            last = epochs_log[-1]
            entry["final_epoch"] = int(last.get("epoch", -1))
            entry["final_train_loss"] = float(last.get("train_loss", float("nan")))
            entry["final_val_loss"] = float(last.get("val_loss", float("nan")))
            entry["final_train_pred"] = float(last.get("train_pred", float("nan")))
            entry["final_train_reward"] = float(last.get("train_reward", float("nan")))
            entry["final_train_done"] = float(last.get("train_done", float("nan")))
        summary.append(entry)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "ablations": [dataclasses.asdict(a) for a in selected],
                "results": summary,
            },
            indent=2,
        )
    )

    csv_path = output_dir / "summary.csv"
    if summary:
        keys: list[str] = []
        for row in summary:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in summary:
                writer.writerow({k: row.get(k, "") for k in keys})

    print(f"\nsummary json: {summary_path}")
    if summary:
        print(f"summary csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
