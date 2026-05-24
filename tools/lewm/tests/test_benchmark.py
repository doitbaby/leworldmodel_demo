"""End-to-end smoke for :mod:`tools.lewm.benchmark`.

Trains a tiny JEPA checkpoint, runs every mode for one episode each,
and asserts that the per-episode CSV + summary CSV are written with
the right columns and finite values for the non-JEPA modes.

Run with::

    python -m tools.lewm.tests.test_benchmark
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

from tools.lewm import benchmark, train


def _train_smoke_checkpoint(outdir: Path) -> Path:
    ckpt = outdir / "checkpoint.pt"
    train.main(
        [
            "--smoke",
            "--epochs",
            "1",
            "--items-per-epoch",
            "16",
            "--batch-size",
            "8",
            "--output",
            str(ckpt),
            "--best-output",
            str(outdir / "best.pt"),
            "--metrics-csv",
            str(outdir / "metrics.csv"),
        ]
    )
    assert ckpt.exists(), f"smoke train did not produce {ckpt}"
    return ckpt


_PER_EPISODE_HEADER = list(benchmark._PER_EPISODE_FIELDS)
_SUMMARY_HEADER = list(benchmark._SUMMARY_FIELDS)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


def test_per_episode_csv_has_expected_columns(tmpdir: Path) -> None:
    ckpt = _train_smoke_checkpoint(tmpdir)
    out_csv = tmpdir / "benchmark.csv"

    benchmark.main(
        [
            "--smoke",
            "--checkpoint",
            str(ckpt),
            "--output-csv",
            str(out_csv),
            # Cover all 5 modes in one shot.
            "--modes",
            "random",
            "mission",
            "mlp_lite",
            "lewm_no_planner",
            "lewm_dreamer",
        ]
    )

    assert out_csv.exists()
    assert out_csv.with_suffix(".summary.csv").exists()

    header, rows = _read_csv(out_csv)
    assert header == _PER_EPISODE_HEADER
    # 1 episode per mode, 5 modes -> 5 rows.
    assert len(rows) == 5, f"expected 5 rows, got {len(rows)}"
    modes_seen = {r["mode"] for r in rows}
    assert modes_seen == {
        "random",
        "mission",
        "mlp_lite",
        "lewm_no_planner",
        "lewm_dreamer",
    }, modes_seen

    for row in rows:
        assert int(row["episode_idx"]) == 0
        assert int(row["steps"]) > 0
        assert float(row["food_left"]) >= 0
        # All five modes produce a non-empty episode_return.
        float(row["episode_return"])
        assert row["died"] in {"0", "1"}


def test_dynamics_loss_finite_when_checkpoint_provided(tmpdir: Path) -> None:
    ckpt = _train_smoke_checkpoint(tmpdir)
    out_csv = tmpdir / "benchmark2.csv"

    benchmark.main(
        [
            "--smoke",
            "--checkpoint",
            str(ckpt),
            "--output-csv",
            str(out_csv),
            "--modes",
            "random",
            "mission",
            "lewm_dreamer",
        ]
    )

    _, rows = _read_csv(out_csv)
    by_mode = {r["mode"]: r for r in rows}

    # The JEPA mode always loads the model, so we ALWAYS measure dynamics
    # loss whenever a checkpoint is loaded -- including for the random /
    # mission modes that ran alongside it.
    for mode in ("random", "mission", "lewm_dreamer"):
        dyn = by_mode[mode]["dynamics_loss"]
        # Smoke runs short episodes; with food_start=100 and at most 30
        # steps, at least one step does not land on the exit, so we
        # expect a finite dynamics loss.
        assert dyn != "nan", f"expected finite dynamics loss for {mode}, got {dyn}"
        assert math.isfinite(float(dyn)), f"dynamics_loss not finite for {mode}: {dyn}"


def test_no_checkpoint_runs_non_jepa_modes(tmpdir: Path) -> None:
    out_csv = tmpdir / "benchmark3.csv"

    benchmark.main(
        [
            "--smoke",
            "--output-csv",
            str(out_csv),
            "--modes",
            "random",
            "mission",
        ]
    )

    header, rows = _read_csv(out_csv)
    assert header == _PER_EPISODE_HEADER
    assert {r["mode"] for r in rows} == {"random", "mission"}
    for r in rows:
        # No JEPA model loaded, so dynamics_loss should be nan for all rows.
        assert r["dynamics_loss"] == "nan", r
        assert r["reward_loss"] == "nan", r


def test_summary_csv_aggregates_correctly(tmpdir: Path) -> None:
    out_csv = tmpdir / "benchmark4.csv"

    benchmark.main(
        [
            "--output-csv",
            str(out_csv),
            "--episodes",
            "2",
            "--max-steps",
            "20",
            "--modes",
            "random",
            "mission",
        ]
    )

    summary_path = out_csv.with_suffix(".summary.csv")
    assert summary_path.exists()
    header, summary_rows = _read_csv(summary_path)
    assert header == _SUMMARY_HEADER
    assert len(summary_rows) == 2  # one row per mode
    for r in summary_rows:
        assert int(r["episodes"]) == 2
        assert float(r["mean_steps"]) > 0
        # std_episode_return is a real number even if all episodes match.
        float(r["std_episode_return"])


def test_jepa_mode_without_checkpoint_raises(tmpdir: Path) -> None:
    try:
        benchmark.main(
            [
                "--smoke",
                "--modes",
                "lewm_dreamer",
                "--output-csv",
                str(tmpdir / "benchmark5.csv"),
            ]
        )
    except ValueError as e:
        assert "checkpoint" in str(e).lower()
        return
    raise AssertionError("expected ValueError when JEPA mode requested without checkpoint")


def test_missing_checkpoint_raises(tmpdir: Path) -> None:
    try:
        benchmark.main(
            [
                "--smoke",
                "--checkpoint",
                str(tmpdir / "does_not_exist.pt"),
                "--modes",
                "lewm_dreamer",
                "--output-csv",
                str(tmpdir / "benchmark6.csv"),
            ]
        )
    except FileNotFoundError as e:
        assert "checkpoint" in str(e).lower()
        return
    raise AssertionError("expected FileNotFoundError for missing --checkpoint")


def main() -> int:
    test_fns = [
        test_per_episode_csv_has_expected_columns,
        test_dynamics_loss_finite_when_checkpoint_provided,
        test_no_checkpoint_runs_non_jepa_modes,
        test_summary_csv_aggregates_correctly,
        test_jepa_mode_without_checkpoint_raises,
        test_missing_checkpoint_raises,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for fn in test_fns:
            fn(tmpdir)
            print(f"OK: {fn.__name__}")
    print("benchmark tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
