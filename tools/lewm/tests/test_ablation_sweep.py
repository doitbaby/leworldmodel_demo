"""Smoke test for ``tools.lewm.scripts.ablation_sweep``.

Verifies:

* ``ABLATIONS`` contains the four expected entries with sane CLI args
  (``--zero-actions`` / ``--sigreg-weight 0`` where appropriate),
* ``build_train_command`` includes the per-ablation overrides on top of
  the shared train invocation,
* ``main(['--dry-run', ...])`` runs end-to-end without subprocessing
  any training and writes a valid ``summary.json``.

Run with::

    python -m tools.lewm.tests.test_ablation_sweep

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from tools.lewm.scripts.ablation_sweep import (
    ABLATIONS,
    AblationSpec,
    build_train_command,
    main,
)


def _test_specs() -> None:
    names = [a.name for a in ABLATIONS]
    assert names == ["full", "no_sigreg", "no_actions", "encoder_only"], names

    by_name = {a.name: a for a in ABLATIONS}
    assert by_name["no_sigreg"].extra_args == ("--sigreg-weight", "0")
    assert by_name["no_actions"].extra_args == ("--zero-actions",)
    assert by_name["encoder_only"].extra_args == ("--sigreg-weight", "0", "--zero-actions")
    assert by_name["full"].extra_args == ()


def _test_build_train_command() -> None:
    spec = AblationSpec(
        name="no_actions",
        description="...",
        extra_args=("--zero-actions",),
    )
    cmd = build_train_command(
        spec=spec,
        jsonl=Path("/tmp/foo.jsonl"),
        output_dir=Path("/tmp/sweep"),
        epochs=2,
        batch_size=8,
        sequence_length=3,
        image_size=32,
        val_split=0.1,
        lr_schedule="cosine",
        warmup_steps=200,
        min_lr_ratio=0.05,
        sigreg_weight=0.09,
        device="cpu",
    )
    # Sanity: train module is the entrypoint and we forward all key args.
    assert "tools.lewm.train" in cmd
    assert "--observation-mode" in cmd
    assert "--zero-actions" in cmd  # override appended
    # checkpoint paths go into the per-ablation directory.
    assert any(arg.endswith("/no_actions/checkpoint.pt") for arg in cmd), cmd
    assert any(arg.endswith("/no_actions/best.pt") for arg in cmd), cmd


def _test_dry_run_main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_dir = tmp / "sweep"
        # Touch the jsonl path so the non-dry-run existence check would also pass
        # if we ever switch this test off dry-run, but we keep --dry-run to avoid
        # spawning four real training subprocesses.
        jsonl_path = tmp / "rogue.jsonl"
        jsonl_path.write_text("")
        rc = main(
            [
                "--jsonl",
                str(jsonl_path),
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        assert rc == 0, rc
        summary_path = out_dir / "summary.json"
        assert summary_path.exists(), "summary.json missing"
        summary = json.loads(summary_path.read_text())
        names = [a["name"] for a in summary["ablations"]]
        assert names == ["full", "no_sigreg", "no_actions", "encoder_only"], names
        # Dry-run means we never trained, so the results array is empty.
        assert summary["results"] == [], summary["results"]


def main_test() -> int:
    _test_specs()
    _test_build_train_command()
    _test_dry_run_main()
    print("test_ablation_sweep: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main_test())
