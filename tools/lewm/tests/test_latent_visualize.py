"""Smoke test for ``tools.lewm.scripts.latent_visualize``.

Trains a tiny CPU JEPA checkpoint, runs the latent-visualize script
against it (with ``--no-plots`` so matplotlib stays optional in CI),
and verifies that the resulting summary + .npz files have the schema
downstream notebooks rely on.

Run with::

    python -m tools.lewm.tests.test_latent_visualize

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from tools.lewm.scripts.generate_jsonl_v3 import generate
from tools.lewm.scripts.latent_visualize import main as latent_main
from tools.lewm.scripts.latent_visualize import pca_2d
from tools.lewm.train import main as train_main


def _train_smoke_checkpoint(jsonl_path: Path, checkpoint_path: Path) -> None:
    """Run a 2-epoch board-jsonl training to produce a real checkpoint."""
    argv = [
        "--observation-mode",
        "board-jsonl",
        "--jsonl-path",
        str(jsonl_path),
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--sequence-length",
        "3",
        "--embed-dim",
        "32",
        "--predictor-depth",
        "1",
        "--predictor-heads",
        "2",
        "--predictor-mlp-dim",
        "64",
        "--sigreg-num-proj",
        "16",
        "--device",
        "cpu",
        "--output",
        str(checkpoint_path),
        "--best-output",
        str(checkpoint_path.with_name(checkpoint_path.stem + "_best.pt")),
        "--metrics-csv",
        str(checkpoint_path.with_suffix(".csv")),
    ]
    rc = train_main(argv)
    assert rc == 0, f"train_main exited with {rc}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jsonl_path = tmp / "rogue_smoke.jsonl"
        checkpoint = tmp / "lewm_smoke.pt"
        out_dir = tmp / "viz"

        generate(
            output=jsonl_path,
            num_transitions=300,
            mission_ratio=0.7,
            epsilon=0.1,
            seed=0,
            board_size=8,
            image_size=32,
            food_start=100,
            food_per_level=50,
            max_steps=200,
            include_obs_stub=True,
        )
        _train_smoke_checkpoint(jsonl_path, checkpoint)

        rc = latent_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--jsonl",
                str(jsonl_path),
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--max-samples",
                "80",
                "--batch-size",
                "16",
                "--no-plots",
            ]
        )
        assert rc == 0, f"latent_main exited with {rc}"

        summary_path = out_dir / "summary.json"
        assert summary_path.exists(), "summary.json missing"
        summary = json.loads(summary_path.read_text())
        assert summary["num_samples"] == 80, summary
        assert summary["embed_dim"] == 32, summary
        evr = summary["pca_explained_variance_ratio"]
        assert len(evr) == 2 and 0.0 <= evr[0] <= 1.0 and 0.0 <= evr[1] <= 1.0, evr
        # No plots requested -> plot list empty.
        assert summary["plots"] == [], summary

        npz_path = out_dir / "embeddings.npz"
        assert npz_path.exists()
        data = np.load(npz_path)
        assert data["embeddings"].shape == (80, 32), data["embeddings"].shape
        for key in ("level", "food", "near_exit", "near_enemy", "action"):
            assert data[key].shape == (80,), (key, data[key].shape)

    # Standalone PCA sanity check (no I/O).
    rng = np.random.default_rng(0)
    cloud = rng.standard_normal((50, 6)).astype(np.float32)
    proj, ratio = pca_2d(cloud)
    assert proj.shape == (50, 2)
    assert ratio.shape == (2,)
    assert 0.0 <= float(ratio[0]) <= 1.0
    assert float(ratio[0]) >= float(ratio[1])  # PC1 always >= PC2 in explained variance

    print("test_latent_visualize: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
