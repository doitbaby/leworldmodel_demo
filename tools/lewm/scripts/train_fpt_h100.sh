#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FPT AI Factory -- one-shot LeWorldModel training + benchmark pipeline.
#
# Wraps M4 (`tools.lewm.train`) + M6 (`tools.lewm.benchmark`) so that a fresh
# AI Notebook / GPU Container / GPU VM can go from "git clone" to
# "downloadable best.pt + benchmark.csv" with a single command.
#
# This script is the implementation of option (b) in
# docs/research/lewm-fpt-aifactory-plan.md (sent to the user 2026-05-18).
#
# Usage:
#   bash tools/lewm/scripts/train_fpt_h100.sh <jsonl_path>
#
# Required positional argument:
#   <jsonl_path>   Path to a `rogue.transition.v3` JSONL recorded by Unity's
#                  `RogueTransitionRecorder`. Pass `--synthetic` to skip the
#                  real-data train and only exercise the synthetic-data smoke
#                  (useful while waiting on Unity gameplay).
#
# Tunable env vars (all optional):
#   EPOCHS=30                    # full-run epoch count
#   BATCH_SIZE=128               # train batch size
#   VAL_SPLIT=0.1                # episode-level val fraction
#   WARMUP_STEPS=200             # cosine warmup steps
#   MIN_LR_RATIO=0.05            # cosine min lr (x base_lr)
#   BENCHMARK_EPISODES=20        # eval episodes / mode
#   BENCHMARK_MAX_STEPS=200      # eval step cap
#   SEED=0                       # train + benchmark seed
#   DEVICE=cuda                  # torch device override (set cpu to dry-run)
#   OUTPUT_DIR=results/lewm      # where checkpoints + CSVs land
#   SKIP_SMOKE=                  # set to any non-empty value to skip the
#                                # 2-epoch synthetic smoke that runs first
#   SKIP_BENCHMARK=              # set to any non-empty value to skip M6
#   ARCHIVE=                     # set to any non-empty value to also tar up
#                                # OUTPUT_DIR into lewm-artifacts-<ts>.tar.gz
#
# Exit codes:
#   0  pipeline finished, best.pt + metrics.csv + benchmark.{csv,summary.csv}
#      written under $OUTPUT_DIR.
#   1  prerequisite check failed (missing repo, missing python, no GPU when
#      DEVICE=cuda, missing JSONL path, etc.).
#   2  a sub-command (train / benchmark / smoke) returned non-zero.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------- pretty logging ------------------------------------------------
_log()  { printf '\033[1;36m[fpt-train %(%H:%M:%S)T]\033[0m %s\n' -1 "$*"; }
_warn() { printf '\033[1;33m[fpt-train %(%H:%M:%S)T] WARN:\033[0m %s\n' -1 "$*" >&2; }
_err()  { printf '\033[1;31m[fpt-train %(%H:%M:%S)T] ERROR:\033[0m %s\n' -1 "$*" >&2; }

# ---------- positional arg ------------------------------------------------
if [[ $# -lt 1 ]]; then
    _err "missing required argument: <jsonl_path> (or --synthetic)"
    sed -n '4,30p' "$0" >&2
    exit 1
fi
JSONL_PATH="$1"

# ---------- defaults ------------------------------------------------------
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-20}"
BENCHMARK_MAX_STEPS="${BENCHMARK_MAX_STEPS:-200}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lewm}"
SKIP_SMOKE="${SKIP_SMOKE:-}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-}"
ARCHIVE="${ARCHIVE:-}"

# ---------- locate repo root ----------------------------------------------
# Script lives at tools/lewm/scripts/train_fpt_h100.sh -> repo root is 3 dirs up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -f "$REPO_ROOT/tools/lewm/train.py" ]]; then
    _err "repo layout unexpected: $REPO_ROOT/tools/lewm/train.py missing"
    _err "this script must live at tools/lewm/scripts/train_fpt_h100.sh"
    exit 1
fi

cd "$REPO_ROOT"
_log "repo root: $REPO_ROOT"
_log "args: jsonl=$JSONL_PATH epochs=$EPOCHS batch=$BATCH_SIZE device=$DEVICE seed=$SEED"
_log "output_dir: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ---------- python + deps -------------------------------------------------
if ! command -v python >/dev/null 2>&1; then
    _err "python not on PATH"
    exit 1
fi
_log "python: $(python --version 2>&1)"
_log "installing tools/lewm/requirements.txt (idempotent)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r tools/lewm/requirements.txt

# ---------- GPU sanity ----------------------------------------------------
# If DEVICE=cuda was requested, fail loudly when CUDA isn't actually wired
# up -- silently falling back to CPU on an FPT notebook would burn budget
# at GPU prices while running 50x slower.
if [[ "$DEVICE" == "cuda" ]]; then
    GPU_NAME="$(
        python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() == False", file=sys.stderr)
    sys.exit(1)
print(torch.cuda.get_device_name(0))
PY
    )" || {
        _err "DEVICE=cuda but CUDA not available."
        _err "On FPT AI Notebook this usually means pip installed the CPU-only"
        _err "torch wheel. Reinstall with the cu124 index:"
        _err "  python -m pip install --upgrade torch torchvision \\"
        _err "      --index-url https://download.pytorch.org/whl/cu124"
        _err "Alternatively, re-run this script with DEVICE=cpu for a dry run."
        exit 1
    }
    _log "gpu: $GPU_NAME"
else
    _warn "DEVICE=$DEVICE (not cuda) -- training will be slow but allowed for dry runs"
fi

# ---------- step 1: synthetic smoke (~30 s on H100) -----------------------
# Cheap sanity check before touching real data. Catches the "torch is
# installed but cuda kernels are broken" case before we burn a 30-min train.
if [[ -z "$SKIP_SMOKE" ]]; then
    _log "step 1/3: synthetic smoke training (--smoke)"
    if ! python -m tools.lewm.train --smoke --device "$DEVICE"; then
        _err "synthetic smoke failed -- aborting before real training"
        exit 2
    fi
else
    _log "step 1/3: SKIPPED (SKIP_SMOKE set)"
fi

# ---------- step 2: real training -----------------------------------------
if [[ "$JSONL_PATH" == "--synthetic" ]]; then
    _log "step 2/3: SKIPPED (--synthetic positional; smoke above is the only train)"
    BEST_CKPT="$OUTPUT_DIR/checkpoint.pt"
else
    if [[ ! -f "$JSONL_PATH" ]]; then
        _err "JSONL file not found: $JSONL_PATH"
        exit 1
    fi
    JSONL_BYTES="$(wc -c <"$JSONL_PATH")"
    JSONL_LINES="$(wc -l <"$JSONL_PATH")"
    _log "step 2/3: real training on $JSONL_PATH (${JSONL_BYTES} bytes, ${JSONL_LINES} transitions)"

    METRICS_CSV="$OUTPUT_DIR/metrics.csv"
    FINAL_CKPT="$OUTPUT_DIR/checkpoint.pt"
    BEST_CKPT="$OUTPUT_DIR/best.pt"

    if ! python -m tools.lewm.train \
            --device "$DEVICE" \
            --observation-mode board-jsonl \
            --jsonl-path "$JSONL_PATH" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --val-split "$VAL_SPLIT" \
            --lr-schedule cosine \
            --warmup-steps "$WARMUP_STEPS" \
            --min-lr-ratio "$MIN_LR_RATIO" \
            --metrics-csv "$METRICS_CSV" \
            --output "$FINAL_CKPT" \
            --best-output "$BEST_CKPT"; then
        _err "training run failed -- check the per-epoch log above"
        exit 2
    fi

    if [[ ! -f "$BEST_CKPT" ]]; then
        # No val_loss improvement at all -- fall back to the final-epoch
        # checkpoint so the benchmark step still has something to load.
        _warn "best.pt not written (val_loss never improved); using checkpoint.pt"
        BEST_CKPT="$FINAL_CKPT"
    fi
fi

# ---------- step 3: 5-mode benchmark --------------------------------------
if [[ -n "$SKIP_BENCHMARK" ]]; then
    _log "step 3/3: SKIPPED (SKIP_BENCHMARK set)"
else
    _log "step 3/3: benchmark 5 modes ($BENCHMARK_EPISODES eps, max_steps=$BENCHMARK_MAX_STEPS)"
    BENCH_CSV="$OUTPUT_DIR/benchmark.csv"

    if ! python -m tools.lewm.benchmark \
            --episodes "$BENCHMARK_EPISODES" \
            --max-steps "$BENCHMARK_MAX_STEPS" \
            --seed "$SEED" \
            --checkpoint "$BEST_CKPT" \
            --modes random mission mlp_lite lewm_no_planner lewm_dreamer \
            --output-csv "$BENCH_CSV"; then
        _err "benchmark run failed -- the checkpoint at $BEST_CKPT may be malformed"
        exit 2
    fi

    SUMMARY_CSV="${BENCH_CSV%.csv}.summary.csv"
    if [[ -f "$SUMMARY_CSV" ]]; then
        _log "summary csv: $SUMMARY_CSV"
        # Echo the summary inline so the user sees mean_levels_cleared etc.
        # immediately even if they only scroll the tail of the log.
        printf '\n---- benchmark summary ----\n'
        cat "$SUMMARY_CSV"
        printf '---- end summary ----\n\n'
    fi
fi

# ---------- step 4 (optional): archive artifacts --------------------------
if [[ -n "$ARCHIVE" ]]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    ARCHIVE_PATH="lewm-artifacts-${TS}.tar.gz"
    _log "archiving $OUTPUT_DIR -> $ARCHIVE_PATH"
    tar -czf "$ARCHIVE_PATH" "$OUTPUT_DIR"
    _log "wrote: $ARCHIVE_PATH ($(wc -c <"$ARCHIVE_PATH") bytes)"
fi

_log "done. download $OUTPUT_DIR/best.pt back to your Unity machine and run:"
_log "  python -m tools.lewm.serve --checkpoint $OUTPUT_DIR/best.pt --host 127.0.0.1 --port 5555"
