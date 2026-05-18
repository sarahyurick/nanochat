#!/bin/bash
# Pretrain on pre-tokenized Megatron .bin/.idx data (Llama2 tokenizer).
#
# Designed for 8xB200 192GB. Goal: best pretrained model you can get with the
# given input data, no time constraint. Adjust DEPTH up/down for different sizes.
#
# Override knobs (env vars):
#   WANDB_RUN          wandb run name; "dummy" disables wandb (default)
#   RUN_TAG            short tag used for the model checkpoint dir (default: $DEPTH-$WEIGHTS_NAME)
#   DATA_DIR           directory with .bin/.idx pairs
#                      (default: /path/to/data)
#   TOKENIZER_MODEL    Llama2 .model file
#                      (default: /path/to/tokenizer.model)
#   DOMAIN_WEIGHTS     'proportional' (default), 'uniform', or a path to a JSON weights file
#   DEPTH              transformer depth (default: 30, "best model" sweet spot for 8xB200)
#   PARAM_DATA_RATIO   tokens:params ratio (default: 20, Chinchilla compute-optimal)
#   DEVICE_BATCH_SIZE  per-rank micro-batch (default: 32; raise if VRAM allows)
#   PILE_VAL           1 to additionally prep + eval on Pile val (default: 1)
#   NANOCHAT_BASE_DIR  cache root (default: /raid/$USER/nanochat/megatron_pretrain_cache)
#
# Example (Run A — proportional baseline):
#   WANDB_RUN=d30_proportional bash runs/megatron_pretrain.sh
# Example (Run B — custom weights):
#   WANDB_RUN=d30_weighted DOMAIN_WEIGHTS=/path/to/weights.json bash runs/megatron_pretrain.sh
#
# Notes:
#   - FP8 is enabled (B200 supports it). Will also work on H100; not on A100.
#   - FA3 is Hopper-only; on B200 we use SDPA fallback which requires window_pattern=L.

set -euo pipefail

# ---------------------------------------------------------------------------
# User-tunable knobs (env-var overrides above take precedence)
DATA_DIR="${DATA_DIR:-/path/to/data}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/path/to/tokenizer.model}"
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-proportional}"
# DOMAIN_WEIGHTS=/path/to/weights.json  # TODO
#24, ~430M, ~9B, ~3 h
#26, ~545M, ~11B, ~4–6 h
DEPTH="${DEPTH:-30}"  # ~830M params, ~17B tokens (ratio 20), estimated single-run wall clock: ~8-12 h
# DEPTH=36  # ~1.4B params, ~28B tokens (ratio 20), estimated single-run wall clock: ~16-24 h
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
PILE_VAL="${PILE_VAL:-1}"
WANDB_RUN="${WANDB_RUN:-dummy}"

# Short tag used in checkpoint dir name. Distinguishes runs differing only in weights.
WEIGHTS_NAME=$(basename "$DOMAIN_WEIGHTS" | sed 's/\.json$//')
RUN_TAG="${RUN_TAG:-d${DEPTH}_${WEIGHTS_NAME}}"

# Caches
export UV_CACHE_DIR="${UV_CACHE_DIR:-/raid/$USER/uv_cache}"
export HF_HOME="${HF_HOME:-/raid/$USER/hf_cache}"
export HF_CACHE="${HF_CACHE:-/raid/$USER/hf_cache}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/raid/$USER/nanochat/megatron_pretrain_cache}"
# Per-run report dir so multiple runs (e.g. proportional vs weighted) don't clobber each other.
export NANOCHAT_REPORT_DIR="${NANOCHAT_REPORT_DIR:-$NANOCHAT_BASE_DIR/reports/$RUN_TAG}"
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0
mkdir -p "$NANOCHAT_BASE_DIR" "$NANOCHAT_REPORT_DIR"

echo "================================================================"
echo "  Megatron pretrain"
echo "================================================================"
echo "  Data dir         : $DATA_DIR"
echo "  Tokenizer model  : $TOKENIZER_MODEL"
echo "  Domain weights   : $DOMAIN_WEIGHTS"
echo "  Depth            : $DEPTH"
echo "  Tokens:Params    : $PARAM_DATA_RATIO"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Pile val         : $PILE_VAL"
echo "  Run tag          : $RUN_TAG"
echo "  wandb run        : $WANDB_RUN"
echo "  Base dir         : $NANOCHAT_BASE_DIR"
echo "  Report dir       : $NANOCHAT_REPORT_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# venv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
# sentencepiece is needed for the Llama2 tokenizer wrapper; install if missing.
python -c "import sentencepiece" 2>/dev/null || uv pip install sentencepiece

# ---------------------------------------------------------------------------
# Report header
python -m nanochat.report reset

# ---------------------------------------------------------------------------
# Build tokenizer artifacts (idempotent: skip if already built)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer_kind.txt" ]; then
    echo ">> Building Llama2 tokenizer artifacts ..."
    python -m scripts.build_llama_tokenizer --tokenizer-model "$TOKENIZER_MODEL"
else
    echo ">> Tokenizer artifacts already present, skipping build."
fi

# ---------------------------------------------------------------------------
# (Optional) prepare Pile val
PILE_VAL_FLAG=""
if [ "$PILE_VAL" = "1" ]; then
    PILE_VAL_DIR="$NANOCHAT_BASE_DIR/pile_val"
    if [ ! -f "$PILE_VAL_DIR/pile_val.idx" ]; then
        echo ">> Preparing Pile val (EleutherAI/pile_val_test, full validation split) ..."
        python -m scripts.prepare_pile_val --tokenizer-model "$TOKENIZER_MODEL" \
            --out-dir "$PILE_VAL_DIR"
    else
        echo ">> Pile val already prepared, skipping."
    fi
    PILE_VAL_FLAG="--pile-val-dir=$PILE_VAL_DIR"
fi

# ---------------------------------------------------------------------------
# Pretraining
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --data-source=megatron \
    --data-dir="$DATA_DIR" \
    --domain-weights="$DOMAIN_WEIGHTS" \
    $PILE_VAL_FLAG \
    --depth="$DEPTH" \
    --target-param-data-ratio="$PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --window-pattern=L \
    --fp8 \
    --sample-every=-1 \
    --core-metric-every=2000 \
    --save-every=2000 \
    --model-tag="$RUN_TAG" \
    --run="$WANDB_RUN"

# ---------------------------------------------------------------------------
# Base eval (CORE + val bpb + samples). Uses the same Megatron data source so
# train/val/pile_val bpb numbers line up with what was logged during training.
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$RUN_TAG" \
    --data-source=megatron \
    --data-dir="$DATA_DIR" \
    --domain-weights="$DOMAIN_WEIGHTS" \
    $PILE_VAL_FLAG

# ---------------------------------------------------------------------------
# Final report
python -m nanochat.report generate
