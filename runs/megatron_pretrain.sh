#!/bin/bash
# Pretrain on pre-tokenized Megatron .bin/.idx data (Llama2 tokenizer).
#
# Override knobs (env vars):
#   WANDB_RUN          wandb run name; "dummy" disables wandb (default)
#   RUN_TAG            short tag used for the model checkpoint dir (default: $DEPTH-$WEIGHTS_NAME)
#   DATA_DIR           directory with .bin/.idx pairs
#   TOKENIZER_MODEL    Llama2 .model file
#   DOMAIN_WEIGHTS     'proportional' (default), 'uniform', or a path to a JSON weights file
#   DEPTH              transformer depth (default: 30, "best model" sweet spot for 8xB200)
#   PARAM_DATA_RATIO   tokens:params ratio (default: 20, Chinchilla compute-optimal)
#   DEVICE_BATCH_SIZE  per-rank micro-batch (default: 32; raise if VRAM allows)
#   PILE_VAL           1 to additionally prep + eval on Pile val (default: 1)
#   PILE_VAL_DIR       shared pre-tokenized Pile val cache (default: $NANOCHAT_BASE_DIR/pile_val)
#   NANOCHAT_BASE_DIR  cache root
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
DATA_DIR="${DATA_DIR:-/datasets/syurick/smollm_climb_data_curation/domains}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/datasets/syurick/nemotron_climb_data_curation/tokenizer.model}"
DOMAIN_WEIGHTS="${DOMAIN_WEIGHTS:-proportional}"
# DOMAIN_WEIGHTS=/path/to/weights.json  # TODO
DEPTH="${DEPTH:-30}"  # ~830M params, ~17B tokens (ratio 20), estimated single-run wall clock on 8xB200: ~8-12 h
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"  # 32 for B200, 16 for H100
PILE_VAL="${PILE_VAL:-1}"
WANDB_RUN="${WANDB_RUN:-dummy}"
export WANDB_MODE=offline

# Short tag used in checkpoint dir name. Distinguishes runs differing only in weights.
WEIGHTS_NAME=$(basename "$DOMAIN_WEIGHTS" | sed 's/\.json$//')
RUN_TAG="${RUN_TAG:-d${DEPTH}_${WEIGHTS_NAME}}"

# Caches
export UV_CACHE_DIR="${UV_CACHE_DIR:-/raid/$USER/uv_cache}"
export HF_HOME="${HF_HOME:-/raid/$USER/hf_cache}"
export HF_CACHE="${HF_CACHE:-/raid/$USER/hf_cache}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/raid/$USER/nanochat/megatron_pretrain_cache}"
export PILE_VAL_DIR="${PILE_VAL_DIR:-$NANOCHAT_BASE_DIR/pile_val}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$NANOCHAT_BASE_DIR/uv_bin}"
export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$PATH"
# Per-run report dir so multiple runs (e.g. proportional vs weighted) don't clobber each other.
export NANOCHAT_REPORT_DIR="${NANOCHAT_REPORT_DIR:-$NANOCHAT_BASE_DIR/reports/$RUN_TAG}"
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0
mkdir -p "$NANOCHAT_BASE_DIR" "$NANOCHAT_REPORT_DIR" "$UV_INSTALL_DIR"

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
echo "  Pile val dir     : $PILE_VAL_DIR"
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
    if [ ! -f "$PILE_VAL_DIR/pile_val.idx" ]; then
        mkdir -p "$PILE_VAL_DIR"
        (
            flock 9
            if [ ! -f "$PILE_VAL_DIR/pile_val.idx" ]; then
                echo ">> Preparing Pile val (EleutherAI/pile_val_test, full validation split) ..."
                python -m scripts.prepare_pile_val --tokenizer-model "$TOKENIZER_MODEL" \
                    --out-dir "$PILE_VAL_DIR"
            fi
        ) 9>"$PILE_VAL_DIR/.prepare.lock"
    else
        echo ">> Pile val already prepared, skipping."
    fi
    PILE_VAL_FLAG="--pile-val-dir=$PILE_VAL_DIR"
fi

# ---------------------------------------------------------------------------
# Pretraining
# --window-pattern=L \ for B200, not H100
# --core-metric-every=2000 \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --data-source=megatron \
    --data-dir="$DATA_DIR" \
    --domain-weights="$DOMAIN_WEIGHTS" \
    $PILE_VAL_FLAG \
    --depth="$DEPTH" \
    --target-param-data-ratio="$PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --fp8 \
    --sample-every=-1 \
    --core-metric-every=-1 \
    --eval-every=-1 \
    --save-every=1000 \
    --model-tag="$RUN_TAG" \
    --run="$WANDB_RUN"

# ---------------------------------------------------------------------------
# Base eval (CORE + val bpb). Uses the same Megatron data source so
# train/val/pile_val bpb numbers line up with what was logged during training.
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --eval=core,bpb \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$RUN_TAG" \
    --data-source=megatron \
    --data-dir="$DATA_DIR" \
    --domain-weights="$DOMAIN_WEIGHTS" \
    $PILE_VAL_FLAG \
    --max-per-task=-1

# ---------------------------------------------------------------------------
# Final report
python -m nanochat.report generate
