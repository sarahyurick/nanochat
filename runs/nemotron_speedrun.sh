#!/bin/bash

# TODO: Update as needed
export UV_CACHE_DIR=/raid/syurick/uv_cache
export PRE_COMMIT_HOME=/raid/syurick/cache
export HF_CACHE=/raid/syurick/hf_cache
export HF_HOME=/raid/syurick/hf_cache
export NCCL_NVLS_ENABLE=0
export NANOCHAT_BASE_DIR="/raid/syurick/nanochat/nemotron_speedrun_cache"

export OMP_NUM_THREADS=1
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# TODO: Update number of files as needed
# Each Parquet file in https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/main/High-Quality is 1-1.5 GB
# We download ~40 GB to match https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh
python -m nanochat.nemotron_dataset -n 3
python -m nanochat.nemotron_dataset -n 35 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# TODO: Update nproc_per_node to # of GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# TODO: Update nproc_per_node to # of GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
python -m nanochat.report generate
