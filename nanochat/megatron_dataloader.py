"""
Distributed BOS-aligned best-fit dataloader for pre-tokenized Megatron .bin/.idx
data. Functional twin of nanochat.dataloader.tokenizing_distributed_data_loader_*,
but reads token arrays directly from .bin files instead of tokenizing text.

Key differences vs. the parquet loader:
  - Documents are mmap'd uint16 (or int32) arrays, never decoded.
  - Per-domain (per-file) sampling weights select which file the next doc comes
    from. Useful for ablating mixture weights without re-tokenizing data.
  - Train/val split is a stratified hold-out: the last (1 - train_fraction) of
    every domain is the val slice. Same val for every run.
  - BOS is synthesized at row positions (data on disk doesn't include BOS; the
    Megatron writer with append_eod appends EOS=2 only).

split semantics:
  "train": docs in [0, train_fraction * N) of every domain
  "val":   docs in [train_fraction * N, N) of every domain (held out, no sampling weights)
  "all":   every doc, in domain order, no sampling weights (used for Pile val etc.)
"""

import os
import json
import random

import numpy as np
import torch

from nanochat.common import get_dist_info
from nanochat.megatron_dataset import open_all


def _resolve_weights(weights, datasets):
    """Return list[float] of length len(datasets) summing to 1."""
    if weights is None or weights == "proportional":
        # Each domain contributes in proportion to its token count (i.e. naive
        # concat of all bin/idx files). This is what "treat the data as is" means.
        raw = [float(d.total_tokens) for d in datasets]
    elif weights == "uniform":
        raw = [1.0] * len(datasets)
    elif isinstance(weights, dict):
        # dict mapping prefix-basename -> weight
        raw = []
        for d in datasets:
            name = os.path.basename(d.prefix)
            if name not in weights:
                raise KeyError(f"Domain weight not provided for {name}. Got keys: {list(weights.keys())}")
            raw.append(float(weights[name]))
    elif isinstance(weights, (list, tuple)):
        if len(weights) != len(datasets):
            raise ValueError(f"weights len {len(weights)} != num domains {len(datasets)}")
        raw = [float(w) for w in weights]
    else:
        raise ValueError(f"Unsupported weights spec: {weights!r}")
    total = sum(raw)
    if total <= 0:
        raise ValueError(f"All-zero weights: {raw}")
    return [w / total for w in raw]


def _load_weights_arg(weights_arg):
    """Convert the CLI --domain-weights value to the form _resolve_weights expects.

    Accepts: 'proportional', 'uniform', a JSON file path, or a JSON string.
    """
    if weights_arg in (None, "", "proportional", "uniform"):
        return weights_arg or "proportional"
    if os.path.exists(weights_arg):
        with open(weights_arg, "r") as f:
            return json.load(f)
    try:
        return json.loads(weights_arg)
    except json.JSONDecodeError:
        raise ValueError(
            f"--domain-weights={weights_arg!r}: expected 'proportional', 'uniform', "
            "a JSON file path, or an inline JSON dict/list"
        )


def _build_doc_stream(datasets, probs, split, train_fraction, ddp_rank, ddp_world_size,
                      bos_token_id, resume_state_dict, seed):
    """Generator over (np.int64 token array, state_dict) for one document at a time.

    Each yielded array has BOS prepended. Wraps infinitely (epoch counter
    increments per domain).
    """
    num_domains = len(datasets)
    # Compute per-domain doc ranges for this split (each rank takes a stride).
    starts, ends = [], []
    for d in datasets:
        N = d.num_documents
        train_end = max(1, int(N * train_fraction))
        if split == "train":
            base_start, base_end = 0, train_end
        elif split == "val":
            base_start, base_end = train_end, N
        elif split == "all":
            base_start, base_end = 0, N
        else:
            raise ValueError(f"Unknown split {split!r}")
        # Rank stride: this rank reads indices base_start+ddp_rank, +world, +2*world, ...
        starts.append(base_start + ddp_rank)
        ends.append(base_end)

    # Resume state
    if resume_state_dict is None:
        cursors = list(starts)
        epochs = [1] * num_domains
    else:
        cursors = list(resume_state_dict.get("cursors", starts))
        epochs = list(resume_state_dict.get("epochs", [1] * num_domains))

    # Deterministic per-rank RNG
    rng = random.Random((seed << 20) ^ (ddp_rank << 4) ^ hash(split))

    # For val / all, iterate domains deterministically (round-robin weighted by probs):
    # avoid randomness so val_bpb is reproducible.
    deterministic = split != "train"
    debt = [0.0] * num_domains  # fractional scheduler

    while True:
        if deterministic:
            # Pick the domain whose current debt is highest (after adding its prob).
            for i in range(num_domains):
                debt[i] += probs[i]
            d = max(range(num_domains), key=lambda i: debt[i])
            debt[d] -= 1.0
        else:
            d = rng.choices(range(num_domains), weights=probs, k=1)[0]

        ds = datasets[d]
        c = cursors[d]
        if c >= ends[d]:
            # wrap: new epoch on this domain
            c = starts[d]
            epochs[d] += 1
            cursors[d] = c

        doc_tokens = ds.get_document(c)
        cursors[d] = c + ddp_world_size

        # Build BOS-prepended doc as int64 (model expects long indices).
        out = np.empty(len(doc_tokens) + 1, dtype=np.int64)
        out[0] = bos_token_id
        out[1:] = doc_tokens
        state = {"cursors": list(cursors), "epochs": list(epochs)}
        yield out, state


def megatron_data_loader_with_state(
    tokenizer, B, T, split,
    data_dir,
    weights="proportional",
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000,
    train_fraction=0.99,
    seed=42,
):
    """Distributed BOS-aligned best-fit dataloader over Megatron .bin/.idx data.

    Yields (inputs, targets, state_dict). Same shape contract as
    tokenizing_distributed_data_loader_with_state_bos_bestfit, but reads
    pre-tokenized token streams directly.
    """
    assert split in ("train", "val", "all")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    datasets = open_all(data_dir)
    probs = _resolve_weights(weights, datasets)
    bos_token_id = tokenizer.get_bos_token_id()
    doc_stream = _build_doc_stream(
        datasets, probs, split, train_fraction,
        ddp_rank, ddp_world_size, bos_token_id, resume_state_dict, seed,
    )

    row_capacity = T + 1
    doc_buffer = []  # list of np.int64 arrays
    latest_state = {"cursors": [], "epochs": []}

    def refill_buffer():
        nonlocal latest_state
        while len(doc_buffer) < buffer_size:
            doc, latest_state = next(doc_stream)
            doc_buffer.append(doc)

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                if len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                # Largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    L = len(doc)
                    if L <= remaining and L > best_len:
                        best_idx = i
                        best_len = L
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    L = len(doc)
                    row_buffer[row_idx, pos:pos + L] = torch.from_numpy(doc)
                    pos += L
                else:
                    # No doc fits; crop the shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.from_numpy(doc[:remaining])
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, dict(latest_state)


def megatron_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, _ in megatron_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
