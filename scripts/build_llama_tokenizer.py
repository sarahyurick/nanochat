"""
One-shot: build nanochat tokenizer artifacts from a Llama2 SentencePiece model.

Writes to $NANOCHAT_BASE_DIR/tokenizer/:
    tokenizer.model     copy of the .model file (sentencepiece)
    tokenizer_kind.txt  marker so get_tokenizer() returns LlamaTokenizer
    token_bytes.pt      per-token utf-8 byte length, used by val_bpb

Usage:
    python -m scripts.build_llama_tokenizer \
        --tokenizer-model /path/to/tokenizer.model
"""

import argparse
import os
import torch

from nanochat.common import get_base_dir
from nanochat.llama_tokenizer import LlamaTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        required=True,
        help="Path to the Llama2 SentencePiece .model file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer_model):
        raise FileNotFoundError(args.tokenizer_model)

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_model_file(args.tokenizer_model)
    tokenizer.save(tokenizer_dir)

    # Build token_bytes lookup for bits-per-byte evaluation.
    # Per-token utf-8 byte length; for byte-fallback single-byte tokens this is 1,
    # for word-piece tokens it's the byte length of the decoded piece. Special
    # tokens (BOS, EOS, UNK) count as 0 bytes (do not contribute to the bpb
    # denominator).
    sp = tokenizer.sp
    vocab_size = tokenizer.get_vocab_size()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    unk_id = sp.unk_id()
    pad_id = sp.pad_id()
    special_ids = {x for x in (bos_id, eos_id, unk_id, pad_id) if x is not None and x >= 0}

    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes.append(0)
            continue
        # sp.decode([id]) returns the surface form. Byte-fallback tokens decode
        # to the corresponding single byte (or its utf-8-replaced form).
        piece = sp.id_to_piece(token_id)
        # Byte fallback tokens look like "<0xAB>": treat them as one byte.
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            token_bytes.append(1)
            continue
        decoded = sp.decode([token_id])
        token_bytes.append(len(decoded.encode("utf-8")))

    token_bytes_t = torch.tensor(token_bytes, dtype=torch.int32)
    out_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(out_path, "wb") as f:
        torch.save(token_bytes_t, f)
    print(f"Saved token_bytes to {out_path}")

    # Sanity print
    nz = token_bytes_t[token_bytes_t > 0]
    print(f"vocab_size={vocab_size}")
    print(f"bos={bos_id} eos={eos_id} unk={unk_id} pad={pad_id}")
    print(f"token_bytes nonzero: count={nz.numel()}, min={nz.min().item()}, max={nz.max().item()}, mean={nz.float().mean().item():.2f}")

    # Quick roundtrip test
    test = "Hello world! Numbers: 123. Unicode: 你好 🌍"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    print(f"roundtrip: {decoded!r}")
    print(f"           ids={ids}")


if __name__ == "__main__":
    main()
