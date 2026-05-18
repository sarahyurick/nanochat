"""
One-shot: download Pile val (default: EleutherAI/pile_val_test, validation split),
tokenize with the same Llama2 tokenizer used for the training data, and write a
single Megatron .bin/.idx pair that our megatron_dataloader can read.

By default writes to: $NANOCHAT_BASE_DIR/pile_val/pile_val.{bin,idx}

Usage:
    python -m scripts.prepare_pile_val
    python -m scripts.prepare_pile_val --hf-dataset EleutherAI/pile_val_test --split validation
    python -m scripts.prepare_pile_val --input-jsonl /path/to/val.jsonl --text-field text
"""

import argparse
import json
import os
import struct

import numpy as np
import sentencepiece as spm

from nanochat.common import get_base_dir


_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODE_UINT16 = 8


def _write_megatron_pair(out_prefix, seq_lengths, token_size=2, dtype_code=_DTYPE_CODE_UINT16):
    """Finalize an in-progress .bin file by writing its companion .idx."""
    seq_lengths = np.asarray(seq_lengths, dtype=np.int32)
    # Byte offsets: cumulative sum * token_size, shifted by one (first seq at offset 0).
    seq_pointers = np.zeros(len(seq_lengths), dtype=np.int64)
    if len(seq_lengths) > 1:
        seq_pointers[1:] = np.cumsum(seq_lengths[:-1].astype(np.int64)) * token_size
    document_indices = np.arange(len(seq_lengths) + 1, dtype=np.int64)
    with open(out_prefix + ".idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))                       # version
        f.write(struct.pack("<B", dtype_code))              # dtype code
        f.write(struct.pack("<Q", len(seq_lengths)))        # sequence count
        f.write(struct.pack("<Q", len(document_indices)))   # document count
        f.write(seq_lengths.tobytes(order="C"))
        f.write(seq_pointers.tobytes(order="C"))
        f.write(document_indices.tobytes(order="C"))


def iter_docs_from_jsonl(path, text_field):
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            yield obj[text_field]


def iter_docs_from_hf(hf_dataset, split, text_field, max_docs):
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_docs is not None and max_docs > 0 and i >= max_docs:
            break
        yield row[text_field]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-model", type=str,
                        default="/path/to/tokenizer.model")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Output directory (default: $NANOCHAT_BASE_DIR/pile_val/)")
    parser.add_argument("--out-name", type=str, default="pile_val",
                        help="Output file name prefix (default: pile_val)")
    parser.add_argument("--hf-dataset", type=str, default="EleutherAI/pile_val_test")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--input-jsonl", type=str, default="",
                        help="Use this local jsonl file instead of an HF dataset")
    parser.add_argument("--max-docs", type=int, default=-1,
                        help="Cap number of documents (-1 = use entire split; Pile val is ~214k docs / ~380M tokens)")
    parser.add_argument("--append-eod", action="store_true", default=True,
                        help="Append EOS=2 at end of each doc (matches the training data convention)")
    parser.add_argument("--no-append-eod", action="store_false", dest="append_eod")
    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = os.path.join(get_base_dir(), "pile_val")
    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, args.out_name)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2
    vocab = sp.get_piece_size()
    assert vocab <= 2**16, f"uint16 only supports vocab <= 65536, got {vocab}"

    if args.input_jsonl:
        print(f"Reading from local jsonl: {args.input_jsonl}")
        docs = iter_docs_from_jsonl(args.input_jsonl, args.text_field)
    else:
        print(f"Streaming from HF: {args.hf_dataset} split={args.split}")
        docs = iter_docs_from_hf(args.hf_dataset, args.split, args.text_field, args.max_docs)

    seq_lengths = []
    total_tokens = 0
    with open(out_prefix + ".bin", "wb") as bin_file:
        n_docs = 0
        for text in docs:
            if not text:
                continue
            ids = sp.encode(text, out_type=int)
            if args.append_eod:
                ids.append(eos_id)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.uint16)
            bin_file.write(arr.tobytes(order="C"))
            seq_lengths.append(len(arr))
            total_tokens += len(arr)
            n_docs += 1
            if n_docs % 1000 == 0:
                print(f"  {n_docs:,} docs, {total_tokens:,} tokens")

    _write_megatron_pair(out_prefix, seq_lengths)
    print(f"Wrote {n_docs:,} docs / {total_tokens:,} tokens to {out_prefix}.{{bin,idx}}")


if __name__ == "__main__":
    main()
