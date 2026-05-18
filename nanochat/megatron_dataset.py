"""
Reader for pre-tokenized Megatron MMIDIDX v1 .bin/.idx files (as produced by
nemo_curator's MegatronTokenizerWriter).

Index file layout (little-endian):
    9 bytes : magic b"MMIDIDX\x00\x00"
    8 bytes : version (uint64, = 1)
    1 byte  : dtype code (uint8). 8 = uint16 (Llama2 vocab fits), 4 = int32.
    8 bytes : sequence_count S (uint64)
    8 bytes : document_count D (uint64, = S + 1 in our data)
    4 * S   : sequence_lengths (int32)
    8 * S   : sequence_pointers (int64, byte offsets into .bin)
    8 * D   : document_indices  (int64, maps doc_id -> first sequence_id)

For our data, the writer wrote one sequence per document, so document_indices
is just [0, 1, 2, ..., S]. We expose iteration at the "document" granularity,
which here is the natural unit (each text item from the source jsonl/parquet).
"""

import os
import struct
import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODE_TO_NP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


class MegatronDataset:
    """Mmap-backed reader for a single .bin/.idx pair.

    After construction:
        self.dtype             : np.dtype of tokens in .bin
        self.num_documents     : number of documents (== number of sequences)
        self.total_tokens      : total token count across all documents
        self.sequence_lengths  : int32 array, len == num_documents
        self.sequence_pointers : int64 array, byte offsets into .bin
        self.tokens            : np.memmap into the .bin file (1D, len total_tokens)
    """

    def __init__(self, prefix):
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"
        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(bin_path)
        self.prefix = prefix

        with open(idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _INDEX_HEADER:
                raise ValueError(f"Bad magic in {idx_path}: {magic!r}")
            version = struct.unpack("<Q", f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported MMIDIDX version {version} in {idx_path}")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            if dtype_code not in _DTYPE_CODE_TO_NP:
                raise ValueError(f"Unknown dtype code {dtype_code} in {idx_path}")
            self.dtype = np.dtype(_DTYPE_CODE_TO_NP[dtype_code])
            seq_count = struct.unpack("<Q", f.read(8))[0]
            doc_count = struct.unpack("<Q", f.read(8))[0]

            header_bytes = 9 + 8 + 1 + 8 + 8  # 34
            offset = header_bytes
            self.sequence_lengths = np.memmap(idx_path, mode="r", dtype=np.int32,
                                              offset=offset, shape=(seq_count,))
            offset += seq_count * 4
            self.sequence_pointers = np.memmap(idx_path, mode="r", dtype=np.int64,
                                                offset=offset, shape=(seq_count,))
            offset += seq_count * 8
            # document_indices is len doc_count (= seq_count+1 in our data).
            # Not used for iteration since we have 1 sequence per document, but
            # we expose it for completeness.
            self.document_indices = np.memmap(idx_path, mode="r", dtype=np.int64,
                                               offset=offset, shape=(doc_count,))

        self.num_documents = int(seq_count)
        self.total_tokens = int(self.sequence_lengths.sum(dtype=np.int64))
        # Memory-map the .bin as a flat array of tokens.
        token_size = self.dtype.itemsize
        bin_size = os.path.getsize(bin_path)
        if bin_size % token_size != 0:
            raise ValueError(f"{bin_path} size {bin_size} not multiple of token size {token_size}")
        self.tokens = np.memmap(bin_path, mode="r", dtype=self.dtype,
                                shape=(bin_size // token_size,))

    def __len__(self):
        return self.num_documents

    def get_document(self, doc_idx):
        """Return the token array for a single document as a numpy view (no copy)."""
        # Sequence pointer is in BYTES from start of .bin; convert to token index.
        byte_offset = int(self.sequence_pointers[doc_idx])
        length = int(self.sequence_lengths[doc_idx])
        token_offset = byte_offset // self.dtype.itemsize
        return self.tokens[token_offset:token_offset + length]


def list_bin_idx_prefixes(data_dir):
    """Return sorted list of prefixes (paths without the .bin/.idx suffix) in a directory."""
    prefixes = set()
    for name in os.listdir(data_dir):
        if name.endswith(".bin") or name.endswith(".idx"):
            prefixes.add(os.path.join(data_dir, name[:-4]))
    out = []
    for p in sorted(prefixes):
        if os.path.exists(p + ".bin") and os.path.exists(p + ".idx"):
            out.append(p)
    return out


def open_all(data_dir):
    """Open every .bin/.idx pair in data_dir, return list of MegatronDataset."""
    prefixes = list_bin_idx_prefixes(data_dir)
    if not prefixes:
        raise FileNotFoundError(f"No .bin/.idx pairs found in {data_dir}")
    return [MegatronDataset(p) for p in prefixes]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect Megatron .bin/.idx files")
    parser.add_argument("data_dir", type=str, help="Directory containing .bin/.idx pairs")
    args = parser.parse_args()

    datasets = open_all(args.data_dir)
    total_docs = 0
    total_tokens = 0
    for ds in datasets:
        name = os.path.basename(ds.prefix)
        print(f"{name:20s}  docs={ds.num_documents:>12,}  tokens={ds.total_tokens:>15,}  dtype={ds.dtype}")
        total_docs += ds.num_documents
        total_tokens += ds.total_tokens
    print(f"{'TOTAL':20s}  docs={total_docs:>12,}  tokens={total_tokens:>15,}")
