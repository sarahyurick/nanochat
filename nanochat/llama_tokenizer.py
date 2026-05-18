"""
LlamaTokenizer — wraps a Llama2 SentencePiece tokenizer.model and exposes the
same interface as nanochat's RustBPETokenizer so the rest of the pipeline does
not need to know it isn't using nanochat's own BPE.

Used when training on pre-tokenized data produced with Llama2's tokenizer
(e.g. NeMo Curator's MegatronTokenizerWriter with model_identifier=Llama-2-7b).
"""

import os
import sentencepiece as spm


# nanochat's pretraining code only needs <|bos|>. Map it to Llama2 BOS=1.
# Chat-stage tokens (<|user_start|>, etc.) are not used in pretraining/base_eval
# and are intentionally not supported here.
_SPECIAL_TOKEN_TO_ID = {
    "<|bos|>": 1,
    "<|endoftext|>": 2,  # alias if anything reaches for the GPT-2-style name
}


class LlamaTokenizer:
    def __init__(self, sp):
        self.sp = sp
        self._vocab_size = sp.get_piece_size()
        self.bos_token_id = sp.bos_id() if sp.bos_id() >= 0 else 1

    @classmethod
    def from_model_file(cls, model_path):
        sp = spm.SentencePieceProcessor(model_file=model_path)
        return cls(sp)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No tokenizer.model in {tokenizer_dir}")
        return cls.from_model_file(model_path)

    def get_vocab_size(self):
        return self._vocab_size

    def get_special_tokens(self):
        return list(_SPECIAL_TOKEN_TO_ID.keys())

    def id_to_token(self, token_id):
        return self.sp.id_to_piece(token_id)

    def encode_special(self, text):
        if text in _SPECIAL_TOKEN_TO_ID:
            return _SPECIAL_TOKEN_TO_ID[text]
        # Unknown special token name -> None (mirrors HuggingFaceTokenizer behavior)
        return None

    def get_bos_token_id(self):
        return self.bos_token_id

    def _encode_one(self, text, prepend=None, append=None):
        ids = self.sp.encode(text, out_type=int)
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids = [prepend_id] + ids
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids = ids + [append_id]
        return ids

    def encode(self, text, prepend=None, append=None, num_threads=None):
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        if isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return self.sp.decode(list(ids))

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        out_path = os.path.join(tokenizer_dir, "tokenizer.model")
        with open(out_path, "wb") as f:
            f.write(self.sp.serialized_model_proto())
        # Marker file so get_tokenizer() can dispatch.
        with open(os.path.join(tokenizer_dir, "tokenizer_kind.txt"), "w") as f:
            f.write("llama\n")
        print(f"Saved Llama tokenizer to {out_path}")
