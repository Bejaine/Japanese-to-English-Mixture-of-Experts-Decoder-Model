"""
src/part1/data.py
─────────────────
Data pipeline for Part 1, aligned with A1 dataset.py conventions:

  • Special token IDs: PAD=0, SOS=1, EOS=2, UNK=3  (matches A1 exactly)
  • HuggingFace tokenizers BPE (byte-level) – handles Japanese / Vietnamese
    natively without romanisation.
  • Sequences are NOT pre-padded to a fixed length; collate_fn uses
    pad_sequence (same pattern as A1 collate_fn).
  • Each triplet (en, vi, ja) → two samples:
        [SOS] [VI] vi_text [EN] en_text [EOS]
        [SOS] [JA] ja_text [EN] en_text [EOS]
  • Dataset split: 95% train / 2.5% val / 2.5% test (fixed seed).
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset


# ── Special token IDs (mirrors A1 dataset.py) ─────────────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Language and meta special tokens
# These are added AFTER the four core ones so they get IDs 4+.
_LANG_TOKENS = ["[VI]", "[JA]", "[EN]"]

# The ordered list for the BPE trainer – order matters for ID assignment
_SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"] + _LANG_TOKENS

VI_TOKEN = "[VI]"
JA_TOKEN = "[JA]"
EN_TOKEN = "[EN]"


# ══════════════════════════════════════════════════════════════════════════
# Tokenizer utilities
# ══════════════════════════════════════════════════════════════════════════

def _text_iterator(dataset, max_samples: int) -> Iterator[str]:
    """Yield raw strings (all three languages) for BPE training."""
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
        yield row["en"]
        yield row["vi"]
        yield row["ja"]


def build_tokenizer(
    dataset,
    vocab_size: int,
    save_path: str,
    max_samples: int = 500_000,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on the trilingual corpus.

    Special tokens are fixed at positions 0–6:
        0: <pad>   1: <sos>   2: <eos>   3: <unk>
        4: [VI]    5: [JA]    6: [EN]

    Byte-level pre-tokenisation handles CJK and Vietnamese diacritics
    without any language-specific preprocessing.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    # Without this, decode() never reverses the byte-level mapping: output
    # text keeps literal "Ġ"/"Ċ" markers and mis-assembled UTF-8 for
    # multi-byte (Vietnamese/Japanese) characters instead of real text.
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size    = vocab_size,
        special_tokens = _SPECIAL_TOKENS,
        min_frequency = 2,
        show_progress = True,
    )

    print(f"[data] Training BPE tokenizer (vocab_size={vocab_size}) …")
    tokenizer.train_from_iterator(
        _text_iterator(dataset, max_samples),
        trainer = trainer,
    )

    # Sanity-check the IDs we depend on
    assert tokenizer.token_to_id("<pad>") == PAD_IDX, "PAD must be 0"
    assert tokenizer.token_to_id("<sos>") == SOS_IDX, "SOS must be 1"
    assert tokenizer.token_to_id("<eos>") == EOS_IDX, "EOS must be 2"
    assert tokenizer.token_to_id("<unk>") == UNK_IDX, "UNK must be 3"

    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, "tokenizer.json")
    tokenizer.save(out)
    print(f"[data] Tokenizer saved to {out}")
    return tokenizer


def load_tokenizer(save_path: str) -> Tokenizer:
    path = os.path.join(save_path, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. "
            "Run training once (it builds the tokenizer first)."
        )
    return Tokenizer.from_file(path)


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """
    Each triplet row (en, vi, ja) produces two sequence samples:
        [SOS] [VI] <vi tokens> [EN] <en tokens> [EOS]
        [SOS] [JA] <ja tokens> [EN] <en tokens> [EOS]

    Sequences longer than N_ctx are truncated (keeping SOS and EOS).
    No pre-padding – the collate_fn handles dynamic padding per batch.

    The 'lang' field ("vi" / "ja") is stored for expert-heatmap analysis.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer: Tokenizer,
        N_ctx: int,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.N_ctx     = N_ctx

        vi_id = tokenizer.token_to_id(VI_TOKEN)
        ja_id = tokenizer.token_to_id(JA_TOKEN)
        en_id = tokenizer.token_to_id(EN_TOKEN)

        limit = max_samples if max_samples else len(hf_dataset)
        print(
            f"[data] Encoding {min(limit, len(hf_dataset))} triplets "
            f"→ up to {2 * min(limit, len(hf_dataset))} samples …"
        )

        # Each entry: (ids_tensor, lang_str, src_len_int)
        self.samples: List[Tuple[torch.Tensor, str, int]] = []

        for i, row in enumerate(hf_dataset):
            if i >= limit:
                break
            for lang_id, src_text, lang_str in [
                (vi_id, row["vi"], "vi"),
                (ja_id, row["ja"], "ja"),
            ]:
                ids, src_len = self._encode_pair(lang_id, src_text, en_id, row["en"])
                self.samples.append((ids, lang_str, src_len))

        print(f"[data] Dataset ready: {len(self.samples)} samples.")

    def _encode_pair(
        self,
        src_lang_id: int,
        src_text: str,
        tgt_lang_id: int,
        tgt_text: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        Build: [SOS] [LANG] <src_tokens> [EN] <tgt_tokens> [EOS]
        Strip any SOS/EOS auto-added by the tokenizer from inner texts.
        Truncate to N_ctx (keeps SOS at front, EOS at back).

        Returns
        -------
        ids     : LongTensor of shape (seq_len,)
        src_len : number of tokens in ids UP TO AND INCLUDING the [EN] boundary,
                  i.e. the position of [EN] in ids (fix #1 – used by BLEU eval
                  to slice source prompt without cutting through target text).
        """
        # Encode without special boundaries – we control layout manually
        src_ids = [t for t in self.tokenizer.encode(src_text).ids
                   if t not in (SOS_IDX, EOS_IDX)]
        tgt_ids = [t for t in self.tokenizer.encode(tgt_text).ids
                   if t not in (SOS_IDX, EOS_IDX)]

        # src_part = [SOS, LANG] + src_tokens; EN separator marks the boundary
        src_part = [SOS_IDX, src_lang_id] + src_ids
        ids      = src_part + [tgt_lang_id] + tgt_ids + [EOS_IDX]

        # src_len = index of [EN] token in ids = len(src_part)
        # After input_ids = ids[:-1], the prompt is input_ids[:src_len].
        src_len = len(src_part)

        # Truncate: keep SOS at [0] and EOS at [-1]
        if len(ids) > self.N_ctx:
            ids = ids[: self.N_ctx - 1] + [EOS_IDX]
            # Clamp src_len so it never exceeds the (truncated) input_ids length
            src_len = min(src_len, self.N_ctx - 2)

        return torch.tensor(ids, dtype=torch.long), src_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ids, lang, src_len = self.samples[idx]
        return {
            "input_ids": ids[:-1],   # (T-1,) – input to model
            "labels":    ids[1:],    # (T-1,) – shifted targets
            "lang":      lang,
            # src_len: number of source tokens in input_ids (up to [EN] boundary).
            # input_ids[:src_len] = the source prompt; input_ids[src_len:] = target.
            # Used by evaluate_bleu to slice cleanly without guessing (fix #1).
            "src_len":   src_len,
        }


# ══════════════════════════════════════════════════════════════════════════
# Collate + DataLoaders
# ══════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """
    Dynamic padding with pad_sequence (A1 style).
    Sequences are right-padded to the longest in the batch.
    src_lens is stacked as a LongTensor so evaluate_bleu can slice each
    sample at the correct [EN] boundary (fix #1).
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first   = True,
        padding_value = PAD_IDX,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch],
        batch_first   = True,
        padding_value = PAD_IDX,
    )
    langs     = [b["lang"] for b in batch]
    src_lens  = torch.tensor([b["src_len"] for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels":    labels,
        "langs":     langs,
        "src_lens":  src_lens,   # (B,) – index of [EN] token in each input_ids row
    }


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Length-bucketed batches: shuffle, cut into chunks of `bucket_mult` batches,
    sort each chunk by length, split into batches and shuffle the batch order.
    With random batching only ~31% of positions were real tokens (median
    sample 30 tokens, batch max often >100); bucketing removes most of the
    padding compute while each epoch still visits every sample exactly once.
    """

    def __init__(self, lengths: List[int], batch_size: int, bucket_mult: int = 100,
                 seed: int = 42):
        self.lengths     = lengths
        self.batch_size  = batch_size
        self.bucket_mult = bucket_mult
        self.seed        = seed
        self.epoch       = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        chunk   = self.batch_size * self.bucket_mult
        batches = []
        for i in range(0, len(idx), chunk):
            c = sorted(idx[i:i + chunk], key=lambda j: self.lengths[j])
            batches += [c[k:k + self.batch_size] for k in range(0, len(c), self.batch_size)]
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def get_dataloaders(
    cfg: dict,
    tokenizer: Tokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataset, create an 80/10/10 split, return (train, val, test) DataLoaders.
    """
    print(f"[data] Loading {cfg['dataset_name']} …")
    raw = load_dataset(cfg["dataset_name"], split="train")

    n      = len(raw)
    n_val  = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))

    indices = list(range(n))
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(indices)

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test: n_test + n_val]
    train_idx = indices[n_test + n_val:]

    N_ctx = cfg["N_ctx"]
    max_s = cfg.get("max_samples", None)

    train_ds = TranslationDataset(raw.select(train_idx), tokenizer, N_ctx, max_samples=max_s)
    val_ds   = TranslationDataset(raw.select(val_idx),   tokenizer, N_ctx)
    test_ds  = TranslationDataset(raw.select(test_idx),  tokenizer, N_ctx)

    bs = cfg["batch_size"]
    if cfg.get("bucket_batches", False):
        lengths  = [len(ids) - 1 for ids, _, _ in train_ds.samples]
        train_dl = DataLoader(
            train_ds, batch_sampler=BucketBatchSampler(lengths, bs, seed=cfg.get("seed", 42)),
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    return train_dl, val_dl, test_dl