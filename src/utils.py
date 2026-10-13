"""
src/utils.py
────────────
Shared utilities used by all three parts.
  • YAML config loading
  • BLEU / perplexity helpers (thin wrappers so all parts import from one place)
  • WandB login helper
  • HuggingFace Hub push helper
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Config ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── WandB ─────────────────────────────────────────────────────────────────

def wandb_login(key_file: str = "key.wandb"):
    import wandb
    if os.path.exists(key_file):
        with open(key_file) as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        wandb.login()   # falls back to WANDB_API_KEY env var


# ── BLEU ──────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Corpus BLEU using sacrebleu.
    Returns a score in [0, 100].
    """
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score
    except ImportError:
        return float("nan")


# ── Perplexity ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
    max_batches: Optional[int] = None,
) -> float:
    """
    Compute token-level perplexity over a DataLoader.
    Expects batches with keys "input_ids" and "labels".
    """
    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # Support models that return (logits, aux) or just logits
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out

        B, T, V     = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_labels = labels.reshape(B * T)

        nll    = F.cross_entropy(flat_logits, flat_labels, ignore_index=pad_id, reduction="sum")
        n_toks = (flat_labels != pad_id).sum().item()

        total_nll    += nll.item()
        total_tokens += n_toks

    model.train()
    if total_tokens == 0:
        return float("inf")
    avg_nll = total_nll / total_tokens
    return math.exp(min(avg_nll, 20))


# ── HuggingFace Hub ───────────────────────────────────────────────────────

def push_model_to_hub(
    state_dict: dict,
    config_dict: dict,
    tokenizer,
    repo_id: str,
):
    """Upload model weights, config JSON, and tokenizer to the Hub."""
    try:
        from huggingface_hub import HfApi
        import tempfile, json

        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            print(f"[hub] create_repo failed: {e}")
            return

        with tempfile.TemporaryDirectory() as tmp:
            torch.save(state_dict, os.path.join(tmp, "model.pt"))
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            if tokenizer is not None:
                tokenizer.save(os.path.join(tmp, "tokenizer.json"))
            api.upload_folder(folder_path=tmp, repo_id=repo_id)
            print(f"[hub] Pushed to https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"[hub] Push failed: {e}")