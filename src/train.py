"""
src/part1/train.py
──────────────────
Training loop for all five MoE variants.

Style follows A1 train.py conventions:
  • tqdm progress bar per epoch-equivalent (log_every steps)
  • Metrics computed similarly to A1's calculate_metrics
  • pad_sequence-based collation (no fixed-length pre-padding)

Assignment-specific requirements kept:
  • Step-based loop (not epoch) for controlled token budgets
  • WandB logging at intervals; eval at eval_every steps
  • HF Hub push after each variant
  • Cosine LR with linear warmup
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.part1.data import (
    PAD_IDX, SOS_IDX, EOS_IDX,
    build_tokenizer, load_tokenizer, get_dataloaders,
)
from src.part1.model import TransformerConfig, Transformer, build_variant


# ══════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _make_base_config(cfg: dict) -> TransformerConfig:
    return TransformerConfig(
        D_model    = cfg["D_model"],
        N_ctx      = cfg["N_ctx"],
        N_heads    = cfg["N_heads"],
        N_layers   = cfg["N_layers"],
        D_ff       = cfg["D_ff"],
        vocab_size = cfg["tokenizer_vocab_size"],
        dropout    = cfg["dropout"],
        use_rope   = cfg.get("use_rope", True),
        norm_type  = cfg.get("norm_type", "rms"),
    )


# ══════════════════════════════════════════════════════════════════════════
# LR schedule: cosine with linear warmup
# ══════════════════════════════════════════════════════════════════════════

def _wb(fn, *args, **kw):
    """Call a wandb function without ever letting a WandB hiccup kill training
    (every metric is also kept in the local history.json)."""
    try:
        return fn(*args, **kw)
    except Exception as e:
        print(f"[wandb] {getattr(fn, '__name__', fn)} failed: {e}")


def _wandb_init(**kw):
    """Retry online init a few times, then fall back to offline (synced later)."""
    for attempt in range(3):
        try:
            return wandb.init(**kw)
        except Exception as e:
            print(f"[wandb] init failed ({e}); retry {attempt + 1}/3")
            time.sleep(10)
    os.environ["WANDB_MODE"] = "offline"
    return wandb.init(**kw)


def _count_train_tokens(train_dl: DataLoader) -> int:
    """Real (non-PAD) target tokens in one pass over the training set — the
    unit of the `epochs` token budget (same for every variant)."""
    return sum(len(ids) - 1 for ids, _, _ in train_dl.dataset.samples)


def _get_lr(step: int, warmup: int, max_steps: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════════
# Mask helpers  (A1 utils.py style, adapted for decoder-only)
# ══════════════════════════════════════════════════════════════════════════

def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """(B, 1, 1, T) bool mask – True where token is NOT padding."""
    return (seq != PAD_IDX).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """(1, 1, T, T) lower-triangular bool mask."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
                      ).unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════
# Greedy decode  (A1 greedy_decode style, decoder-only version)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Autoregressively generate up to max_new_tokens tokens from prompt.

    Fix #6: tracks a per-sequence 'finished' boolean so that as soon as a
    sequence emits EOS_IDX it stops contributing new tokens. Generation halts
    early once every sequence in the batch has emitted EOS.

    Parameters
    ----------
    prompt : (B, T_prompt) – already on the correct device

    Returns
    -------
    gen : (B, T_prompt + generated_len)
    """
    gen      = prompt.clone()
    B        = gen.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=gen.device)

    for _ in range(max_new_tokens):
        ctx       = gen[:, -model.config.N_ctx:]
        logits, _ = model(ctx)
        next_tok  = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        # For already-finished sequences, emit PAD instead of overwriting
        next_tok[finished] = PAD_IDX

        gen      = torch.cat([gen, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(-1) == EOS_IDX)

        if finished.all():
            break

    return gen


# ══════════════════════════════════════════════════════════════════════════
# Evaluation: loss + perplexity
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = 50,
) -> Dict[str, float]:
    """Token-level cross-entropy and perplexity, ignoring PAD positions."""
    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        logits, _ = model(input_ids)

        B, T, V     = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_labels = labels.reshape(B * T)

        nll    = F.cross_entropy(flat_logits, flat_labels,
                                 ignore_index=PAD_IDX, reduction="sum")
        n_toks = (flat_labels != PAD_IDX).sum().item()

        total_nll    += nll.item()
        total_tokens += n_toks

    model.train()
    if total_tokens == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf")}

    avg_loss = total_nll / total_tokens
    ppl      = math.exp(min(avg_loss, 20))
    return {"val_loss": avg_loss, "val_ppl": ppl}


# ══════════════════════════════════════════════════════════════════════════
# Evaluation: BLEU  (sacrebleu corpus BLEU)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 128,
    max_batches: int    = 10,
    variant_name: str   = "",
    tag: str            = "val",
    step: Optional[int] = None,
    n_print: int        = 3,
    samples: Optional[list] = None,
) -> float:
    """
    Greedy-decode source prompts; compute corpus BLEU against target references.
    Returns BLEU score in [0, 100].

    Fix #1: uses batch["src_lens"] (set by collate_fn) to slice each sequence at
    the [EN] boundary rather than blindly at T//2.

    Fix #2: decodes one sample at a time to avoid right-padding contaminating the
    prompt. (The model would otherwise try to predict after a run of <pad> tokens.)
    This is acceptable because BLEU eval runs on only max_batches batches.

    Prints the first n_print (source, reference, generation) triples to the log
    (via tqdm.write) so translation quality can be inspected as training runs.
    """
    try:
        import sacrebleu
    except ImportError:
        return float("nan")

    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []
    n_printed = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"]   # (B, T) – keep on CPU for slicing
        labels    = batch["labels"]      # (B, T)
        src_lens  = batch["src_lens"]    # (B,) – index of [EN] in input_ids
        langs     = batch["langs"]       # (B,) – "vi" / "ja" per sample

        B = input_ids.size(0)

        # Fix #2: iterate sample-by-sample to avoid right-padding in prompt.
        # Fix #1 (off-by-one): include the [EN] token in the prompt so the
        # model's first generated token is the first translated word, not [EN].
        #
        # Token layout in input_ids:
        #   [SOS, LANG, src..., EN, tgt...]
        #    0    1     2..sl-1  sl  sl+1..
        # sl = src_len = index of [EN] in input_ids
        # prompt = input_ids[:sl+1] → ends with [EN], no padding
        # ref    = labels[sl:]      → ids[sl+1:] → [tgt_1, ..., EOS]
        # gen    = gen_b[0, sl+1:] → tokens generated after [EN]
        for b in range(B):
            sl = src_lens[b].item()                        # index of [EN] in input_ids

            # Prompt includes [EN] so model generates translations, not [EN]
            prompt_b = input_ids[b, : sl + 1].unsqueeze(0).to(device)  # (1, sl+1)
            ref_b    = labels[b, sl:].tolist()             # ids[sl+1:] = [tgt..., EOS]

            gen_b   = greedy_decode(model, prompt_b, max_new_tokens)    # (1, sl+1+gen)
            gen_ids = gen_b[0, sl + 1:].tolist()           # tokens after [EN]

            # Strip special tokens before decoding
            gen_ids = [t for t in gen_ids if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
            ref_ids = [t for t in ref_b  if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]

            hyp_text = tokenizer.decode(gen_ids)
            ref_text = tokenizer.decode(ref_ids)
            hypotheses.append(hyp_text)
            references.append(ref_text)

            if n_printed < n_print:
                # [SOS, LANG] + src_ids → strip SOS/LANG/EN/pad for display
                src_ids  = [t for t in input_ids[b, : sl].tolist()
                            if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                src_text = tokenizer.decode(src_ids)
                header = f"{variant_name} step={step} {tag} sample {n_printed + 1} [{langs[b]}]" \
                    if tag == "val" else f"{variant_name} {tag} sample {n_printed + 1} [{langs[b]}]"
                tqdm.write(f"--- {header} ---")
                tqdm.write(f"  SRC : {src_text}")
                tqdm.write(f"  REF : {ref_text}")
                tqdm.write(f"  OUT : {hyp_text}")
                n_printed += 1
                if samples is not None:
                    samples.append({"step": step, "lang": langs[b], "source": src_text,
                                    "reference": ref_text, "output": hyp_text})

    model.train()
    if not hypotheses:
        return 0.0
    return sacrebleu.corpus_bleu(hypotheses, [references], force=True).score


# ══════════════════════════════════════════════════════════════════════════
# Core training loop for a single variant
# ══════════════════════════════════════════════════════════════════════════

def train_variant(
    variant: int,
    cfg: dict,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    tokenizer,
    device: torch.device,
) -> tuple:
    """Train one FFN variant and return (model, test_metrics)."""

    base_config             = _make_base_config(cfg)
    model, variant_name     = build_variant(variant, base_config)
    model                   = model.to(device)

    # ── WandB init ────────────────────────────────────────────────────────
    run_name = cfg.get("wandb_run_name_fmt", "part1_v{variant}_{run_tag}").format(
        variant=variant, run_tag=cfg.get("wandb_run_tag", "run01")
    )
    out_dir = Path(cfg["output_dir"]) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "latest.pt"
    resume = (torch.load(resume_path, map_location=device, weights_only=False)
              if resume_path.exists() else None)
    # Re-attach to the same WandB run when resuming after an interruption
    id_path  = out_dir / "wandb_id.txt"
    # (only when training state is actually restored — a from-scratch restart
    # gets a fresh run so it never overlays an earlier attempt's steps)
    wandb_id = (id_path.read_text().strip() if id_path.exists() and resume is not None
                else uuid.uuid4().hex[:8])
    id_path.write_text(wandb_id)
    _wandb_init(
        id      = wandb_id,
        resume  = "allow",
        project = cfg.get("wandb_project", "anlp-a2-part1"),
        entity  = cfg.get("wandb_entity", None),
        name    = run_name,
        group   = f"part1_{cfg.get('wandb_run_tag', 'run01')}",
        tags    = ["part1", cfg.get("wandb_run_tag", "run01"), variant_name],
        config  = {**cfg, "variant": variant, "variant_name": variant_name,
                   **model.count_parameters()},
        reinit  = True,
    )
    _wb(wandb.define_metric, "train/tokens_seen")
    _wb(wandb.define_metric, "val/*", step_metric="train/tokens_seen")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["learning_rate"],
        betas        = (cfg["beta1"], cfg["beta2"]),
        eps          = cfg["eps"],
        weight_decay = cfg["weight_decay"],
    )

    max_steps  = cfg["max_steps"]
    max_tokens = cfg.get("max_tokens")   # primary stop criterion; max_steps is a safety ceiling
    epoch_tokens = 0
    if cfg.get("epochs"):
        # Token budget = epochs × real tokens per pass over the train split,
        # identical for all variants. max_steps is set from it so the cosine
        # schedule actually decays to ~0 by the end of the budget.
        epoch_tokens = _count_train_tokens(train_dl)
        max_tokens   = int(cfg["epochs"] * epoch_tokens)
        tok_per_step = epoch_tokens / len(train_dl) * cfg.get("grad_accum_steps", 1)
        max_steps    = min(max_steps, math.ceil(max_tokens / tok_per_step))
        print(f"[{variant_name}] epochs={cfg['epochs']}  epoch_tokens={epoch_tokens:,}  "
              f"max_tokens={max_tokens:,}  ≈max_steps={max_steps:,}")
        _wb(wandb.config.update, {"epoch_tokens": epoch_tokens, "max_tokens_effective": max_tokens,
                                  "max_steps_effective": max_steps}, allow_val_change=True)
    use_amp    = cfg.get("amp", False) and device.type == "cuda"
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: List[dict] = []
    warmup     = cfg["warmup_steps"]
    aux_alpha  = cfg.get("aux_loss_alpha", 0.01)
    accum      = cfg.get("grad_accum_steps", 1)
    log_every  = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 500)
    save_every = cfg.get("save_every", 2000)

    # ── Training ──────────────────────────────────────────────────────────
    step         = 0
    tokens_seen  = 0
    running_loss = 0.0
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        step, tokens_seen, history = resume["step"], resume["tokens_seen"], resume["history"]
        if hasattr(train_dl.batch_sampler, "epoch"):
            train_dl.batch_sampler.epoch = resume.get("sampler_epoch", 0)
        print(f"[{variant_name}] Resumed from {resume_path}: step={step:,} tokens_seen={tokens_seen:,}")
        del resume
    t0           = time.time()
    tokens_start = tokens_seen
    model.train()

    def _save_latest():
        # Resume point (overwritten in place; rename makes it atomic)
        tmp = out_dir / "latest.pt.tmp"
        torch.save({
            "step": step, "tokens_seen": tokens_seen, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": base_config, "variant": variant,
            "sampler_epoch": getattr(train_dl.batch_sampler, "epoch", 0),
        }, tmp)
        os.replace(tmp, resume_path)

    def _save_checkpoint(step: int) -> Path:
        ckpt = out_dir / f"step_{step}.pt"
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    base_config,
            "variant":   variant,
        }, ckpt)
        return ckpt

    train_iter = iter(train_dl)
    pbar = tqdm(total=max_steps, initial=step, desc=f"[{variant_name}]", unit="step",
                mininterval=30)

    while step < max_steps and (max_tokens is None or tokens_seen < max_tokens):
        # LR update
        lr = _get_lr(step, warmup, max_steps, cfg["learning_rate"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch      = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, aux_loss = model(input_ids)

                B, T, V     = logits.shape
                flat_logits = logits.reshape(B * T, V)
                flat_labels = labels.reshape(B * T)

                ce_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=PAD_IDX)
                loss    = (ce_loss + aux_alpha * aux_loss.float()) / accum
            scaler.scale(loss).backward()

            accum_loss  += loss.item()
            tokens_seen += int((flat_labels != PAD_IDX).sum().item())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss
        step         += 1
        pbar.update(1)

        # ── Logging ───────────────────────────────────────────────────────
        if step % log_every == 0:
            avg_loss  = running_loss / log_every
            elapsed   = time.time() - t0
            tok_per_s = (tokens_seen - tokens_start) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}",
                             tok_per_s=f"{tok_per_s:.0f}")
            _wb(wandb.log,
                {"train/loss": avg_loss, "train/lr": lr,
                 "train/tokens_seen": tokens_seen,
                 "train/epoch": tokens_seen / epoch_tokens if cfg.get("epochs") else 0.0,
                 "train/tok_per_s": tok_per_s},
                step=step,
            )
            running_loss = 0.0

        # ── Evaluation ────────────────────────────────────────────────────
        if step % eval_every == 0:
            val_samples: List[dict] = []
            metrics = evaluate(model, val_dl, device)
            bleu    = evaluate_bleu(
                model, val_dl, tokenizer, device,
                max_new_tokens = cfg.get("max_new_tokens", 128),
                max_batches    = cfg.get("val_bleu_batches", 10),
                variant_name   = variant_name, tag = "val", step = step,
                n_print        = cfg.get("n_print_samples", 3), samples = val_samples,
            )
            metrics["val_bleu"] = bleu
            tqdm.write(
                f"[{variant_name}] step={step}  epoch={tokens_seen / max(1, epoch_tokens):.2f}  "
                f"val_loss={metrics['val_loss']:.4f}  "
                f"val_ppl={metrics['val_ppl']:.2f}  "
                f"val_bleu={bleu:.2f}"
            )
            log = {**{f"val/{k}": v for k, v in metrics.items()}, "train/tokens_seen": tokens_seen}
            if val_samples:
                log["val/samples"] = wandb.Table(columns=list(val_samples[0].keys()),
                                                 data=[list(d.values()) for d in val_samples])
            _wb(wandb.log, log, step=step)
            history.append({"step": step, "tokens_seen": tokens_seen, **metrics})
            with open(out_dir / "history.json", "w") as f:
                json.dump({"variant": variant_name, "history": history}, f, indent=2)

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % save_every == 0:
            _save_latest()
            tqdm.write(f"[{variant_name}] Resume checkpoint → {resume_path} (step {step})")

    pbar.close()

    print(
        f"[{variant_name}] Stopped at step={step:,}  tokens_seen={tokens_seen:,}  "
        f"(max_tokens={max_tokens}, max_steps={max_steps})"
    )

    # The loop can exit early on the token budget, between save_every
    # intervals — guarantee a checkpoint exists at the exact stopping step.
    _save_latest()   # an interrupted test eval resumes straight into evaluation
    final_ckpt_path = out_dir / f"step_{step}.pt"
    if not final_ckpt_path.exists():
        ckpt = _save_checkpoint(step)
        print(f"[{variant_name}] Final checkpoint → {ckpt}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print(f"\n[{variant_name}] Final test evaluation …")
    test_samples: List[dict] = []
    test_metrics         = evaluate(model, test_dl, device, max_batches=None)
    test_bleu            = evaluate_bleu(
        model, test_dl, tokenizer, device,
        max_new_tokens = cfg.get("max_new_tokens", 128),
        max_batches    = cfg.get("test_bleu_batches", 30),
        variant_name   = variant_name, tag = "TEST",
        n_print        = cfg.get("n_print_samples", 3), samples = test_samples,
    )
    test_metrics["test_bleu"] = test_bleu
    test_metrics["test_ppl"]  = test_metrics["val_ppl"]    # evaluate() names keys val_*; this is the test split
    test_metrics["test_loss"] = test_metrics["val_loss"]
    print(f"[{variant_name}] TEST: {test_metrics}")
    _wb(wandb.log, {f"test/{k}": v for k, v in test_metrics.items()}, step=step)
    _wb(wandb.run.summary.update, {"final/test_ppl": test_metrics["test_ppl"],
                                   "final/test_bleu": test_bleu, "final/tokens_seen": tokens_seen})
    if test_samples:
        _wb(wandb.log, {"test/samples": wandb.Table(
            columns=list(test_samples[0].keys()),
            data=[list(d.values()) for d in test_samples])}, step=step)
        with open(out_dir / "test_samples.json", "w") as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "step": step, "tokens_seen": tokens_seen}, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump({"variant": variant_name, "history": history, "test": test_metrics}, f, indent=2)

    # ── Push to HuggingFace Hub ───────────────────────────────────────────
    if cfg.get("push_to_hub", True):
        _push_to_hub(model, tokenizer, variant_name,
                     repo_id=cfg.get("hf_repo_id", "Bejaine/Mixture-of-Experts-Optimisers-and-Decoding-Strategies-aNLP"),
                     path_prefix="part1")

    _wb(wandb.finish)
    resume_path.unlink(missing_ok=True)   # done; step_<N>.pt is the final checkpoint
    return model, test_metrics


# ══════════════════════════════════════════════════════════════════════════
# HuggingFace Hub push
# ══════════════════════════════════════════════════════════════════════════

def _push_to_hub(model: Transformer, tokenizer, variant_name: str, repo_id: str,
                  path_prefix: str = "part1"):
    """
    Pushes to a subfolder of one shared repo (repo_id) rather than creating a
    separate repo per variant — this repo hosts checkpoints for all of
    parts 1-3 of the assignment, organised as "{path_prefix}/{variant_name}/".
    """
    try:
        from huggingface_hub import HfApi
        import tempfile, json

        api = HfApi()
        try:
            repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)
            # create_repo resolves a bare name to "<namespace>/<name>", but
            # upload_folder needs that fully-qualified id explicitly — passing
            # the original bare repo_id 404s on its preupload check.
            repo_id = repo_url.repo_id
        except Exception as e:
            print(f"[hub] create_repo failed: {e}")
            return

        path_in_repo = f"{path_prefix}/{variant_name}"
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(model.state_dict(), os.path.join(tmp, "model.pt"))
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(vars(model.config), f, indent=2)
            tokenizer.save(os.path.join(tmp, "tokenizer.json"))
            api.upload_folder(folder_path=tmp, repo_id=repo_id, path_in_repo=path_in_repo)
            print(f"[hub] → https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
    except Exception as e:
        print(f"[hub] Push failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# WandB login
# ══════════════════════════════════════════════════════════════════════════

def _wandb_login(host: str = "https://api.forge.coreweave.com"):
    # WANDB_BASE_URL also makes the later wandb.init() calls in this same
    # process target the Forge host, not just this login() call.
    os.environ["WANDB_BASE_URL"] = host
    key_file = "key.wandb"
    if os.path.exists(key_file):
        with open(key_file) as f:
            wandb.login(key=f.read().strip(), host=host)
    else:
        wandb.login(host=host)   # falls back to WANDB_API_KEY env var


# ══════════════════════════════════════════════════════════════════════════
# Public entrypoint
# ══════════════════════════════════════════════════════════════════════════

def run(config_path: str, variant_arg: str = "all"):
    """
    Called from main.py.
    variant_arg: "1"–"5", "all", or comma-separated list e.g. "1,3".
    """
    import random
    import numpy as np

    cfg  = _load_yaml(config_path)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    _wandb_login()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_path = cfg["tokenizer_save_path"]
    if os.path.exists(os.path.join(tok_path, "tokenizer.json")):
        print("[train] Loading cached tokenizer …")
        tokenizer = load_tokenizer(tok_path)
    else:
        from datasets import load_dataset as _ld
        raw       = _ld(cfg["dataset_name"], split="train")
        tokenizer = build_tokenizer(
            raw,
            vocab_size  = cfg["tokenizer_vocab_size"],
            save_path   = tok_path,
            max_samples = cfg.get("max_samples", 500_000),
        )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_dl, val_dl, test_dl = get_dataloaders(cfg, tokenizer)

    # ── Variants ──────────────────────────────────────────────────────────
    variants = [1, 2, 3, 4, 5] if variant_arg == "all" else [
        int(v.strip()) for v in variant_arg.split(",")
    ]

    all_results = {}
    for v in variants:
        done = list(Path(cfg["output_dir"]).glob(f"v{v}_*/test_metrics.json"))
        if done:
            print(f"[train] Variant {v} already finished ({done[0]}), skipping.")
            continue
        print(f"\n{'='*60}\n  Training Variant {v}\n{'='*60}\n")
        _, metrics      = train_variant(v, cfg, train_dl, val_dl, test_dl, tokenizer, device)
        all_results[v] = metrics

    print("\n[train] All variants complete.")
    for v, m in all_results.items():
        print(f"  V{v}: {m}")