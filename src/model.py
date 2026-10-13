"""
src/part1/model.py
──────────────────
Architecture closely follows the A1 codebase conventions:
  • Split W_q / W_k / W_v / W_o projections (matching A1 MultiHeadAttention)
  • RoPE positional encoding (RotaryPositionEmbedding, same class as A1)
  • RMSNorm or LayerNorm selectable via config (same classes as A1 norm.py)
  • F.scaled_dot_product_attention for the attention kernel (allowed per spec)
  • PAD_IDX=0, SOS_IDX=1, EOS_IDX=2, UNK_IDX=3 (matches A1 dataset.py)

Five FFN variants (enforced by build_variant):
  V1: dense 2-layer MLP (baseline)
  V2: MoE 4 experts, top-1    – total params ≈ V1
  V3: MoE 4 experts, top-2    – total params ≈ V1
  V4: 1 shared + 3 routed, top-1 routed (2 active of 4) – total params ≈ V1
  V5: MoE 4 experts, top-2    – active params ≈ V1 (larger total)
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Special token IDs (mirrors A1 dataset.py) ─────────────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ══════════════════════════════════════════════════════════════════════════
# Positional encodings  (ported from A1 positional.py)
# ══════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding (A1 style)."""

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE – identical to A1 positional.py.
    Applied inside attention to Q and K only; V is unmodified.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t        = torch.arange(max_seq_len, dtype=torch.float)
        freqs    = torch.einsum("i,j->ij", t, inv_freq)
        emb      = torch.cat((freqs, freqs), dim=-1)   # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, H, T, d_k)
        seq_len = q.size(2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_r = (q * cos) + (self._rotate_half(q) * sin)
        k_r = (k * cos) + (self._rotate_half(k) * sin)
        return q_r, k_r


# ══════════════════════════════════════════════════════════════════════════
# Norms  (ported from A1 norm.py)
# ══════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms    = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


def _make_norm(d_model: int, norm_type: str) -> nn.Module:
    if norm_type == "rms":
        return RMSNorm(d_model)
    return LayerNorm(d_model)


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TransformerConfig:
    # Core dimensions
    D_model:    int   = 256
    N_ctx:      int   = 256
    N_heads:    int   = 4
    N_layers:   int   = 6
    D_ff:       int   = 1024      # hidden dim of *dense* baseline MLP
    vocab_size: int   = 32000
    dropout:    float = 0.1

    # Positional / norm (A1-style choices)
    use_rope:   bool  = True       # True → RoPE; False → sinusoidal
    norm_type:  str   = "rms"      # "rms" or "layer"

    # FFN selector
    ffn_type:   str   = "mlp"      # "mlp" or "moe"

    # MoE-specific (ignored when ffn_type == "mlp")
    N_experts:   int  = 4
    N_active:    int  = 1
    N_shared:    int  = 0          # always-on experts (variant 4)
    D_ff_expert: int  = 0          # 0 → auto-set in __post_init__

    def __post_init__(self):
        assert self.D_model % self.N_heads == 0, "D_model must be divisible by N_heads"
        if self.D_ff_expert == 0:
            self.D_ff_expert = self.D_ff

    @property
    def D_head(self) -> int:
        return self.D_model // self.N_heads


# ══════════════════════════════════════════════════════════════════════════
# Attention  (A1 MultiHeadAttention style: split W_q/W_k/W_v/W_o)
# ══════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Mirrors A1 MultiHeadAttention: separate W_q, W_k, W_v, W_o projections.
    Uses F.scaled_dot_product_attention for the kernel (explicitly allowed).
    RoPE applied to Q and K when config.use_rope is True.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.N_heads
        self.d_k       = config.D_head
        self.d_model   = config.D_model
        self.use_rope  = config.use_rope

        # A1-style: separate projection matrices
        self.W_q = nn.Linear(config.D_model, config.D_model, bias=False)
        self.W_k = nn.Linear(config.D_model, config.D_model, bias=False)
        self.W_v = nn.Linear(config.D_model, config.D_model, bias=False)
        self.W_o = nn.Linear(config.D_model, config.D_model, bias=False)

        self.drop = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rope = RotaryPositionEmbedding(self.d_k, max_seq_len=config.N_ctx + 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, D_model)
        padding_mask : (B, T) bool, True = real token, False = pad.
                       When provided, combined with the causal mask so that
                       padding positions are never attended to.
        """
        B, T, D = x.shape

        # Project and reshape to (B, H, T, d_k) – A1 style split projections
        def proj_and_reshape(proj):
            return proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q = proj_and_reshape(self.W_q)
        K = proj_and_reshape(self.W_k)
        V = proj_and_reshape(self.W_v)

        if self.use_rope:
            Q, K = self.rope(Q, K)

        # Build attention mask combining causal + padding (fix #8)
        # SDPA expects attn_mask where True = attend, False = ignore.
        # causal: (1, 1, T, T) lower-triangular
        causal = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)                              # (1, 1, T, T)

        if padding_mask is not None:
            # padding_mask: (B, T) → (B, 1, 1, T) to broadcast over queries
            pad = padding_mask.unsqueeze(1).unsqueeze(2)         # (B, 1, 1, T)
            attn_mask = causal & pad                             # (B, 1, T, T)
        else:
            attn_mask = causal                                   # (1, 1, T, T)

        # Convert bool mask to additive float mask for SDPA
        # SDPA accepts bool directly on PyTorch >= 2.0: True = keep, False = -inf
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = attn_mask,
            dropout_p = self.drop.p if self.training else 0.0,
        )   # (B, H, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════════
# MLP  (dense FFN baseline)
# ══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Standard 2-layer feed-forward with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ══════════════════════════════════════════════════════════════════════════
# Router  (top-k gating)
# ══════════════════════════════════════════════════════════════════════════

class Router(nn.Module):
    """
    Linear gating → softmax → top-k selection.

    Returns
    -------
    topk_weights  : (B, T, N_active)   renormalised router scores
    topk_indices  : (B, T, N_active)   which experts (LongTensor)
    router_probs  : (B, T, N_pool)     full softmax (for aux loss)
    """

    def __init__(self, d_model: int, n_experts: int, n_active: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_active  = n_active
        self.gate      = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits       = self.gate(x)                          # (B, T, N_experts)
        router_probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(router_probs, self.n_active, dim=-1)
        # Re-normalise so selected weights sum to 1
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_weights, topk_idx, router_probs


# ══════════════════════════════════════════════════════════════════════════
# MoE  (variants 2–5)
# ══════════════════════════════════════════════════════════════════════════

class MoE(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Variant 4 (N_shared=1, N_experts=3, N_active=2):
        1 always-on shared expert + top-1 from 3 routed experts.
        The shared expert is stored separately in self.shared_experts;
        self.experts holds the routed pool only.

    Expert counts are accumulated in self.expert_counts (a buffer) so
    analyse.py can build the language-specialisation heatmap.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.N_experts = config.N_experts    # size of routed pool
        self.N_active  = config.N_active     # total active (shared + routed)
        self.N_shared  = config.N_shared
        self.N_routed  = config.N_active - config.N_shared

        # Always-on experts
        self.shared_experts = nn.ModuleList([
            MLP(config.D_model, config.D_ff_expert, config.dropout)
            for _ in range(self.N_shared)
        ])

        # Routed expert pool
        self.experts = nn.ModuleList([
            MLP(config.D_model, config.D_ff_expert, config.dropout)
            for _ in range(self.N_experts)
        ])

        # Router operates over the routed pool only
        self.router = Router(config.D_model, self.N_experts, self.N_routed)

        # Usage tracking buffer – shape (N_experts,), counts token dispatches
        # Index i corresponds to self.experts[i] (routed pool only).
        self.register_buffer(
            "expert_counts", torch.zeros(self.N_experts, dtype=torch.long)
        )

    def reset_counts(self):
        self.expert_counts.zero_()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (B, T, D_model)
        padding_mask : (B, T) bool, True = real token.
                       Used to exclude padding positions from expert_counts
                       and from the aux loss computation. (fix #4)

        Returns
        -------
        out      : (B, T, D_model)
        aux_loss : scalar – Switch-Transformer load-balancing loss
                   aux_alpha * N * sum_i(f_i * P_i)  (computed over real tokens only)
        """
        B, T, D = x.shape

        # ── Shared (always-on) experts ────────────────────────────────────
        out = torch.zeros_like(x)
        for expert in self.shared_experts:
            out = out + expert(x)

        # ── Routed experts ────────────────────────────────────────────────
        aux_loss = torch.tensor(0.0, device=x.device)

        if self.N_routed > 0:
            weights, indices, router_probs = self.router(x)
            # weights      : (B, T, N_routed)
            # indices      : (B, T, N_routed)  into self.experts
            # router_probs : (B, T, N_experts) full softmax

            # Dispatch + combine without host syncs: scatter each token's
            # renormalised top-k weights into a dense (B, T, N_experts) gate
            # (zero for unselected experts), then combine every expert's
            # output with its gate. Same result as the per-expert masked
            # accumulation used in run01 (each expert was already evaluated
            # densely there); the .any()/boolean-index checks it used forced a
            # GPU->CPU sync per expert per layer and made MoE ~2.5x slower.
            gates = torch.zeros(
                B, T, self.N_experts, device=x.device, dtype=weights.dtype
            ).scatter_add_(-1, indices, weights)                  # (B, T, N_experts)
            for e in range(self.N_experts):
                out = out + gates[..., e:e + 1] * self.experts[e](x)

            # Per-expert dispatch counts (real tokens only) and the
            # Switch-Transformer load-balancing terms f_i (fraction of routed
            # slots sent to expert i) and P_i (mean router prob of expert i).
            N        = self.N_experts
            real     = (padding_mask if padding_mask is not None
                        else torch.ones(B, T, dtype=torch.bool, device=x.device))
            realf    = real.unsqueeze(-1).to(router_probs.dtype)   # (B, T, 1)
            onehot   = F.one_hot(indices, N).sum(dim=2)             # (B, T, N)
            dispatch = (onehot * real.unsqueeze(-1)).sum(dim=(0, 1))  # (N,)
            with torch.no_grad():
                self.expert_counts += dispatch.long()
            n_real = realf.sum().clamp(min=1.0)
            f = dispatch.to(router_probs.dtype) / (n_real * self.N_routed)
            P = (router_probs * realf).sum(dim=(0, 1)) / n_real

            aux_loss = N * (f * P).sum()

        return out, aux_loss


# ══════════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block: Norm → Attn → residual; Norm → FFN → residual.
    Norm type and FFN type are controlled by config.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = _make_norm(config.D_model, config.norm_type)
        self.norm2 = _make_norm(config.D_model, config.norm_type)
        self.attn  = CausalSelfAttention(config)
        self.drop  = nn.Dropout(config.dropout)

        if config.ffn_type == "mlp":
            self.ffn     = MLP(config.D_model, config.D_ff, config.dropout)
            self.use_moe = False
        elif config.ffn_type == "moe":
            self.ffn     = MoE(config)
            self.use_moe = True
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (B, T, D_model)
        padding_mask : (B, T) bool – passed to attention and MoE (#4, #8)
        """
        # Attention sub-layer (pre-norm, residual)
        x = x + self.drop(self.attn(self.norm1(x), padding_mask=padding_mask))

        # FFN sub-layer (pre-norm, residual)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x), padding_mask=padding_mask)
            x = x + self.drop(ffn_out)
        else:
            x = x + self.drop(self.ffn(self.norm2(x)))
            aux_loss = torch.tensor(0.0, device=x.device)

        return x, aux_loss


# ══════════════════════════════════════════════════════════════════════════
# Decoder-only Transformer LM
# ══════════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Decoder-only causal language model.
    When use_rope=True, positional information comes from RoPE inside
    attention; no separate positional embedding is added to token embeddings.
    When use_rope=False, sinusoidal encoding is added to embeddings.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.D_model,
                                      padding_idx=PAD_IDX)
        # Positional encoding: sinusoidal when not using RoPE
        self.pos_enc: Optional[SinusoidalPositionalEncoding] = (
            None if config.use_rope
            else SinusoidalPositionalEncoding(config.D_model, config.N_ctx + 1)
        )

        self.drop   = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.N_layers)]
        )
        self.norm_final = _make_norm(config.D_model, config.norm_type)
        self.lm_head    = nn.Linear(config.D_model, config.vocab_size, bias=False)

        # Weight tying: token embedding ↔ LM head
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : (B, T)

        Returns
        -------
        logits   : (B, T, vocab_size)
        aux_loss : scalar  (0 for dense variant)
        """
        B, T = input_ids.shape
        assert T <= self.config.N_ctx, \
            f"Sequence length {T} exceeds N_ctx={self.config.N_ctx}"

        # Padding mask: True = real token, False = pad (fix #4, #8)
        padding_mask = (input_ids != PAD_IDX)        # (B, T) bool

        x = self.drop(self.token_emb(input_ids))    # (B, T, D)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        total_aux = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux = block(x, padding_mask=padding_mask)
            total_aux = total_aux + aux

        x      = self.norm_final(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)
        return logits, total_aux

    # ── Expert-usage helpers (for analyse.py) ─────────────────────────────

    @torch.no_grad()
    def get_expert_usage(self) -> Dict[str, torch.Tensor]:
        """Returns {layer_idx_str: count_tensor(N_experts)} for MoE layers."""
        return {
            str(i): block.ffn.expert_counts.clone().cpu()
            for i, block in enumerate(self.blocks)
            if block.use_moe
        }

    def reset_expert_counts(self):
        for block in self.blocks:
            if block.use_moe:
                block.ffn.reset_counts()

    # ── Parameter counting ────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total  = sum(p.numel() for p in self.parameters())
        active = self._count_active_params()
        return {"total": total, "active": active}

    def _count_active_params(self) -> int:
        # Embeddings, final norm, LM head are always active.
        # lm_head.weight is tied to token_emb.weight (see __init__), so count
        # it once — summing both .parameters() separately double-counts the
        # same tensor (this previously made active > total).
        active = (
            sum(p.numel() for p in self.token_emb.parameters())
            + sum(p.numel() for p in self.norm_final.parameters())
        )
        if self.lm_head.weight is not self.token_emb.weight:
            active += sum(p.numel() for p in self.lm_head.parameters())
        if self.pos_enc is not None:
            active += sum(p.numel() for p in self.pos_enc.parameters())

        for block in self.blocks:
            active += (
                sum(p.numel() for p in block.norm1.parameters())
                + sum(p.numel() for p in block.norm2.parameters())
                + sum(p.numel() for p in block.attn.parameters())
            )
            if block.use_moe:
                moe: MoE = block.ffn
                active += sum(p.numel() for p in moe.router.parameters())
                for se in moe.shared_experts:
                    active += sum(p.numel() for p in se.parameters())
                # Only N_routed active experts
                if moe.N_experts > 0:
                    active += moe.N_routed * sum(
                        p.numel() for p in moe.experts[0].parameters()
                    )
            else:
                active += sum(p.numel() for p in block.ffn.parameters())

        return active


# ══════════════════════════════════════════════════════════════════════════
# Factory: build_variant
# ══════════════════════════════════════════════════════════════════════════

def _expert_d_ff(baseline_d_ff: int, d_model: int, n_total_experts: int) -> int:
    """
    Scale expert hidden dim so that
        n_total_experts * (2 * d_model * d_ff_expert) ≈ 2 * d_model * baseline_d_ff
    → d_ff_expert = baseline_d_ff // n_total_experts
    Minimum of 64 to stay meaningful.
    """
    return max(baseline_d_ff // n_total_experts, 64)


def build_variant(
    variant: int,
    base_cfg: TransformerConfig,
) -> Tuple[Transformer, str]:
    """
    Construct one of the five FFN variants.

    Parameters
    ----------
    variant  : 1–5
    base_cfg : TransformerConfig with D_ff, D_model, etc. set; ffn_type ignored.

    Returns
    -------
    (model, variant_name_string)
    """
    cfg = deepcopy(base_cfg)

    if variant == 1:
        cfg.ffn_type = "mlp"
        name = "v1_dense_mlp"

    elif variant == 2:
        # Total params ≈ baseline: 4 experts, each with d_ff/4 hidden dim
        cfg.ffn_type    = "moe"
        cfg.N_experts   = 4
        cfg.N_active    = 1
        cfg.N_shared    = 0
        cfg.D_ff_expert = _expert_d_ff(cfg.D_ff, cfg.D_model, n_total_experts=4)
        name = "v2_moe_4e_top1"

    elif variant == 3:
        # Total params ≈ baseline: 4 experts, each with d_ff/4 hidden dim
        cfg.ffn_type    = "moe"
        cfg.N_experts   = 4
        cfg.N_active    = 2
        cfg.N_shared    = 0
        cfg.D_ff_expert = _expert_d_ff(cfg.D_ff, cfg.D_model, n_total_experts=4)
        name = "v3_moe_4e_top2"

    elif variant == 4:
        # 1 shared + 3 routed (top-1 routed → 2 active of 4 total)
        # Total params ≈ baseline: 4 experts total, each with d_ff/4
        cfg.ffn_type    = "moe"
        cfg.N_experts   = 3          # routed pool
        cfg.N_active    = 2          # 1 shared + 1 routed
        cfg.N_shared    = 1
        cfg.D_ff_expert = _expert_d_ff(cfg.D_ff, cfg.D_model, n_total_experts=4)
        name = "v4_moe_1shared_3routed_top1"

    elif variant == 5:
        # Active params ≈ baseline: 2 active experts, each with D_ff//2 hidden dim.
        # Active FFN capacity = 2 × (D_ff//2) = D_ff, matching the dense MLP.
        # Total params are larger (4 experts × D_ff//2 each = 2× dense total FFN).
        cfg.ffn_type    = "moe"
        cfg.N_experts   = 4
        cfg.N_active    = 2
        cfg.N_shared    = 0
        cfg.D_ff_expert = cfg.D_ff // 2   # fix #3: active parity, not total parity
        name = "v5_moe_4e_top2_active_parity"

    else:
        raise ValueError(f"variant must be 1–5, got {variant}")

    model = Transformer(cfg)
    pc    = model.count_parameters()
    print(
        f"[model] Variant {variant} ({name}): "
        f"total={pc['total']:,}  active≈{pc['active']:,}"
    )
    return model, name