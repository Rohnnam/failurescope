"""
FailureScope Layer 3 — Transformer Model
==========================================
Upgraded to 6-layer, 8-head, d_model=256 for richer attention patterns.

Per context1 [Layer 2: Transformer Learning]: minimum 4 layers, 8 heads.
This exceeds the minimum to produce richer per-head specialisation —
different heads learn to track different physical variables (voltage,
altitude, pitch) which is essential for the attention extractor to find
induced relationships rather than trivial terminal-state patterns.

Architecture:
  1. Linear input projection: FEATURE_DIM (37) → d_model (256)
  2. Learnable CLS token prepended to sequence
  3. Sinusoidal positional encoding (up to MAX_SEQ_LEN + 1 positions)
  4. 6× TransformerEncoderLayer (d_model=256, nhead=8, ffn=1024)
  5. Classification head on CLS output: d_model → 128 → 2 (binary)

VRAM budget (RTX 4060 8GB, batch=64, seq=1200, fp16, grad_ckpt):
  Attention (1 layer recomputed): ~1.48 GB
  Activations + optimizer:        ~0.29 GB
  Peak estimated:                  ~1.76 GB  ← safe

Two forward modes:
  - Training (return_attention=False):
      Gradient checkpointing per layer. use_reentrant=False (torch ≥ 2.0).
  - Inference / extraction (return_attention=True):
      Full per-layer, per-head attention weights returned.
      Used by attention_extractor.py for induced relationship discovery.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dataset import FEATURE_DIM, MAX_SEQ_LEN


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    max_len = MAX_SEQ_LEN + 1 to accommodate the prepended CLS token.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = MAX_SEQ_LEN + 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


# ── Custom Encoder Layer (with extractable attention) ─────────────────────────

class _EncoderLayer(nn.Module):
    """
    Thin wrapper around nn.TransformerEncoderLayer that exposes attention weights
    when return_attention=True without interfering with gradient checkpointing
    during training.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        # Keep a direct reference to self-attn for weight extraction
        self.self_attn = self.layer.self_attn

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not return_attention:
            out = self.layer(x, src_key_padding_mask=src_key_padding_mask)
            return out, None

        # ── Manual forward to expose attention weights ─────────────────────────
        # Replicates nn.TransformerEncoderLayer (norm_first=False) exactly:
        #   x = norm1(x + dropout(self_attn(x, x, x)[0]))
        #   x = norm2(x + dropout(ff(x)))

        l = self.layer

        # Self-attention with weight extraction
        attn_out, attn_weights = l.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep all heads: (B, nhead, S, S)
        )
        x = l.norm1(x + l.dropout(attn_out))

        # Feed-forward
        ff_out = l.linear2(l.dropout(l.activation(l.linear1(x))))
        x = l.norm2(x + l.dropout(ff_out))

        return x, attn_weights  # attn_weights: (B, nhead, S+1, S+1)


# ── Main Model ─────────────────────────────────────────────────────────────────

class FailureScopeTransformer(nn.Module):
    """
    Binary classifier: predicts catastrophic (1) vs recoverable (0) from
    a padded trajectory sequence.

    Args:
        feature_dim  : input features per timestep (default 37)
        d_model      : transformer hidden dim (default 128)
        nhead        : attention heads (default 8, per context1)
        num_layers   : encoder depth (default 4, per context1)
        dim_feedforward : FFN width (default 512)
        dropout      : dropout probability (default 0.1)
        max_seq_len  : maximum sequence length before CLS (default 1200)
        use_gradient_checkpointing : enable during training for VRAM savings

    Forward inputs:
        x           : (B, seq, feature_dim) — padded feature sequences
        pad_mask    : (B, seq) bool — True = real step, False = padding
        return_attention : if True, also return list of per-layer attn weights

    Forward outputs (return_attention=False):
        logits : (B, 2) — raw unnormalised class scores
    Forward outputs (return_attention=True):
        logits : (B, 2)
        attn_weights : list of (B, nhead, seq+1, seq+1) per layer
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ── Input projection ───────────────────────────────────────────────────
        self.input_proj = nn.Linear(feature_dim, d_model)

        # ── Learnable CLS token ────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding ────────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len + 1,
            dropout=dropout,
        )

        # ── Encoder layers ─────────────────────────────────────────────────────
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Classification head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        # ── Weight initialisation ──────────────────────────────────────────────
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        B, S, _ = x.shape

        # Project and prepend CLS
        x = self.input_proj(x)                            # (B, S, d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, S+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Extend padding mask: CLS token is always valid (not padded)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_valid_mask = torch.cat([cls_valid, pad_mask], dim=1)  # (B, S+1)

        # nn.TransformerEncoderLayer convention: key_padding_mask True = IGNORE
        # Our pad_mask convention:              True = REAL  → invert
        src_key_padding_mask = ~full_valid_mask  # (B, S+1)

        attn_list: List[torch.Tensor] = []
        use_ckpt = self.use_gradient_checkpointing and self.training and not return_attention

        for enc_layer in self.encoder_layers:
            if use_ckpt:
                # Gradient checkpointing: wrap layer call so intermediates
                # are not stored.  use_reentrant=False: required for torch ≥ 2.0.
                def make_fn(layer, mask):
                    def fn(inp):
                        out, _ = layer(inp, mask, return_attention=False)
                        return out
                    return fn

                x = checkpoint(
                    make_fn(enc_layer, src_key_padding_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x, attn_w = enc_layer(
                    x,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                )
                if return_attention and attn_w is not None:
                    attn_list.append(attn_w.detach().cpu())

        cls_out = x[:, 0, :]          # (B, d_model) — CLS representation
        logits  = self.classifier(cls_out)  # (B, 2)

        if return_attention:
            return logits, attn_list
        return logits

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)