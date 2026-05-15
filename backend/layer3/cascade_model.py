"""
FailureScope Layer 3 — Cascade Window Transformer
===================================================
Transformer classifier trained on CASCADE WINDOWS ONLY.
Key difference from the previous model:
  - Input: states from inject_t onward (not full trajectory)
  - Features: 13 STATE_KEYS only — NO failure one-hot
  - Context token: 27-dim hole-alignment features prepended before sequence
  - Forces model to learn from state variable CHANGES, not failure identity

Why this produces genuine induced relationships:
  - Every input timestep has changing physics (v dropping, pump_eff degrading)
  - Model cannot use failure type as a shortcut (removed from sequence)
  - Attention will concentrate on the STATE COMBINATIONS that precede crashes
  - Kill zones will show actual degradation ranges (v: 10.5–13.5V, etc.)

Architecture:
  1. Context projection: 27 → d_model (hole-alignment conditions)
  2. State projection:   13 → d_model per timestep
  3. Concatenate: [context_token, state_t0, state_t1, ..., state_tN]
  4. Positional encoding
  5. 4× TransformerEncoderLayer (d_model=128, nhead=8, ffn=512)
  6. Classification from context token output (position 0)

Per context1 [Layer 2: Transformer Learning]: 4 layers, 8 heads minimum.
d_model=128 is safe for batch=128 on RTX 4060 8GB with cascade windows
(cascade_max_steps=400, attention matrix = 401² × 128 bytes = 20MB per batch).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .constants import CASCADE_CONTEXT_DIM, CASCADE_FEATURE_DIM, CASCADE_MAX_STEPS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = CASCADE_MAX_STEPS + 1, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    """Thin wrapper exposing attention weights for extraction."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.self_attn = self.layer.self_attn

    def forward(
        self, x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not return_attention:
            return self.layer(x, src_key_padding_mask=src_key_padding_mask), None

        l = self.layer
        attn_out, attn_w = l.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = l.norm1(x + l.dropout(attn_out))
        x = l.norm2(x + l.dropout(l.linear2(l.dropout(l.activation(l.linear1(x))))))
        return x, attn_w


class CascadeTransformer(nn.Module):
    """
    Classifies cascade windows as catastrophic (1) or recoverable (0).

    Inputs:
      seq      : (B, cascade_len, CASCADE_FEATURE_DIM=13)  cascade window states
      context  : (B, CASCADE_CONTEXT_DIM=27)               hole-alignment features
      pad_mask : (B, cascade_len)  True = real cascade step

    Forward outputs (return_attention=False):
      logits : (B, 2)

    Forward outputs (return_attention=True):
      logits      : (B, 2)
      attn_list   : list of (B, nhead, seq+1, seq+1) per layer — context token at pos 0
    """

    def __init__(
        self,
        feature_dim:  int   = CASCADE_FEATURE_DIM,
        context_dim:  int   = CASCADE_CONTEXT_DIM,
        d_model:      int   = 128,
        nhead:        int   = 8,
        num_layers:   int   = 4,
        dim_feedforward: int = 512,
        dropout:      float = 0.1,
        max_steps:    int   = CASCADE_MAX_STEPS,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # State sequence projection (13 → d_model)
        self.state_proj   = nn.Linear(feature_dim, d_model)

        # Context token projection (27 → d_model)
        # This encodes the hole-alignment conditions (altitude, fatigue, failure type)
        self.context_proj = nn.Linear(context_dim, d_model)

        # Positional encoding (max_steps + 1 for context token at position 0)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_steps + 1, dropout=dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Classification head on context token output
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # Weight init
        for m in [self.state_proj, self.context_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        seq:      torch.Tensor,
        context:  torch.Tensor,
        pad_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        B, S, _ = seq.shape

        # Project state sequence
        x_seq = self.state_proj(seq)                          # (B, S, d_model)

        # Project context → single token
        ctx_token = self.context_proj(context).unsqueeze(1)   # (B, 1, d_model)

        # Prepend context token
        x = torch.cat([ctx_token, x_seq], dim=1)              # (B, S+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Padding mask: context token always valid
        ctx_valid    = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_valid   = torch.cat([ctx_valid, pad_mask], dim=1)  # (B, S+1)
        src_kp_mask  = ~full_valid                               # nn convention: True = ignore

        # Transformer layers
        attn_list: List[torch.Tensor] = []
        use_ckpt = self.use_gradient_checkpointing and self.training and not return_attention

        for enc_layer in self.encoder_layers:
            if use_ckpt:
                def make_fn(layer, mask):
                    def fn(inp):
                        out, _ = layer(inp, mask, return_attention=False)
                        return out
                    return fn
                x = checkpoint(make_fn(enc_layer, src_kp_mask), x, use_reentrant=False)
            else:
                x, attn_w = enc_layer(x, src_kp_mask, return_attention=return_attention)
                if return_attention and attn_w is not None:
                    attn_list.append(attn_w.detach().cpu())

        # Classify from context token output (position 0)
        ctx_out = x[:, 0, :]                  # (B, d_model)
        logits  = self.classifier(ctx_out)    # (B, 2)

        if return_attention:
            return logits, attn_list
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)