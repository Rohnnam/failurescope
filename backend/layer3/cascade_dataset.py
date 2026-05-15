"""
FailureScope Layer 3 — Cascade Window Dataset
===============================================
Extracts and streams CASCADE WINDOWS for Transformer training.

WINDOW SELECTION STRATEGY (updated for Option B):

  Short cascade (cascade_dur <= CASCADE_MAX_STEPS):
    Window = states[inject_idx : inject_idx + cascade_dur]
    (Identical to old behaviour. All physics happens in this window.)

  Long cascade (cascade_dur > CASCADE_MAX_STEPS, catastrophic only):
    Window = states[ttc_idx - CASCADE_MAX_STEPS : ttc_idx]
    (Pre-criticality window. Captures the actual degradation — voltage
     dropping through R003/R004 thresholds, pump_eff declining, RPM
     decaying — rather than the flat stable period right after injection.)

  Recoverable trajectories:
    Window = states[inject_idx : inject_idx + CASCADE_MAX_STEPS]
    (No TTC to anchor to. Captures the period where the pilot successfully
     managed the failure — important negative examples for the Transformer.)

Rationale:
  For electrical failures (alternator, battery), cascade duration is often
  1600–2000s. With DT=1.0s and CASCADE_MAX_STEPS=400, the inject-forward
  window covers the first 400s where voltage is still ~11.8V (above both
  R003=11.5V and R004=10.5V). No state variables change. The Transformer
  sees flat signals and learns nothing about the cascade.

  The pre-TTC window captures the 400s before criticality: voltage decaying
  from ~11.5V through 10.5V, pump_eff declining 0.8→0.0, RPM dropping
  2400→0. These are the dynamics that distinguish recoverable from
  catastrophic outcomes.

Feature vector (CASCADE_FEATURE_DIM = 13):
  13 STATE_KEYS only, normalised to [0,1] using STATE_SCALES
  ← No failure one-hot. Failure type provided as context token separately.

Context token (separate, not broadcast across sequence):
  [alt_norm, fatigue, skill_norm, weather_norm, failure_one_hot(23)]
  = 27 features describing "hole alignment" at failure injection
  Prepended as a learnable position-0 context embedding.

Shard format:
  Each .pt file contains CASCADE_SHARD_SIZE trajectories:
    sequences:   (N, max_len_in_shard, 13)  fp16 — cascade window states
    lengths:     (N,)                        int16 — actual cascade steps
    context:     (N, 27)                     fp16  — context token features
    labels:      (N,)                        int8  — 1=catastrophic, 0=recoverable

Filter: cascade_duration >= MIN_CASCADE_DURATION_S (default 5s)
"""

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .constants import (
    CASCADE_FEATURE_DIM, CASCADE_MAX_STEPS, FAILURE_TO_IDX,
    MIN_CASCADE_DURATION_S, N_FAILURE_COMPONENTS, N_STATE_KEYS,
    SKILL_TO_IDX, STATE_KEYS, STATE_SCALES, WEATHER_TO_IDX,
)

CASCADE_SHARD_DIR = Path("/opt/failurescope_data/cascade_shards")

_SCALE = torch.tensor(STATE_SCALES[:N_STATE_KEYS], dtype=torch.float32)


def extract_context_vector(record: dict, inject_state: Optional[dict]) -> torch.Tensor:
    """
    Build the 27-dim context token from init dict + state at inject_t.
    This encodes the "hole alignment" — conditions at the moment of failure.
    """
    init     = record.get("init", {})
    fails    = record.get("failures", [])
    comp     = fails[0].get("component", "") if fails else ""
    fail_idx = FAILURE_TO_IDX.get(comp, -1)

    alt_norm     = float(inject_state.get("alt", 5000) if inject_state else init.get("altitude_ft", 5000)) / 16500.0
    fatigue      = float(init.get("fatigue", 0.5))
    skill_idx    = SKILL_TO_IDX.get(str(init.get("skill", "commercial")), 2)
    skill_norm   = skill_idx / 4.0
    weather_idx  = WEATHER_TO_IDX.get(str(init.get("weather", "vmc")), 0)
    weather_norm = weather_idx / 3.0

    failure_vec = torch.zeros(N_FAILURE_COMPONENTS, dtype=torch.float32)
    if 0 <= fail_idx < N_FAILURE_COMPONENTS:
        failure_vec[fail_idx] = 1.0

    context = torch.cat([
        torch.tensor([alt_norm, fatigue, skill_norm, weather_norm], dtype=torch.float32),
        failure_vec,
    ])  # (27,)
    return context


def extract_cascade_sequence(
    states: list,
    inject_t: float,
    ttc_s: float,
    max_steps: int = CASCADE_MAX_STEPS,
    is_catastrophic: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Convert stored states to tensor. Windowing is done in batch_runner.py.

    For long electrical cascades, batch_runner stores states subsampled at
    5s intervals from the R003 threshold crossing. For short/non-electrical
    cascades, states are at 0.1s intervals from inject_idx.

    This function simply converts stored states to a normalised tensor.
    inject_t/ttc_s/is_catastrophic kept for API compatibility.
    """
    if not states:
        return torch.zeros(1, N_STATE_KEYS, dtype=torch.float16), 1

    actual_len = min(len(states), max_steps)

    seq = torch.zeros(actual_len, N_STATE_KEYS, dtype=torch.float16)
    for i in range(actual_len):
        s    = states[i]
        vals = torch.tensor(
            [float(s.get(k, 0.0)) for k in STATE_KEYS],
            dtype=torch.float32,
        )
        seq[i] = (vals / _SCALE).clamp(-5.0, 5.0).half()

    return seq, actual_len

# ── Streaming Dataset ──────────────────────────────────────────────────────────

class CascadeShardDataset(IterableDataset):
    """
    Streams pre-built cascade window shards.
    Each worker handles a disjoint subset of shards.

    Yields: (seq, context, pad_mask, label)
      seq      : (CASCADE_MAX_STEPS, CASCADE_FEATURE_DIM) float32
      context  : (CASCADE_CONTEXT_DIM,) float32
      pad_mask : (CASCADE_MAX_STEPS,) bool — True = real cascade step
      label    : () int64 — 1=catastrophic, 0=recoverable
    """

    def __init__(
        self,
        shard_dir: Path = CASCADE_SHARD_DIR,
        split: str = "train",
        shuffle: bool = True,
        max_steps: int = CASCADE_MAX_STEPS,
    ) -> None:
        assert split in ("train", "val")
        self.split_dir = shard_dir / split
        self.shuffle   = shuffle and (split == "train")
        self.max_steps = max_steps

        shards = sorted(self.split_dir.glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(
                f"No cascade shards found in {self.split_dir}.\n"
                "Run: python preprocess_cascade.py"
            )
        self.shards = shards

    def __iter__(self) -> Iterator:
        worker    = get_worker_info()
        wid       = worker.id          if worker else 0
        n_workers = worker.num_workers if worker else 1

        my_shards = self.shards[wid::n_workers]
        if self.shuffle:
            random.shuffle(my_shards)

        for shard_path in my_shards:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            yield from self._iter_shard(shard)

    def _iter_shard(self, shard: dict) -> Iterator:
        sequences = shard["sequences"]   # (N, shard_max_len, 13) fp16
        lengths   = shard["lengths"]     # (N,) int16
        contexts  = shard["context"]     # (N, 27) fp16
        labels    = shard["labels"]      # (N,) int8

        N       = labels.shape[0]
        indices = list(range(N))
        if self.shuffle:
            random.shuffle(indices)

        for i in indices:
            actual_len = min(int(lengths[i].item()), self.max_steps)
            if actual_len == 0:
                actual_len = 1

            seq = sequences[i, :actual_len].float()   # (actual_len, 13)

            # Pad to max_steps
            pad_len = self.max_steps - actual_len
            if pad_len > 0:
                seq = torch.nn.functional.pad(seq, (0, 0, 0, pad_len))

            context  = contexts[i].float()
            pad_mask = torch.zeros(self.max_steps, dtype=torch.bool)
            pad_mask[:actual_len] = True
            label    = torch.tensor(int(labels[i].item()), dtype=torch.long)

            yield seq, context, pad_mask, label


def cascade_collate_fn(batch: list) -> tuple:
    """
    Dynamic collation — pads to BATCH maximum cascade length, not global max.
    """
    import torch.nn.functional as F

    seqs, contexts, masks, labels = zip(*batch)
    labels   = torch.stack(labels)
    contexts = torch.stack(contexts)

    real_lens = [int(m.sum().item()) for m in masks]
    batch_max = max(real_lens)

    padded_seqs  = []
    padded_masks = []
    for seq, mask, rlen in zip(seqs, masks, real_lens):
        s = seq[:rlen]
        m = mask[:rlen]
        pad = batch_max - rlen
        if pad > 0:
            s = F.pad(s, (0, 0, 0, pad))
            m = F.pad(m, (0, pad), value=False)
        padded_seqs.append(s)
        padded_masks.append(m)

    return (
        torch.stack(padded_seqs),    # (B, batch_max, 13)
        contexts,                     # (B, 27)
        torch.stack(padded_masks),   # (B, batch_max)
        labels,                       # (B,)
    )


def shards_exist(shard_dir: Path = CASCADE_SHARD_DIR) -> bool:
    return (
        (shard_dir / "train").exists()
        and len(list((shard_dir / "train").glob("shard_*.pt"))) > 0
        and (shard_dir / "val").exists()
        and len(list((shard_dir / "val").glob("shard_*.pt"))) > 0
    )