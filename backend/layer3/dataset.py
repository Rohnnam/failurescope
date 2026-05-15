"""
FailureScope Layer 3 — Streaming Dataset
=========================================
Streams trajectories.jsonl.gz without loading into RAM.

Trajectory schema (per PROGRESS.md §12):
  Each line: JSON with keys scenario_id, aircraft, init (nested), failures (list),
  outcome (top-level — THIS is the label), ttc_s, recovery_s, failure_chain,
  reason, duration_s, steps, states (list of dicts).

State dict keys (abbreviated):
  t, alt, alt_agl, spd, pitch, roll, yaw, vs, rpm, ff,
  fuel_l, fuel_r, v, bat, pump_eff, outcome (per-step — IGNORED),
  failures (per-step active failures list — IGNORED for label)

Label: record["outcome"] == "catastrophic" → 1, else → 0

Design notes:
- MAX_SEQ_LEN = 1200 (option B from session 3 kickoff — covers all but
  extreme edge cases; truncated to actual outcome time for most trajectories)
- FEATURE_DIM = 37 = 13 numeric state keys + 1 fatigue scalar + 23 failure one-hot
- Multi-worker IterableDataset: each worker handles a disjoint line slice
  to avoid duplicate batches (standard pattern for streaming .gz files)
- Normalisation uses known physical ranges (not learned stats) so it works
  at inference time without a separate normalisation pass
"""

import gzip
import json
import math
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SEQ_LEN: int = 1200

# 13 numeric keys extracted from each state dict (per trajectory schema above)
STATE_KEYS = [
    "alt",      # altitude MSL (ft)
    "spd",      # airspeed (knots)
    "v",        # electrical voltage (V)
    "rpm",      # engine RPM
    "ff",       # fuel flow (GPH)
    "fuel_l",   # left tank fuel (gal)
    "fuel_r",   # right tank fuel (gal)
    "bat",      # battery charge (Ah)
    "pump_eff", # fuel pump efficiency (0–1)
    "pitch",    # pitch angle (degrees)
    "roll",     # roll angle (degrees)
    "vs",       # vertical speed (FPM)
    "t",        # time (seconds)
]

# Physical normalisation denominators — known ranges, not learned statistics.
# This makes the normalisation stable at inference without a separate stats pass.
STATE_SCALES = [
    16500.0,   # alt      — C182 service ceiling per PROGRESS.md §6
    200.0,     # spd      — reasonable upper bound for light GA
    32.0,      # v        — above any realistic aircraft voltage
    3000.0,    # rpm      — C182 red-line ~2400 RPM, leave headroom
    15.0,      # ff       — max fuel flow for C182 (GPH)
    50.0,      # fuel_l   — tank capacity upper bound (gal)
    50.0,      # fuel_r
    33.0,      # bat      — 33 Ah per PROGRESS.md §4 (Service Manual D2006R4-13)
    1.0,       # pump_eff — already 0–1
    45.0,      # pitch    — ±45° covers all non-inverted attitudes
    90.0,      # roll     — ±90°
    2000.0,    # vs       — FPM, sign-preserving; covers climb and steep dive
    3600.0,    # t        — MAX_DURATION per PROGRESS.md §4 Layer 2
]

# 23 FailureComponent enum values — order matches PROGRESS.md §4 Layer 0 exactly
FAILURE_COMPONENTS = [
    "engine_left", "engine_right", "engine_both",
    "alternator", "battery", "voltage_regulator",
    "fuel_pump", "fuel_selector", "fuel_system",
    "fuel_tank_left", "fuel_tank_right",
    "carburetor", "magnetos",
    "vacuum_pump", "gyroscope", "attitude_indicator",
    "flaps", "trim_system", "hydraulics",
    "landing_gear", "rudder", "ailerons", "elevator",
]
FAILURE_TO_IDX = {c: i for i, c in enumerate(FAILURE_COMPONENTS)}
N_FAILURE_COMPONENTS = len(FAILURE_COMPONENTS)  # 23

# FEATURE_DIM = 13 numeric + 1 fatigue + 23 failure one-hot
FEATURE_DIM: int = len(STATE_KEYS) + 1 + N_FAILURE_COMPONENTS  # = 37

# Train / val split — first 400K = train, remaining 100K = val
# Per context1 [Layer 2: Transformer Learning]: "Train on 80% of trajectories (400K)"
TRAIN_SIZE: int = 400_000
TOTAL_SIZE: int = 500_000

_SCALE_TENSOR = torch.tensor(STATE_SCALES, dtype=torch.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(IterableDataset):
    """
    Streams trajectories.jsonl.gz, yielding (features_tensor, pad_mask, label).

    features_tensor : (MAX_SEQ_LEN, FEATURE_DIM) float32
    pad_mask        : (MAX_SEQ_LEN,) bool  — True = real step, False = padding
    label           : () int64  — 1 = catastrophic, 0 = recoverable
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.path = path
        self.split = split
        self.max_seq_len = max_seq_len

    # ── IterableDataset protocol ───────────────────────────────────────────────

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id   = worker.id          if worker else 0
        num_workers = worker.num_workers if worker else 1

        # Line range for this split
        lo = 0          if self.split == "train" else TRAIN_SIZE
        hi = TRAIN_SIZE if self.split == "train" else TOTAL_SIZE

        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for abs_idx, line in enumerate(f):
                # Stop once we're past the split's range
                if abs_idx >= hi:
                    break
                # Skip lines before this split's range
                if abs_idx < lo:
                    continue
                # Worker sharding: each worker handles a disjoint slice
                if (abs_idx - lo) % num_workers != worker_id:
                    continue

                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                result = self._process_record(record)
                if result is not None:
                    yield result

    # ── Record processing ──────────────────────────────────────────────────────

    def _process_record(
        self, record: dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert a raw trajectory dict to (features, pad_mask, label)."""
        states = record.get("states")
        if not states:
            return None

        # ── Label ──────────────────────────────────────────────────────────────
        outcome = record.get("outcome", "recoverable")
        label = torch.tensor(
            1 if outcome == "catastrophic" else 0,
            dtype=torch.long,
        )

        # ── Fatigue scalar (broadcast across all time steps) ───────────────────
        init = record.get("init", {})
        fatigue = float(init.get("fatigue", 0.5))
        fatigue_t = torch.tensor([fatigue], dtype=torch.float32)

        # ── Primary failure one-hot (first injected failure component) ──────────
        failures = record.get("failures", [])
        failure_vec = torch.zeros(N_FAILURE_COMPONENTS, dtype=torch.float32)
        if failures:
            comp = failures[0].get("component", "")
            idx = FAILURE_TO_IDX.get(comp)
            if idx is not None:
                failure_vec[idx] = 1.0
        # failure_vec is constant across all steps — broadcast at assembly time

        # ── Build padded feature tensor ────────────────────────────────────────
        seq_len = min(len(states), self.max_seq_len)
        features = torch.zeros(
            (self.max_seq_len, FEATURE_DIM), dtype=torch.float32
        )

        for i in range(seq_len):
            s = states[i]
            numeric = torch.tensor(
                [float(s.get(k, 0.0)) for k in STATE_KEYS],
                dtype=torch.float32,
            )
            numeric = numeric / _SCALE_TENSOR  # normalise to ~[0, 1]

            features[i, :13]   = numeric
            features[i, 13]    = fatigue          # scalar broadcast
            features[i, 14:]   = failure_vec      # 23-dim one-hot

        # ── Padding mask: True = real step, False = pad ────────────────────────
        pad_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        pad_mask[:seq_len] = True

        return features, pad_mask, label


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_fn(
    batch: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Static collate: pads all sequences to MAX_SEQ_LEN.
    Kept for compatibility. Use dynamic_collate_fn for training.
    """
    features  = torch.stack([b[0] for b in batch])
    pad_masks = torch.stack([b[1] for b in batch])
    labels    = torch.stack([b[2] for b in batch])
    return features, pad_masks, labels


def dynamic_collate_fn(
    batch: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dynamic collate: pads each batch only to ITS OWN maximum real sequence
    length, not to global MAX_SEQ_LEN (1200).

    Why this matters:
      Diagnostics showed sequence lengths of mean=9, median=10 in catastrophic
      trajectories. With static padding to 1200, every batch ran attention over
      (B, nhead, 1201, 1201) matrices where only ~10 positions were real.
      Attention cost is O(seq²) — static padding was doing ~13,000× more work
      than necessary for short sequences.

    With dynamic padding:
      A batch of 9-step sequences → attention over (B, nhead, 10, 10)
      A batch of 500-step sequences → attention over (B, nhead, 501, 501)
      Expected speedup: 50–100× for catastrophic-trajectory batches.

    Input batch: list of (features, pad_mask, label) where each features is
      either (MAX_SEQ_LEN, FEATURE_DIM) [from TrajectoryDataset] or
      (actual_len, FEATURE_DIM) [from ShardedDataset with yield_unpadded=True].

    Output: same format as collate_fn but seq dim = batch_max_len.
    """
    import torch.nn.functional as F

    # Find the true length of each item (last True in pad_mask, or tensor dim)
    features_list = [b[0] for b in batch]
    masks_list    = [b[1] for b in batch]
    labels        = torch.stack([b[2] for b in batch])

    # Real length = number of True entries in pad_mask (or full length if no mask)
    real_lens = []
    for feat, mask in zip(features_list, masks_list):
        if mask.any():
            real_lens.append(int(mask.sum().item()))
        else:
            real_lens.append(feat.shape[0])

    batch_max = max(real_lens)

    # Pad/truncate each item to batch_max
    padded_features = []
    padded_masks    = []
    for feat, mask, rlen in zip(features_list, masks_list, real_lens):
        # Truncate to real content first (removes static padding)
        f = feat[:rlen]                              # (rlen, FEATURE_DIM)
        m = mask[:rlen]                              # (rlen,)

        # Pad to batch_max
        pad = batch_max - rlen
        if pad > 0:
            f = F.pad(f, (0, 0, 0, pad))            # (batch_max, FEATURE_DIM)
            m = F.pad(m, (0, pad), value=False)      # (batch_max,)

        padded_features.append(f)
        padded_masks.append(m)

    return (
        torch.stack(padded_features),   # (B, batch_max, FEATURE_DIM)
        torch.stack(padded_masks),      # (B, batch_max)
        labels,                         # (B,)
    )