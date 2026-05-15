"""
FailureScope Layer 3 — Shard Dataset
======================================
Replaces TrajectoryDataset (gzip streaming) with a fast binary shard loader.

Problem with gzip streaming (observed in session 3):
  With num_workers=4 and IterableDataset, each worker opens the full gz file
  and skips to its slice. Effective IO = 4 × 1.74 GB = 7 GB per epoch.
  GPU was idle >95% of the time. VRAM only 0.11 GB used.

This solution:
  1. preprocess_shards.py runs ONCE: converts gz → binary .pt shards
  2. ShardedDataset loads .pt shards directly (torch.load = memory-mapped binary)
  3. Worker i handles shards[i::num_workers] — zero redundant IO
  4. Result: each worker reads exactly 1/num_workers of the data, no skipping

Shard format (saved by preprocess_shards.py):
  {
    "sequences":   Tensor(N, max_len_in_shard, 13)  fp16 — 13 numeric state keys
    "lengths":     Tensor(N,)                        int16 — actual steps per trajectory
    "fatigue":     Tensor(N,)                        fp16  — scalar per trajectory
    "failure_idx": Tensor(N,)                        int16 — index into FAILURE_COMPONENTS
    "labels":      Tensor(N,)                        int8  — 1=catastrophic, 0=recoverable
  }

Collate output (identical to original TrajectoryDataset):
  features  : (B, max_len_in_batch, 37)   float32
  pad_mask  : (B, max_len_in_batch)        bool
  labels    : (B,)                         int64

Storage estimate:
  Sorted by length within train/val, so shard max_len reflects actual data:
  - Short shards (catastrophic, ~300 steps): small
  - Long shards (recoverable, ~1200 steps): larger
  Total: ~11-14 GB on /opt (vs 44 GB uncompressed unoptimised)
"""

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .dataset import FAILURE_COMPONENTS, FEATURE_DIM, MAX_SEQ_LEN, STATE_KEYS

N_FAILURE_COMPONENTS = len(FAILURE_COMPONENTS)   # 23
N_NUMERIC_KEYS       = len(STATE_KEYS)            # 13

SHARD_DIR = Path("/opt/failurescope_data/shards")


# ── ShardedDataset ─────────────────────────────────────────────────────────────

class ShardedDataset(IterableDataset):
    """
    Iterates pre-built .pt shard files. Each worker handles a disjoint
    subset of shards — no redundant file IO.

    Args:
        shard_dir  : directory containing train/ or val/ subdirectory with .pt files
        split      : "train" or "val"
        shuffle    : if True, shuffle shard order and within-shard order each epoch
        max_seq_len: truncate/pad sequences to this length
    """

    def __init__(
        self,
        shard_dir: Path = SHARD_DIR,
        split: str = "train",
        shuffle: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        assert split in ("train", "val")
        self.split_dir  = shard_dir / split
        self.shuffle    = shuffle and (split == "train")
        self.max_seq_len = max_seq_len

        shard_paths = sorted(self.split_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found in {self.split_dir}. "
                "Run: python preprocess_shards.py"
            )
        self.shard_paths = shard_paths

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker     = get_worker_info()
        worker_id  = worker.id          if worker else 0
        n_workers  = worker.num_workers if worker else 1

        # Each worker handles a disjoint slice of shards
        my_shards = self.shard_paths[worker_id::n_workers]

        if self.shuffle:
            random.shuffle(my_shards)

        for shard_path in my_shards:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            yield from self._iter_shard(shard)

    def _iter_shard(
        self, shard: dict
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Yield (features, pad_mask, label) from a loaded shard dict.
        Sequences are yielded at their ACTUAL length (not padded to max_seq_len).
        dynamic_collate_fn handles batch-level padding to the batch's own max.
        """
        sequences   = shard["sequences"]    # (N, shard_max_len, 13) fp16
        lengths     = shard["lengths"]      # (N,) int16
        fatigue_v   = shard["fatigue"]      # (N,) fp16
        failure_idx = shard["failure_idx"]  # (N,) int16
        labels      = shard["labels"]       # (N,) int8

        N = labels.shape[0]
        indices = list(range(N))
        if self.shuffle:
            random.shuffle(indices)

        for i in indices:
            actual_len = min(int(lengths[i].item()), self.max_seq_len)
            if actual_len == 0:
                actual_len = 1

            # Yield ONLY the real steps — no padding to max_seq_len here.
            # dynamic_collate_fn pads to batch maximum, which is much smaller
            # than max_seq_len for short catastrophic trajectories.
            seq = sequences[i, :actual_len].float()   # (actual_len, 13)

            fatigue_scalar = float(fatigue_v[i].item())
            fail_idx       = int(failure_idx[i].item())

            failure_vec = torch.zeros(N_FAILURE_COMPONENTS, dtype=torch.float32)
            if 0 <= fail_idx < N_FAILURE_COMPONENTS:
                failure_vec[fail_idx] = 1.0

            # Static features broadcast to actual_len only
            static = torch.cat([
                torch.full((actual_len, 1), fatigue_scalar),
                failure_vec.unsqueeze(0).expand(actual_len, -1),
            ], dim=1)                                          # (actual_len, 24)

            features = torch.cat([seq, static], dim=1)        # (actual_len, 37)

            # pad_mask = all True (real steps only, no padding at item level)
            pad_mask = torch.ones(actual_len, dtype=torch.bool)

            label = torch.tensor(int(labels[i].item()), dtype=torch.long)

            yield features, pad_mask, label


# ── Shard existence check ──────────────────────────────────────────────────────

def shards_exist(shard_dir: Path = SHARD_DIR) -> bool:
    return (
        (shard_dir / "train").exists()
        and len(list((shard_dir / "train").glob("shard_*.pt"))) > 0
        and (shard_dir / "val").exists()
        and len(list((shard_dir / "val").glob("shard_*.pt"))) > 0
    )


def shard_info(shard_dir: Path = SHARD_DIR) -> dict:
    n_train = len(list((shard_dir / "train").glob("shard_*.pt")))
    n_val   = len(list((shard_dir / "val").glob("shard_*.pt")))
    size_gb = sum(
        p.stat().st_size for p in shard_dir.rglob("*.pt")
    ) / 1e9
    return {
        "train_shards": n_train,
        "val_shards":   n_val,
        "total_size_gb": round(size_gb, 2),
    }