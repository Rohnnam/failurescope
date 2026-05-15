#!/usr/bin/env python3
"""
FailureScope — Cascade Preprocessor
======================================
Replaces preprocess_shards.py. Builds two datasets from one gz scan:

  1. CASCADE SHARDS (/opt/failurescope_data/cascade_shards/)
     For Transformer training. States from inject_t onward (cascade window only).
     Filter: cascade_duration >= MIN_CASCADE_DURATION_S (default 30s).
     Feature: 13 STATE_KEYS only (no failure one-hot).
     Context: 27-dim hole-alignment features (separate from sequence).

  2. SNAPSHOT ARRAYS (/opt/failurescope_data/snapshots/)
     For XGBoost training. State at inject_t for ALL 500K trajectories.
     16 features: altitude, fatigue, voltage, pump_eff, failure_type, etc.

Both datasets use the same stratified 80/20 train/val split.

Memory-safe two-pass approach:
  Pass 1 (~3min): metadata only — collect line indices + labels + cascade durations
  Split:           stratified random 80/20 within each label class
  Pass 2 (~15min): extract tensors for cascade shards + snapshots simultaneously

Usage:
  conda activate aviate
  python preprocess_cascade.py          # default
  python preprocess_cascade.py --check  # verify existing data
  python preprocess_cascade.py --inspect-first  # run data inspector first
"""

import argparse
import gzip
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from backend.layer3.constants import (
    CASCADE_MAX_STEPS, CASCADE_CONTEXT_DIM, FAILURE_TO_IDX,
    MIN_CASCADE_DURATION_S, N_FAILURE_COMPONENTS, N_STATE_KEYS,
    SKILL_TO_IDX, SNAPSHOT_FEATURE_NAMES, STATE_KEYS, STATE_SCALES,
    TOTAL_TRAJECTORIES, TRAIN_FRACTION, TRAJECTORIES_PATH_STR,
    WEATHER_TO_IDX,
)
from backend.layer3.cascade_dataset import (
    CASCADE_SHARD_DIR, extract_cascade_sequence, extract_context_vector,
)
from backend.layer3.snapshot_dataset import (
    SNAPSHOT_DIR, extract_snapshot_features,
)

TRAJECTORIES_PATH = Path(TRAJECTORIES_PATH_STR)
CASCADE_SHARD_SIZE = 5_000
_SCALE = torch.tensor(STATE_SCALES[:N_STATE_KEYS], dtype=torch.float32)


# ── Pass 1: Metadata collection ───────────────────────────────────────────────

def _pass1_metadata() -> Tuple[List[tuple], List[tuple]]:
    """
    Collect metadata for all 500K trajectories — FAST version.

    The bottleneck in the original was json.loads on full lines:
    each line has up to 5937 state dicts × 15 keys = 89K dict creations.
    At 500K lines that's 44 billion Python dict creations.

    Fix: find the "states" key in each line and parse only the prefix.
    Count actual_len by counting {"t": occurrences (each state has a "t" key).
    This reduces dict creations from ~89K to ~10 per line → ~100x faster.
    """
    all_meta: List[tuple] = []
    t0 = time.time()

    with gzip.open(TRAJECTORIES_PATH, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                # Find where "states" array starts
                states_marker = '"states"'
                states_pos = line.find(states_marker)

                if states_pos > 0:
                    # Count actual_len by counting {"t": occurrences (fast string op)
                    actual_len_raw = line[states_pos:].count('"t":')
                    actual_len = min(actual_len_raw, CASCADE_MAX_STEPS)

                    # Parse only the prefix (no states array → ~10 dicts not 89K)
                    prefix = line[:states_pos].rstrip(', \t\n') + '}'
                    rec = json.loads(prefix)
                else:
                    # Short line — parse fully
                    rec = json.loads(line)
                    actual_len = min(len(rec.get("states", [])), CASCADE_MAX_STEPS)

                label    = 1 if rec.get("outcome") == "catastrophic" else 0
                init     = rec.get("init", {})
                fatigue  = float(init.get("fatigue", 0.5))
                fails    = rec.get("failures", [])
                fail_idx = (
                    FAILURE_TO_IDX.get(fails[0].get("component", ""), -1)
                    if fails else -1
                )
                inject_t = float(fails[0].get("trigger_time_s", 0.0)) if fails else 0.0
                ttc_s    = float(rec.get("ttc_s") or rec.get("duration_s") or 0.0)
                cascade_dur = ttc_s - inject_t

                all_meta.append((line_idx, label, cascade_dur, actual_len, inject_t))

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

            if (line_idx + 1) % 10_000 == 0:
                elapsed = time.time() - t0
                rate    = (line_idx + 1) / elapsed
                eta     = (TOTAL_TRAJECTORIES - line_idx - 1) / rate
                pct     = (line_idx + 1) / TOTAL_TRAJECTORIES * 100
                print(
                    f"\r  Pass 1: {line_idx+1:,}/{TOTAL_TRAJECTORIES:,} "
                    f"({pct:.0f}%)  {elapsed:.0f}s  ETA {eta:.0f}s  "
                    f"({rate:.0f} lines/s)",
                    end="", flush=True,
                )

    print(f"\r  Pass 1 complete: {len(all_meta):,} records  ({time.time()-t0:.0f}s)          ")
    return all_meta


def _stratified_split(
    all_meta: List[tuple],
    rng: np.random.Generator,
) -> Tuple[List[tuple], List[tuple]]:
    """
    Stratified 80/20 random split within each label class.
    Cascade meta: filter to cascade_duration >= MIN_CASCADE_DURATION_S.
    """
    labels = np.array([m[1] for m in all_meta], dtype=np.int8)
    durs   = np.array([m[2] for m in all_meta], dtype=np.float32)

    cat_idx = np.where(labels == 1)[0]
    rec_idx = np.where(labels == 0)[0]

    # Random 80/20 split
    cat_perm = rng.permutation(len(cat_idx))
    rec_perm = rng.permutation(len(rec_idx))

    n_cat_tr = int(TRAIN_FRACTION * len(cat_idx))
    n_rec_tr = int(TRAIN_FRACTION * len(rec_idx))

    train_cat = [all_meta[cat_idx[i]] for i in cat_perm[:n_cat_tr]]
    train_rec = [all_meta[rec_idx[i]] for i in rec_perm[:n_rec_tr]]
    val_cat   = [all_meta[cat_idx[i]] for i in cat_perm[n_cat_tr:]]
    val_rec   = [all_meta[rec_idx[i]] for i in rec_perm[n_rec_tr:]]

    train_all = train_cat + train_rec
    val_all   = val_cat   + val_rec

    print(f"  All trajectories  — train: {len(train_all):,}  val: {len(val_all):,}")
    print(f"  Train label dist: cat={len(train_cat):,} ({len(train_cat)/len(train_all)*100:.1f}%)  "
          f"rec={len(train_rec):,}")
    print(f"  Val   label dist: cat={len(val_cat):,} ({len(val_cat)/len(val_all)*100:.1f}%)  "
          f"rec={len(val_rec):,}")

    # Cascade filter: cascade_duration >= MIN_CASCADE_DURATION_S
    train_cascade = [m for m in train_all if m[2] >= MIN_CASCADE_DURATION_S]
    val_cascade   = [m for m in val_all   if m[2] >= MIN_CASCADE_DURATION_S]

    n_tr_c = len(train_cascade)
    n_va_c = len(val_cascade)
    n_tr_c_cat = sum(1 for m in train_cascade if m[1] == 1)
    n_va_c_cat = sum(1 for m in val_cascade   if m[1] == 1)

    print(f"\n  Cascade filter (>={MIN_CASCADE_DURATION_S:.0f}s):")
    print(f"  Train: {n_tr_c:,}  (cat={n_tr_c_cat:,} {n_tr_c_cat/max(n_tr_c,1)*100:.1f}%)")
    print(f"  Val:   {n_va_c:,}  (cat={n_va_c_cat:,} {n_va_c_cat/max(n_va_c,1)*100:.1f}%)")

    return train_all, val_all, train_cascade, val_cascade


# ── Pass 2: Tensor extraction ──────────────────────────────────────────────────

def _pass2_extract(
    train_all: List[tuple],
    val_all: List[tuple],
    train_cascade: List[tuple],
    val_cascade: List[tuple],
    partial_dir: Path,
) -> Tuple[Dict, Dict, List, List, List, List]:
    """
    Single gz scan. Builds:
      - Cascade shard buffers (train + val)
      - Snapshot arrays (train + val, all trajectories)
    Returns cascade partial files + snapshot lists.
    """
    # Build lookup maps
    all_needed_lines: set = {m[0] for m in train_all + val_all}
    line_to_split:    Dict[int, str] = {}
    for m in train_all: line_to_split[m[0]] = "train"
    for m in val_all:   line_to_split[m[0]] = "val"

    # Cascade line maps (position → shard_idx)
    tr_cascade_map: Dict[int, int] = {m[0]: pos // CASCADE_SHARD_SIZE
                                       for pos, m in enumerate(train_cascade)}
    va_cascade_map: Dict[int, int] = {m[0]: pos // CASCADE_SHARD_SIZE
                                       for pos, m in enumerate(val_cascade)}
    cascade_inject: Dict[int, float] = {m[0]: m[4] for m in train_cascade + val_cascade}

    # Snapshot lists
    snap_tr_X, snap_tr_y = [], []
    snap_va_X, snap_va_y = [], []

    # Cascade shard buffers
    cascade_bufs:  Dict[str, Dict[int, list]] = {"train": defaultdict(list), "val": defaultdict(list)}
    partial_files: Dict[str, Dict[int, List[Path]]] = {"train": defaultdict(list), "val": defaultdict(list)}
    FLUSH = 300

    t0       = time.time()
    n_written = 0

    with gzip.open(TRAJECTORIES_PATH, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx not in all_needed_lines:
                continue
            split = line_to_split[line_idx]
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            # ── Snapshot (all trajectories) ────────────────────────────────────
            feat = extract_snapshot_features(rec)
            if feat is not None:
                label = 1 if rec.get("outcome") == "catastrophic" else 0
                if split == "train":
                    snap_tr_X.append(feat); snap_tr_y.append(label)
                else:
                    snap_va_X.append(feat); snap_va_y.append(label)

            # ── Cascade shard (filtered trajectories) ──────────────────────────
            in_cascade = line_idx in tr_cascade_map or line_idx in va_cascade_map
            if in_cascade:
                states   = rec.get("states", [])
                inject_t = cascade_inject.get(line_idx, 0.0)
                ttc_s    = float(rec.get("ttc_s") or len(states))

                
                is_cat = rec.get("outcome") == "catastrophic"
                seq, actual_len = extract_cascade_sequence(states, inject_t, ttc_s, is_catastrophic=is_cat)


                inject_state = None
                for s in states:
                    if float(s.get("t", 0.0)) >= inject_t - 0.5:
                        inject_state = s
                        break

                ctx   = extract_context_vector(rec, inject_state)
                label = 1 if rec.get("outcome") == "catastrophic" else 0

                item = {
                    "seq":    seq,
                    "len":    actual_len,
                    "ctx":    ctx.half(),
                    "label":  label,
                }

                shard_idx = (tr_cascade_map if split == "train" else va_cascade_map)[line_idx]
                cascade_bufs[split][shard_idx].append(item)

                if len(cascade_bufs[split][shard_idx]) >= FLUSH:
                    part_n = len(partial_files[split][shard_idx])
                    ppath  = partial_dir / f"cas_{split}_{shard_idx:04d}_p{part_n:04d}.pkl"
                    import pickle as _pk
                    with open(ppath, "wb") as pf:
                        _pk.dump(cascade_bufs[split][shard_idx], pf, protocol=4)
                    partial_files[split][shard_idx].append(ppath)
                    cascade_bufs[split][shard_idx] = []

            n_written += 1
            if n_written % 50_000 == 0:
                print(f"\r  Pass 2: {n_written:,}/{len(line_to_split):,} records  ({time.time()-t0:.0f}s)", end="", flush=True)

    # Flush remaining cascade buffers
    import pickle as _pk
    for split in ("train", "val"):
        for shard_idx, buf in cascade_bufs[split].items():
            if buf:
                part_n = len(partial_files[split][shard_idx])
                ppath  = partial_dir / f"cas_{split}_{shard_idx:04d}_p{part_n:04d}.pkl"
                with open(ppath, "wb") as pf:
                    _pk.dump(buf, pf, protocol=4)
                partial_files[split][shard_idx].append(ppath)

    print(f"\r  Pass 2 complete: {n_written:,} records  ({time.time()-t0:.0f}s)          ")
    return partial_files, snap_tr_X, snap_tr_y, snap_va_X, snap_va_y


# ── Finalize ───────────────────────────────────────────────────────────────────

def _write_cascade_shards(
    partial_files: dict,
    n_shards: Dict[str, int],
) -> None:
    import pickle as _pk

    for split in ("train", "val"):
        split_dir = CASCADE_SHARD_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0

        for shard_idx in range(n_shards[split]):
            parts = partial_files[split].get(shard_idx, [])
            if not parts:
                continue

            all_items = []
            for p in parts:
                with open(p, "rb") as f:
                    all_items.extend(_pk.load(f))

            if not all_items:
                continue

            N       = len(all_items)
            max_len = max(it["len"] for it in all_items)

            sequences = torch.zeros(N, max_len, N_STATE_KEYS, dtype=torch.float16)
            lengths   = torch.zeros(N, dtype=torch.int16)
            contexts  = torch.zeros(N, CASCADE_CONTEXT_DIM, dtype=torch.float16)
            labels    = torch.zeros(N, dtype=torch.int8)

            for i, it in enumerate(all_items):
                L = it["len"]
                sequences[i, :L] = it["seq"]
                lengths[i]       = L
                contexts[i]      = it["ctx"]
                labels[i]        = it["label"]

            shard_path = split_dir / f"shard_{shard_idx:04d}.pt"
            torch.save({
                "sequences": sequences,
                "lengths":   lengths,
                "context":   contexts,
                "labels":    labels,
            }, shard_path)
            n_written += 1

            for p in parts:
                p.unlink(missing_ok=True)

            if (shard_idx + 1) % 10 == 0:
                mb = shard_path.stat().st_size / 1e6
                print(f"  {split} shard {shard_idx+1}/{n_shards[split]}  ({mb:.0f} MB)")

        print(f"  {n_written} {split} cascade shards written")


def _save_snapshots(
    snap_tr_X, snap_tr_y, snap_va_X, snap_va_y,
) -> None:
    import numpy as _np
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _np.save(SNAPSHOT_DIR / "train_X.npy", _np.array(snap_tr_X, dtype=_np.float32))
    _np.save(SNAPSHOT_DIR / "train_y.npy", _np.array(snap_tr_y, dtype=_np.int8))
    _np.save(SNAPSHOT_DIR / "val_X.npy",   _np.array(snap_va_X, dtype=_np.float32))
    _np.save(SNAPSHOT_DIR / "val_y.npy",   _np.array(snap_va_y, dtype=_np.int8))
    with open(SNAPSHOT_DIR / "feature_names.json", "w") as f:
        json.dump(SNAPSHOT_FEATURE_NAMES, f, indent=2)
    n_tr = len(snap_tr_X)
    n_va = len(snap_va_X)
    print(f"  Snapshots: train={n_tr:,}  val={n_va:,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess(inspect_first: bool = False) -> None:
    if not TRAJECTORIES_PATH.exists():
        print(f"ERROR: {TRAJECTORIES_PATH} not found.")
        sys.exit(1)

    if inspect_first:
        from backend.layer3.data_inspector import inspect as run_inspect
        result = run_inspect(n=50_000)
        go_cascade = result["go_nogo"]["cascade_transformer"]["go"]
        if not go_cascade:
            print("⚠ Cascade transformer may have insufficient data. Continuing anyway.")

    partial_dir = CASCADE_SHARD_DIR / "_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    rng     = np.random.default_rng(42)

    print(f"\nCascade preprocessing")
    print(f"  Min cascade duration : {MIN_CASCADE_DURATION_S:.0f}s")
    print(f"  Max cascade steps    : {CASCADE_MAX_STEPS}")
    print(f"  Cascade shard size   : {CASCADE_SHARD_SIZE:,}")

    # Pass 1
    print("\n[Pass 1] Reading metadata …")
    all_meta = _pass1_metadata()

    # Split
    print("\n[Split] Stratified 80/20 split …")
    train_all, val_all, train_cascade, val_cascade = _stratified_split(all_meta, rng)

    # Save val line indices for attention extractor
    val_line_set = {m[0] for m in val_all}
    val_idx_path = CASCADE_SHARD_DIR / "val_line_indices.pkl"
    CASCADE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(val_idx_path) + ".tmp", "wb") as f:
        pickle.dump(val_line_set, f, protocol=4)

    n_shards = {
        "train": math.ceil(len(train_cascade) / CASCADE_SHARD_SIZE),
        "val":   math.ceil(len(val_cascade)   / CASCADE_SHARD_SIZE),
    }
    print(f"  Cascade shards: train={n_shards['train']}  val={n_shards['val']}")

    # Pass 2
    print("\n[Pass 2] Extracting tensors …")
    partial_files, snap_tr_X, snap_tr_y, snap_va_X, snap_va_y = _pass2_extract(
        train_all, val_all, train_cascade, val_cascade, partial_dir
    )

    # Write cascade shards
    print("\n[Write] Writing cascade shards …")
    _write_cascade_shards(partial_files, n_shards)

    # Save snapshots
    print("\n[Write] Saving snapshot arrays …")
    _save_snapshots(snap_tr_X, snap_tr_y, snap_va_X, snap_va_y)

    # Finalize val indices
    import os
    os.rename(str(val_idx_path) + ".tmp", val_idx_path)

    # Cleanup
    try:
        partial_dir.rmdir()
    except OSError:
        pass

    elapsed = time.time() - t_total
    cas_gb  = sum(p.stat().st_size for p in CASCADE_SHARD_DIR.rglob("*.pt")) / 1e9
    snap_gb = sum(p.stat().st_size for p in SNAPSHOT_DIR.glob("*.npy")) / 1e9

    print(f"\n{'='*60}")
    print(f"Preprocessing complete  ({elapsed:.0f}s = {elapsed/60:.1f} min)")
    print(f"  Cascade shards : {cas_gb:.1f} GB")
    print(f"  Snapshots      : {snap_gb:.1f} GB")
    print(f"{'='*60}")
    print("\nReady to train:")
    print("  python train_layer3.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--inspect-first", action="store_true")
    args = parser.parse_args()

    if args.check:
        from backend.layer3.cascade_dataset import shards_exist, CASCADE_SHARD_DIR
        from backend.layer3.snapshot_dataset import snapshots_exist, SNAPSHOT_DIR
        print(f"Cascade shards: {'✓' if shards_exist() else '✗'}  ({CASCADE_SHARD_DIR})")
        print(f"Snapshots:      {'✓' if snapshots_exist() else '✗'}  ({SNAPSHOT_DIR})")
        return

    preprocess(inspect_first=args.inspect_first)


if __name__ == "__main__":
    main()