"""
FailureScope Layer 3 — Snapshot Dataset
=========================================
Extracts snapshot features at inject_t for XGBoost swiss cheese analysis.

A snapshot is the aircraft state at the MOMENT of failure injection:
  "Which combination of pre-existing conditions determines whether
   this failure becomes catastrophic?"

This is the swiss cheese model question: which holes in defences were aligned
at the moment the failure was introduced?

Features (N_SNAPSHOT_FEATURES = 16):
  From state at inject_t:
    alt_at_inject, spd_at_inject, v_at_inject, rpm_at_inject,
    ff_at_inject, fuel_total_at_inject, pump_eff_at_inject,
    pitch_at_inject, vs_at_inject, bat_at_inject, roll_at_inject
  From init dict:
    fatigue, skill_encoded (0-4), weather_encoded (0-3)
  From failures list:
    failure_encoded (0-22)
  Derived:
    inject_time_frac (inject_t / 3600 — how late in flight)

Output files:
  /opt/failurescope_data/snapshots/train_X.npy  — (N_train, 16) float32
  /opt/failurescope_data/snapshots/train_y.npy  — (N_train,) int8
  /opt/failurescope_data/snapshots/val_X.npy
  /opt/failurescope_data/snapshots/val_y.npy
  /opt/failurescope_data/snapshots/feature_names.json

All 500K trajectories are usable for snapshot model (no cascade duration filter).
Stratified 80/20 split preserving class balance.
"""

import gzip
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .constants import (
    FAILURE_TO_IDX, N_SNAPSHOT_FEATURES, SKILL_TO_IDX,
    SNAPSHOT_FEATURE_NAMES, STATE_KEYS, TOTAL_TRAJECTORIES,
    TRAIN_FRACTION, TRAJECTORIES_PATH_STR, WEATHER_TO_IDX,
)

SNAPSHOT_DIR = Path("/opt/failurescope_data/snapshots")


def extract_snapshot_features(record: dict) -> Optional[np.ndarray]:
    """
    Extract 16 snapshot features from a trajectory record.
    Returns None if required fields are missing.
    """
    fails   = record.get("failures", [])
    states  = record.get("states", [])
    init    = record.get("init", {})
    if not fails or not states:
        return None

    inject_t = float(fails[0].get("trigger_time_s", 0.0))
    comp     = fails[0].get("component", "")

    # Find state at inject_t (first state at or after inject_t)
    inject_state = None
    for s in states:
        if float(s.get("t", 0.0)) >= inject_t - 0.5:
            inject_state = s
            break
    if inject_state is None:
        inject_state = states[0]

    # Extract features
    fuel_total = float(inject_state.get("fuel_l", 0.0)) + float(inject_state.get("fuel_r", 0.0))
    fail_idx   = float(FAILURE_TO_IDX.get(comp, -1))
    skill_idx  = float(SKILL_TO_IDX.get(str(init.get("skill", "commercial")), 2))
    weather_idx = float(WEATHER_TO_IDX.get(str(init.get("weather", "vmc")), 0))

    features = np.array([
        float(inject_state.get("alt",      5000.0)),  # alt_at_inject
        float(inject_state.get("spd",      100.0)),   # spd_at_inject
        float(inject_state.get("v",        14.0)),    # v_at_inject
        float(inject_state.get("rpm",      2400.0)),  # rpm_at_inject
        float(inject_state.get("ff",       11.0)),    # ff_at_inject
        fuel_total,                                    # fuel_total_at_inject
        float(inject_state.get("pump_eff", 1.0)),     # pump_eff_at_inject
        float(inject_state.get("pitch",    5.0)),     # pitch_at_inject
        float(inject_state.get("vs",       0.0)),     # vs_at_inject
        float(init.get("fatigue",          0.5)),     # fatigue
        skill_idx,                                     # skill_encoded
        weather_idx,                                   # weather_encoded
        fail_idx,                                      # failure_encoded
        inject_t / 3600.0,                             # inject_time_frac
        float(inject_state.get("bat",      33.0)),    # bat_at_inject
        float(inject_state.get("roll",     0.0)),     # roll_at_inject
    ], dtype=np.float32)

    return features


def build_snapshot_arrays(
    trajectories_path: Path,
    val_line_indices_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single gz scan. Returns (train_X, train_y, val_X, val_y).
    Uses saved val_line_indices.pkl for split if available.
    Falls back to stratified random 80/20 split within each label class.
    """
    print("[Snapshot] Reading all trajectories for snapshot extraction …")

    all_X = []
    all_y = []

    # Optional: load saved val indices from cascade preprocessing
    val_line_set = None
    if val_line_indices_path and val_line_indices_path.exists():
        with open(val_line_indices_path, "rb") as f:
            val_line_set = pickle.load(f)
        print(f"[Snapshot] Using saved val line indices ({len(val_line_set):,} lines)")

    line_assignments = []  # "train" or "val" per line

    with gzip.open(trajectories_path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                rec  = json.loads(line)
                feat = extract_snapshot_features(rec)
                if feat is None:
                    continue

                label = 1 if rec.get("outcome") == "catastrophic" else 0
                all_X.append(feat)
                all_y.append(label)

                if val_line_set is not None:
                    line_assignments.append("val" if line_idx in val_line_set else "train")
                else:
                    line_assignments.append(None)

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

            if (line_idx + 1) % 50_000 == 0:
                print(f"  {line_idx+1:,}/{TOTAL_TRAJECTORIES:,}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int8)
    print(f"[Snapshot] Extracted {len(X):,} snapshots  "
          f"(cat={y.sum():,}  rec={(y==0).sum():,})")

    # ── Split ──────────────────────────────────────────────────────────────────
    if val_line_set is not None:
        assignments = np.array(line_assignments)
        train_mask  = assignments == "train"
        val_mask    = assignments == "val"
        train_X, train_y = X[train_mask], y[train_mask]
        val_X,   val_y   = X[val_mask],   y[val_mask]
    else:
        # Stratified random 80/20 split
        rng = np.random.default_rng(42)
        cat_idx = np.where(y == 1)[0]
        rec_idx = np.where(y == 0)[0]

        cat_perm = rng.permutation(len(cat_idx))
        rec_perm = rng.permutation(len(rec_idx))

        n_cat_tr = int(TRAIN_FRACTION * len(cat_idx))
        n_rec_tr = int(TRAIN_FRACTION * len(rec_idx))

        train_idx = np.concatenate([cat_idx[cat_perm[:n_cat_tr]], rec_idx[rec_perm[:n_rec_tr]]])
        val_idx   = np.concatenate([cat_idx[cat_perm[n_cat_tr:]], rec_idx[rec_perm[n_rec_tr:]]])

        train_X, train_y = X[train_idx], y[train_idx]
        val_X,   val_y   = X[val_idx],   y[val_idx]

    print(f"[Snapshot] Train: {len(train_X):,}  "
          f"(cat={train_y.sum():,} {train_y.mean()*100:.1f}%)")
    print(f"[Snapshot] Val:   {len(val_X):,}  "
          f"(cat={val_y.sum():,} {val_y.mean()*100:.1f}%)")

    return train_X, train_y, val_X, val_y


def save_snapshots(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X:   np.ndarray, val_y:   np.ndarray,
    out_dir: Path = SNAPSHOT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_X.npy", train_X)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "val_X.npy",   val_X)
    np.save(out_dir / "val_y.npy",   val_y)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(SNAPSHOT_FEATURE_NAMES, f, indent=2)
    print(f"[Snapshot] Saved to {out_dir}")


def load_snapshots(
    out_dir: Path = SNAPSHOT_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(out_dir / "train_X.npy"),
        np.load(out_dir / "train_y.npy"),
        np.load(out_dir / "val_X.npy"),
        np.load(out_dir / "val_y.npy"),
    )


def snapshots_exist(out_dir: Path = SNAPSHOT_DIR) -> bool:
    return (out_dir / "train_X.npy").exists()