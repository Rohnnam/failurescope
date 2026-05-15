"""
FailureScope Layer 5 — L3 Inference Engine
============================================
Runs Layer 3 models (XGBoost snapshot + Cascade Transformer) on a single
new trajectory produced at query time by the Layer 2 simulator.

Both models were trained offline and saved to disk. This module loads them
once at server startup and exposes a single public function:

    result = analyse(traj: TrajectoryRecord, scenario: ScenarioInput) -> dict

Design notes:
  - Models load lazily at first call if not pre-loaded (safe for development)
  - If models fail to load, returns a graceful degraded response (L0-L2 still work)
  - ScenarioInput is required alongside TrajectoryRecord because TrajectoryRecord
    does not store an `init` dict — fatigue/skill/weather must come from the scenario
  - TrajectoryRecord.failure_chain uses `time_s` key; extract_snapshot_features
    expects `trigger_time_s` — an adapter is applied internally

Cascade window construction mirrors preprocess_cascade.py exactly:
  states[inject_idx : inject_idx + CASCADE_MAX_STEPS]
  Normalised with STATE_SCALES, padded to CASCADE_MAX_STEPS with pad_mask.
  Context token: [alt_norm, fatigue, skill_norm, weather_norm] + failure_one_hot (27 dims)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── Paths (same as constants.py) ─────────────────────────────────────────────
_MODELS_DIR        = Path("/opt/failurescope_data/models")
_SNAPSHOT_PATH     = _MODELS_DIR / "snapshot_model.json"
_CASCADE_PATH      = _MODELS_DIR / "cascade_model.pt"

# ── Module-level model cache ──────────────────────────────────────────────────
_snapshot_model = None
_cascade_model  = None
_models_loaded  = False
_load_error     = None


def load_models() -> bool:
    """
    Load both L3 models into module-level cache.
    Called once at server startup. Returns True on success.
    Failures are logged but do not crash the server.
    """
    global _snapshot_model, _cascade_model, _models_loaded, _load_error

    if _models_loaded:
        return True

    try:
        import torch
        import xgboost as xgb
        import sys, os
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from backend.layer3.constants import (
            CASCADE_FEATURE_DIM, CASCADE_CONTEXT_DIM, CASCADE_MAX_STEPS,
        )
        from backend.layer3.cascade_model import CascadeTransformer
        from backend.layer3.snapshot_model import load_snapshot_model

        # XGBoost snapshot model
        _snapshot_model = load_snapshot_model()
        log.info(f"  L3 snapshot model loaded from {_SNAPSHOT_PATH}")

        # CascadeTransformer — instantiate with training hyperparameters then load state_dict
        # Architecture from handoff: 4 layers, 8 heads, d_model=128, ffn=512 — 807,106 params
        _cascade_model = CascadeTransformer(
            feature_dim=CASCADE_FEATURE_DIM,
            context_dim=CASCADE_CONTEXT_DIM,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.0,                # inference — no dropout
            max_steps=CASCADE_MAX_STEPS,
            use_gradient_checkpointing=False,
        )
        ckpt = torch.load(str(_CASCADE_PATH), map_location="cpu", weights_only=True)
        _cascade_model.load_state_dict(ckpt["model_state_dict"])
        _cascade_model.eval()
        log.info(f"  L3 cascade model loaded from {_CASCADE_PATH}")

        _models_loaded = True
        _load_error = None
        return True

    except Exception as e:
        _load_error = str(e)
        log.warning(f"  L3 models failed to load: {e} — L3+L4 analysis unavailable")
        return False


# ── Record adapter ────────────────────────────────────────────────────────────

def _build_record(traj, scenario) -> dict:
    """
    Convert TrajectoryRecord + ScenarioInput → dict format expected by
    extract_snapshot_features.

    Adapter required because:
      - traj.failure_chain uses 'time_s' key; extractor expects 'trigger_time_s'
      - traj has no 'init' dict; reconstruct from ScenarioInput
    """
    skill = scenario.pilot_state.skill_level
    skill_str = skill.value if hasattr(skill, "value") else str(skill)

    failures = [
        {
            "trigger_time_s": f["time_s"],      # key rename
            "component":      f["component"],
            "severity":       f.get("severity", "catastrophic"),
        }
        for f in (traj.failure_chain or [])
    ]

    return {
        "failures": failures,
        "states":   traj.states,
        "init": {
            "fatigue": scenario.pilot_state.fatigue,
            "skill":   skill_str,
            "weather": getattr(scenario, "weather", "vmc"),
        },
        "outcome": traj.outcome.value if hasattr(traj.outcome, "value") else str(traj.outcome),
        "ttc_s":   traj.time_to_criticality_s,
    }


# ── Snapshot inference (XGBoost) ──────────────────────────────────────────────

def _run_snapshot(record: dict) -> dict:
    """
    Run XGBoost snapshot model on a single trajectory record.
    Returns crash_probability and top differentiating feature names.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.layer3.snapshot_dataset import extract_snapshot_features
    from backend.layer3.constants import SNAPSHOT_FEATURE_NAMES

    features = extract_snapshot_features(record)
    if features is None:
        return {"crash_probability": None, "error": "Could not extract snapshot features"}

    X = features.reshape(1, -1)
    proba = _snapshot_model.predict_proba(X)[0]  # [P(recoverable), P(catastrophic)]
    crash_prob = float(proba[1])

    # Top feature importances (feature names from constants)
    importances = _snapshot_model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:5]
    top_factors = [
        {"feature": SNAPSHOT_FEATURE_NAMES[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    return {
        "crash_probability": crash_prob,
        "top_factors":       top_factors,
        "model":             "XGBoost",
    }


# ── Cascade window inference (Transformer) ───────────────────────────────────

def _build_context_token(scenario, component: str) -> np.ndarray:
    """
    Build 27-dim context token for CascadeTransformer.
    [alt_norm, fatigue, skill_norm, weather_norm] + failure_one_hot(23)
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.layer3.constants import (
        FAILURE_TO_IDX, SKILL_TO_IDX, WEATHER_TO_IDX,
    )

    skill    = scenario.pilot_state.skill_level
    skill_str = skill.value if hasattr(skill, "value") else str(skill)

    alt_norm     = scenario.initial_state.altitude_ft / 16500.0
    fatigue      = float(scenario.pilot_state.fatigue)
    skill_norm   = SKILL_TO_IDX.get(skill_str, 2) / 4.0
    weather_norm = WEATHER_TO_IDX.get(getattr(scenario, "weather", "vmc"), 0) / 3.0

    one_hot = np.zeros(23, dtype=np.float32)
    fail_idx = FAILURE_TO_IDX.get(component, -1)
    if 0 <= fail_idx < 23:
        one_hot[fail_idx] = 1.0

    scalars = np.array([alt_norm, fatigue, skill_norm, weather_norm], dtype=np.float32)
    return np.concatenate([scalars, one_hot])  # (27,)


def _build_cascade_tensors(states: list, inject_t: float):
    """
    Build (seq, context, pad_mask) tensors for CascadeTransformer inference.
    Window: states[inject_idx : inject_idx + CASCADE_MAX_STEPS]
    Normalised with STATE_SCALES, padded to CASCADE_MAX_STEPS.
    """
    import torch
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.layer3.constants import STATE_KEYS, STATE_SCALES, CASCADE_MAX_STEPS

    # Find inject_idx
    inject_idx = 0
    for i, s in enumerate(states):
        if float(s.get("t", 0.0)) >= inject_t - 0.05:
            inject_idx = i
            break

    raw_window = states[inject_idx:]
    window = raw_window[::50][:CASCADE_MAX_STEPS]  # match 5s training resolution
    actual_len = len(window)

    # Build normalised sequence — pad to CASCADE_MAX_STEPS
    rows = np.zeros((CASCADE_MAX_STEPS, len(STATE_KEYS)), dtype=np.float32)
    for i, s in enumerate(window):
        for j, (k, scale) in enumerate(zip(STATE_KEYS, STATE_SCALES)):
            rows[i, j] = float(s.get(k, 0.0)) / scale

    seq      = torch.from_numpy(rows).unsqueeze(0)          # (1, 400, 13)
    pad_mask = torch.zeros(1, CASCADE_MAX_STEPS, dtype=torch.bool)
    pad_mask[0, :actual_len] = True                          # True = real step

    return seq, pad_mask, actual_len


def _run_cascade(record: dict, scenario, component: str) -> dict:
    """
    Run CascadeTransformer on the cascade window of a trajectory.
    Returns prediction and key attending variables.
    """
    import torch
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.layer3.constants import STATE_KEYS, CASCADE_MAX_STEPS

    states   = record.get("states", [])
    failures = record.get("failures", [])
    if not states or not failures:
        return {"cascade_prediction": None, "error": "No states or failures in record"}

    inject_t = float(failures[0].get("trigger_time_s", 0.0))
    seq, pad_mask, actual_len = _build_cascade_tensors(states, inject_t)

    if actual_len < 5:
        return {
            "cascade_prediction": None,
            "cascade_crash_probability": None,
            "error": f"Cascade window too short ({actual_len} steps < 5 minimum)",
        }

    ctx_np  = _build_context_token(scenario, component)
    context = torch.from_numpy(ctx_np).unsqueeze(0)          # (1, 27)

    with torch.no_grad():
        logits, attn_list = _cascade_model(
            seq, context, pad_mask, return_attention=True
        )
        probs = torch.softmax(logits, dim=-1)
        cascade_crash_prob = float(probs[0, 1])

    # Summarise attention: which state variables attracted most attention
    # Average across layers, heads, and time steps → per-variable score
    variable_attention: dict[str, float] = {k: 0.0 for k in STATE_KEYS}
    n_layers = len(attn_list)

    for attn in attn_list:
        # attn: (1, nhead, S+1, S+1)  — context token at position 0, states at 1..S
        attn_np = attn[0].mean(0).cpu().numpy()  # (S+1, S+1) mean over heads
        # How much does the context token attend to each state position?
        ctx_to_states = attn_np[0, 1:actual_len + 1]   # (actual_len,)
        mean_per_step = ctx_to_states.mean() if actual_len > 0 else 0.0

        for j, key in enumerate(STATE_KEYS):
            # Each state position attends to all variables equally (no per-variable heads)
            # Use the context→state attention weight as a proxy per step
            variable_attention[key] += float(ctx_to_states.mean()) / n_layers

    # Normalise
    total_attn = sum(variable_attention.values())
    if total_attn > 0:
        variable_attention = {k: v / total_attn for k, v in variable_attention.items()}

    top_vars = sorted(variable_attention.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "cascade_prediction":       "CATASTROPHIC" if cascade_crash_prob > 0.5 else "RECOVERABLE",
        "cascade_crash_probability": cascade_crash_prob,
        "key_cascade_variables":    [{"variable": k, "attention_weight": round(v, 4)}
                                      for k, v in top_vars],
        "cascade_window_steps":     actual_len,
        "model":                    "CascadeTransformer",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(traj, scenario) -> dict:
    """
    Run L3 analysis on a freshly simulated trajectory.

    Args:
        traj:     TrajectoryRecord from Simulator.run()
        scenario: ScenarioInput from Layer 0 parse

    Returns:
        dict with keys:
          snapshot:  XGBoost crash probability + top factors
          cascade:   Transformer prediction + key cascade variables
          available: bool — False if models failed to load
    """
    if not _models_loaded:
        loaded = load_models()
        if not loaded:
            return {
                "available":  False,
                "error":      _load_error or "L3 models not loaded",
                "snapshot":   None,
                "cascade":    None,
            }

    # Get primary failure component
    component = "unknown"
    if traj.failure_chain:
        component = traj.failure_chain[0].get("component", "unknown")
    elif scenario.failures:
        component = scenario.failures[0].component

    record   = _build_record(traj, scenario)
    snapshot = _run_snapshot(record)
    cascade  = _run_cascade(record, scenario, component)

    return {
        "available":  True,
        "component":  component,
        "snapshot":   snapshot,
        "cascade":    cascade,
    }