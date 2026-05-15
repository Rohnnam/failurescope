"""
FailureScope Layer 3 — Attention Extractor (v3)
=================================================
Fully spec-compliant rewrite per context1 [Layer 3: Relationship Extraction]
and context2 [Stage 6: Relationship Extraction].

SPEC REQUIREMENTS IMPLEMENTED:
  1. ttc_s > 300 filter — only trajectories with multi-step cascades
     (short-crash trajectories <300s cannot contain induced temporal patterns)
  2. Crash probability in kill zone — requires recoverable contrast sample
     "crash_probability_in_this_zone": 0.87 per context1 output format
  3. Multi-variable interaction rules — "voltage < 24V AND pitch > 8° AND
     fuel_flow < 3 GPH" as compound rule, not separate variables
  4. time_between_failures — context1 output: "time_between_failures: <2 min"
  5. Variable attention ranking — context2: voltage(0.8), pitch(0.7), ff(0.6)
  6. Context1 output format — induced_relationship, critical_state_combination,
     crash_probability_in_this_zone, recovery_probability_in_this_zone

OUTPUT: attention_analysis.json with:
  per_component: full analysis per failure type
  emergent_rules: ranked list in exact context1 format
  cross_component_patterns: shared voltage danger zones
  variable_importance_global: which state variables matter most overall

Run after training:
  python train_layer3.py --extract-only
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .dataset import (
    FAILURE_COMPONENTS,
    FAILURE_TO_IDX,
    MAX_SEQ_LEN,
    STATE_KEYS,
    TrajectoryDataset,
    TRAIN_SIZE,
)
from .model import FailureScopeTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAJECTORIES_PATH = Path(
    "/opt/failurescope_data/simulation_trajectories/trajectories.jsonl.gz"
)
MODEL_PATH  = Path("/opt/failurescope_data/models/trained_model.pt")
OUTPUT_PATH = Path("/opt/failurescope_data/models/attention_analysis.json")

# ── Sampling config ────────────────────────────────────────────────────────────

# Context2: "voltage 20-24V + rpm 400-800 + pitch 5-15° + altitude 1000-3000ft"
# These cascades require hundreds of seconds to develop.
# Short trajectories (<300s) crash before any multi-step cascade can form.
MIN_CASCADE_TTC_S = 300.0

# Electrically cascading components per PROGRESS.md §4 Layer 1 rules R001-R013
CASCADE_COMPONENTS = {
    "alternator", "battery", "voltage_regulator",
    "fuel_pump", "fuel_selector", "fuel_system",
    "engine_left", "engine_right", "engine_both",
    "carburetor", "magnetos",
}

N_PER_CASCADE  = 200   # catastrophic, ttc > 300
N_PER_OTHER    = 100   # catastrophic, ttc > 300, non-cascade components
N_RECOVERABLE  = 400   # recoverable contrast sample for kill zone probability

# Base rates from PROGRESS.md §4 Layer 2
BASE_RATE_CAT = 0.338
BASE_RATE_REC = 0.662

# Attention classification
TERMINAL_FRACTION = 0.80
ATTN_THRESHOLD    = 0.05

# Kill zone variable thresholds from context2 [What Gets Learned]
# voltage 20-24V + rpm 400-800 + pitch 5-15° + altitude 1000-3000ft
KILL_ZONE_SPEC = {
    "v":     (14.0, 24.0),   # danger below 24V (fuel pump degrades below 14V per R003)
    "rpm":   (400,  1800),   # danger in 400-1800 RPM
    "pitch": (5.0,  20.0),   # danger in 5-20° (recovery maneuver range)
    "alt":   (1000, 5000),   # danger in 1000-5000 ft
    "ff":    (0.0,  3.0),    # danger below 3 GPH (per context1 Pattern 4)
}


# ── Phase 1: Build index (catastrophic ttc>300 and recoverable) ────────────────

def _build_index(
    path: Path,
    val_start: int = TRAIN_SIZE,
) -> Tuple[Dict[str, List[Tuple[int, float]]], List[int]]:
    """
    Single pass over val set.
    Returns:
      cat_index  : {component: [(line_num, ttc_s), ...]}  only ttc_s > MIN_CASCADE_TTC_S
      rec_lines  : [line_num, ...]  recoverable trajectories (any length)

    Uses saved val_line_indices.pkl from preprocess_shards.py when available.
    With stratified split, val items are scattered throughout the gz file —
    not just lines >= TRAIN_SIZE. Falls back to line cutoff if pkl not found.
    """
    import pickle

    val_indices_path = Path("/opt/failurescope_data/shards/val_line_indices.pkl")
    if val_indices_path.exists():
        with open(val_indices_path, "rb") as f:
            val_line_set = pickle.load(f)
        print(f"[Index] Using saved val line indices ({len(val_line_set):,} lines)")
        use_line_set = True
    else:
        print(f"[Index] No val_line_indices.pkl — using lines >= {val_start:,}")
        val_line_set = None
        use_line_set = False

    cat_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    rec_lines: List[int] = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if use_line_set:
                if line_idx not in val_line_set:
                    continue
            else:
                if line_idx < val_start:
                    continue
            try:
                rec = json.loads(line)
                outcome = rec.get("outcome", "")

                if outcome == "catastrophic":
                    ttc_s = float(rec.get("ttc_s") or rec.get("duration_s") or 0)
                    if ttc_s < MIN_CASCADE_TTC_S:
                        continue
                    failures = rec.get("failures", [])
                    if not failures:
                        continue
                    comp = failures[0].get("component", "unknown")
                    cat_index[comp].append((line_idx, ttc_s))

                elif outcome == "recoverable":
                    rec_lines.append(line_idx)

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    print(f"[Index] Catastrophic (ttc>{MIN_CASCADE_TTC_S:.0f}s): "
          f"{sum(len(v) for v in cat_index.values()):,} across "
          f"{len(cat_index)} components")
    for comp in sorted(cat_index.keys()):
        n   = len(cat_index[comp])
        tag = "CASCADE" if comp in CASCADE_COMPONENTS else "other"
        print(f"  {comp:25s}  {n:5d}  [{tag}]")
    print(f"[Index] Recoverable (contrast): {len(rec_lines):,}")
    return dict(cat_index), rec_lines


def _stratified_sample(
    cat_index: Dict[str, List[Tuple[int, float]]],
    rec_lines: List[int],
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Sample cat per component + recoverable contrast. Return {comp: [lines]}, [lines]."""
    sampled_cat: Dict[str, List[int]] = {}
    for comp, entries in cat_index.items():
        n = N_PER_CASCADE if comp in CASCADE_COMPONENTS else N_PER_OTHER
        n = min(n, len(entries))
        chosen = rng.choice(len(entries), size=n, replace=False)
        sampled_cat[comp] = sorted([entries[i][0] for i in chosen])

    n_rec = min(N_RECOVERABLE, len(rec_lines))
    sampled_rec = sorted(rng.choice(rec_lines, size=n_rec, replace=False).tolist())
    return sampled_cat, sampled_rec


def _load_lines(
    path: Path,
    cat_line_sets: Dict[str, List[int]],
    rec_lines: List[int],
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Single gz scan to load all needed records."""
    needed_cat: Dict[int, str] = {}
    for comp, lines in cat_line_sets.items():
        for ln in lines:
            needed_cat[ln] = comp
    needed_rec = set(rec_lines)
    all_needed = set(needed_cat) | needed_rec

    if not all_needed:
        return {}, []

    max_line = max(all_needed)
    cat_records: Dict[str, List[dict]] = defaultdict(list)
    rec_records: List[dict] = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx > max_line:
                break
            if line_idx not in all_needed:
                continue
            try:
                rec = json.loads(line)
                if line_idx in needed_cat:
                    cat_records[needed_cat[line_idx]].append(rec)
                elif line_idx in needed_rec:
                    rec_records.append(rec)
            except (json.JSONDecodeError, ValueError):
                continue

    return dict(cat_records), rec_records


# ── Phase 2: Recoverable state pool for kill zone probability ──────────────────

def _build_recoverable_pool(rec_records: List[dict]) -> List[dict]:
    """
    Extract all state snapshots from recoverable trajectories.
    Used to compute: what fraction of recoverable trajectories
    pass through the kill zone state combination?
    """
    pool = []
    for rec in rec_records:
        for s in rec.get("states", []):
            pool.append({
                "v":        float(s.get("v", 14.0)),
                "rpm":      float(s.get("rpm", 2400.0)),
                "ff":       float(s.get("ff", 8.5)),
                "pitch":    float(s.get("pitch", 0.0)),
                "pump_eff": float(s.get("pump_eff", 1.0)),
                "alt":      float(s.get("alt", 5000.0)),
            })
    return pool


def _trajectories_in_zone(
    records: List[dict],
    zone: Dict[str, Tuple[float, float]],
) -> int:
    """Count trajectories where ANY state falls within ALL zone bounds simultaneously."""
    count = 0
    for rec in records:
        for s in rec.get("states", []):
            in_zone = all(
                lo <= float(s.get(var, lo - 1)) <= hi
                for var, (lo, hi) in zone.items()
            )
            if in_zone:
                count += 1
                break  # count trajectory, not individual states
    return count


def _compute_kill_zone_probability(
    cat_records: List[dict],
    rec_records: List[dict],
    zone: Dict[str, Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Bayesian kill zone crash probability per context1 [Layer 3]:
    "crash_probability_in_this_zone": 0.87

    Uses base rates from PROGRESS.md §4 to correct for oversampling.
    P(crash|zone) = f_cat * BASE_RATE_CAT /
                   (f_cat * BASE_RATE_CAT + f_rec * BASE_RATE_REC)
    """
    n_cat = max(len(cat_records), 1)
    n_rec = max(len(rec_records), 1)

    cat_in_zone = _trajectories_in_zone(cat_records, zone)
    rec_in_zone = _trajectories_in_zone(rec_records, zone)

    f_cat = cat_in_zone / n_cat
    f_rec = rec_in_zone / n_rec

    denom = f_cat * BASE_RATE_CAT + f_rec * BASE_RATE_REC
    if denom < 1e-9:
        return 0.5, 0.5  # no information

    crash_prob    = (f_cat * BASE_RATE_CAT) / denom
    recovery_prob = (f_rec * BASE_RATE_REC) / denom
    return round(crash_prob, 3), round(recovery_prob, 3)


# ── Phase 3: Cascade event timing ─────────────────────────────────────────────

def _find_cascade_timestamps(record: dict) -> Dict[str, float]:
    """Find when each failure first appeared in active failures list."""
    first_t: Dict[str, float] = {}
    for s in record.get("states", []):
        t = s.get("t", 0.0)
        for f in s.get("failures", []):
            if f not in first_t:
                first_t[f] = t
    return first_t


def _time_between_failures(cascade_events: Dict[str, float], inject_t: float) -> Optional[str]:
    """
    Per context1 output: "time_between_failures": "<2 minutes"
    Returns human-readable string of time from injection to first cascade event.
    """
    post_inject = {k: v for k, v in cascade_events.items() if v > inject_t}
    if not post_inject:
        return None
    first_cascade = min(post_inject.values())
    delta = first_cascade - inject_t
    if delta < 60:
        return f"<1 minute ({delta:.0f}s)"
    elif delta < 120:
        return f"<2 minutes ({delta:.0f}s)"
    elif delta < 300:
        return f"{delta/60:.0f}-{(delta+60)/60:.0f} minutes ({delta:.0f}s)"
    else:
        return f"{delta/60:.1f} minutes ({delta:.0f}s)"


def _classify_timing(
    attn_t: float,
    first_cascade_t: Optional[float],
    ttc_s: float,
) -> str:
    """EARLY_CASCADE: before first cascade (predictive). TERMINAL: last 20%."""
    if first_cascade_t is not None and attn_t < first_cascade_t:
        return "EARLY_CASCADE"
    if attn_t >= ttc_s * TERMINAL_FRACTION:
        return "TERMINAL"
    return "MID_CASCADE"


# ── Phase 4: Per-trajectory attention analysis ─────────────────────────────────

def _analyse_one(
    model: FailureScopeTransformer,
    record: dict,
    device: torch.device,
    ds: TrajectoryDataset,
) -> Optional[dict]:
    """
    Run one trajectory through the trained model.
    Extracts:
      - Per-variable attention ranking (context2: voltage=0.8, pitch=0.7, ff=0.6)
      - Multi-variable state combination at peak attention
      - Cascade timing (time_between_failures)
      - Attention timing classification (EARLY/MID/TERMINAL)
    """
    result = ds._process_record(record)
    if result is None:
        return None

    features, pad_mask, _ = result
    features  = features.unsqueeze(0).to(device)
    pad_mask  = pad_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_list = model(features, pad_mask, return_attention=True)

    prob_cat = logits.softmax(dim=-1)[0, 1].item()
    real_len = int(pad_mask[0].sum().item())
    states   = record.get("states", [])
    ttc_s    = float(record.get("ttc_s") or real_len)

    # Per-layer, per-head CLS attention → average = (real_len,)
    all_head_attns = []
    for lw in attn_list:
        cls_row = lw[0, :, 0, 1:real_len + 1].numpy()   # (nhead, real_len)
        all_head_attns.append(cls_row)
    all_head_attns = np.array(all_head_attns)            # (n_layers, nhead, real_len)
    avg_attn = all_head_attns.mean(axis=(0, 1))          # (real_len,)

    if len(avg_attn) == 0:
        return None

    # Cascade timing
    cascade_ts  = _find_cascade_timestamps(record)
    failures    = record.get("failures", [])
    primary     = failures[0]["component"] if failures else "unknown"
    inject_t    = float(failures[0].get("trigger_time_s", 0.0)) if failures else 0.0
    cascade_events = {k: v for k, v in cascade_ts.items() if v > inject_t}
    first_cascade_t = min(cascade_events.values()) if cascade_events else None
    tbf = _time_between_failures(cascade_events, inject_t)

    # High-attention timesteps
    max_a     = float(avg_attn.max())
    threshold = max_a * ATTN_THRESHOLD
    high_idx  = np.where(avg_attn >= threshold)[0]

    peak_idx = int(avg_attn.argmax())
    peak_t   = states[peak_idx]["t"] if peak_idx < len(states) else 0.0
    peak_cls = _classify_timing(peak_t, first_cascade_t, ttc_s)

    # Collect full state vectors at high-attention non-terminal moments
    critical_states = []
    timing_classes  = []
    for idx in sorted(high_idx, key=lambda i: -avg_attn[i])[:20]:
        if idx >= len(states):
            continue
        s  = states[idx]
        t  = float(s.get("t", 0.0))
        tc = _classify_timing(t, first_cascade_t, ttc_s)
        timing_classes.append(tc)
        critical_states.append({
            "t":         t,
            "attn_w":    float(avg_attn[idx]),
            "timing_cls": tc,
            "alt":       float(s.get("alt", 0.0)),
            "v":         float(s.get("v", 14.0)),
            "rpm":       float(s.get("rpm", 0.0)),
            "ff":        float(s.get("ff", 0.0)),
            "pump_eff":  float(s.get("pump_eff", 1.0)),
            "pitch":     float(s.get("pitch", 0.0)),
            "roll":      float(s.get("roll", 0.0)),
            "vs":        float(s.get("vs", 0.0)),
        })

    cnt   = Counter(timing_classes)
    total = max(len(timing_classes), 1)
    timing_fractions = {
        "EARLY_CASCADE": cnt.get("EARLY_CASCADE", 0) / total,
        "MID_CASCADE":   cnt.get("MID_CASCADE",   0) / total,
        "TERMINAL":      cnt.get("TERMINAL",       0) / total,
    }

    # ── Variable attention ranking (context2 spec) ─────────────────────────────
    # For each state variable, compute mean attention at timesteps where
    # that variable is in its "danger zone". Higher = model attends when
    # that variable is in danger. This gives the ranked list:
    # voltage(0.8), pitch_angle(0.7), fuel_flow(0.6), time_since_failure(0.5)
    var_ranking = _compute_variable_attention_ranking(
        avg_attn, states, real_len, cascade_events, inject_t
    )

    return {
        "scenario_id":        record.get("scenario_id"),
        "primary_failure":    primary,
        "ttc_s":              ttc_s,
        "prob_cat":           prob_cat,
        "inject_t":           inject_t,
        "first_cascade_t":    first_cascade_t,
        "cascade_events":     cascade_events,
        "time_between_failures": tbf,
        "peak_attn_t":        peak_t,
        "peak_timing_cls":    peak_cls,
        "timing_fractions":   timing_fractions,
        "critical_states":    critical_states,
        "var_ranking":        var_ranking,
    }


def _compute_variable_attention_ranking(
    avg_attn: np.ndarray,
    states: list,
    real_len: int,
    cascade_events: Dict[str, float],
    inject_t: float,
) -> List[dict]:
    """
    Per context2: rank state variables by mean CLS attention weight when in danger zone.
    Returns list of {variable, mean_attention, danger_zone} sorted descending.
    This produces the spec output: voltage(0.8), pitch(0.7), fuel_flow(0.6)...
    """
    if real_len == 0 or not states:
        return []

    # time_since_failure is derived (not a direct state key)
    first_cascade_t = min(cascade_events.values()) if cascade_events else None

    results = []
    for var, (lo, hi) in KILL_ZONE_SPEC.items():
        # Find timesteps where variable is in danger zone
        danger_attns = []
        for i in range(min(real_len, len(states))):
            val = float(states[i].get(var, 0.0))
            if lo <= val <= hi:
                danger_attns.append(float(avg_attn[i]) if i < len(avg_attn) else 0.0)
        if danger_attns:
            results.append({
                "variable":      var,
                "mean_attention": round(float(np.mean(danger_attns)), 3),
                "danger_zone":   [lo, hi],
                "n_steps_in_zone": len(danger_attns),
            })

    # Also add time_since_failure as a derived variable
    if first_cascade_t is not None:
        post_cascade_attns = [
            float(avg_attn[i]) if i < len(avg_attn) else 0.0
            for i in range(min(real_len, len(states)))
            if float(states[i].get("t", 0.0)) >= first_cascade_t
        ]
        if post_cascade_attns:
            results.append({
                "variable":      "time_since_failure",
                "mean_attention": round(float(np.mean(post_cascade_attns)), 3),
                "danger_zone":   [0, 300],
                "n_steps_in_zone": len(post_cascade_attns),
            })

    return sorted(results, key=lambda x: -x["mean_attention"])


# ── Phase 5: Component aggregation ────────────────────────────────────────────

def _aggregate_component(
    analyses: List[dict],
    component: str,
    cat_records: List[dict],
    rec_records: List[dict],
) -> dict:
    """
    Full aggregation per component.
    Produces context1-format induced_rule with all required fields.
    """
    n = len(analyses)
    if n == 0:
        return {}

    # ── Timing distribution ────────────────────────────────────────────────────
    timing_agg: Dict[str, list] = {"EARLY_CASCADE": [], "MID_CASCADE": [], "TERMINAL": []}
    for a in analyses:
        for cls, frac in a["timing_fractions"].items():
            timing_agg[cls].append(frac)
    timing_summary = {cls: round(float(np.mean(v)), 3) for cls, v in timing_agg.items()}

    # ── Kill zone: state values at NON-TERMINAL high-attention moments ─────────
    induced_states = [
        s
        for a in analyses
        for s in a["critical_states"]
        if s["timing_cls"] in ("EARLY_CASCADE", "MID_CASCADE")
    ]

    kill_zone_ranges: dict = {}
    if induced_states:
        for var in ("v", "pump_eff", "pitch", "alt", "rpm", "ff", "vs"):
            vals = [s[var] for s in induced_states]
            kill_zone_ranges[var] = {
                "p25": round(float(np.percentile(vals, 25)), 2),
                "p75": round(float(np.percentile(vals, 75)), 2),
                "mean": round(float(np.mean(vals)), 2),
                "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 2),
            }
        times = [s["t"] for s in induced_states]
        kill_zone_ranges["time_window_s"] = {
            "p25": round(float(np.percentile(times, 25)), 0),
            "p75": round(float(np.percentile(times, 75)), 0),
        }

    # ── Kill zone probability (context1: crash_probability_in_this_zone) ──────
    crash_prob    = None
    recovery_prob = None
    if kill_zone_ranges and "v" in kill_zone_ranges:
        zone_bounds = {
            var: (kill_zone_ranges[var]["p25"], kill_zone_ranges[var]["p75"])
            for var in ("v", "pitch", "ff")
            if var in kill_zone_ranges
            and kill_zone_ranges[var]["iqr"] > 0  # skip constant variables
        }
        if zone_bounds:
            crash_prob, recovery_prob = _compute_kill_zone_probability(
                cat_records, rec_records, zone_bounds
            )

    # ── Variable attention ranking (context2 spec) ────────────────────────────
    var_vote: Dict[str, list] = defaultdict(list)
    for a in analyses:
        for entry in a.get("var_ranking", []):
            var_vote[entry["variable"]].append(entry["mean_attention"])

    var_ranking_agg = sorted(
        [
            {
                "variable":      var,
                "mean_attention": round(float(np.mean(scores)), 3),
            }
            for var, scores in var_vote.items()
        ],
        key=lambda x: -x["mean_attention"],
    )

    # ── Cascade timing ─────────────────────────────────────────────────────────
    tbf_values = [a["time_between_failures"] for a in analyses if a["time_between_failures"]]
    most_common_tbf = Counter(tbf_values).most_common(1)[0][0] if tbf_values else None

    cascade_events_seen: Counter = Counter()
    for a in analyses:
        for evt in a.get("cascade_events", {}):
            cascade_events_seen[evt] += 1

    first_cascade_times = [
        a["first_cascade_t"] for a in analyses if a["first_cascade_t"] is not None
    ]

    # ── Quality score ──────────────────────────────────────────────────────────
    early_mid  = timing_summary["EARLY_CASCADE"] + timing_summary["MID_CASCADE"]
    iqr_tight  = 0.0
    if kill_zone_ranges:
        v_iqr     = kill_zone_ranges.get("v",        {}).get("iqr", 99)
        p_iqr     = kill_zone_ranges.get("pump_eff", {}).get("iqr", 99)
        iqr_tight = max(0.0, 1.0 - v_iqr / 14.0) * 0.5 + max(0.0, 1.0 - p_iqr) * 0.5

    quality_score = round(early_mid * 0.6 + iqr_tight * 0.4, 3)

    # ── Build context1-format induced rule ─────────────────────────────────────
    induced_rule = _build_context1_rule(
        component, kill_zone_ranges, timing_summary, var_ranking_agg,
        crash_prob, recovery_prob, most_common_tbf,
        cascade_events_seen, quality_score, n,
    )

    return {
        "n_trajectories":              n,
        "n_min_cascade_ttc":           n,  # all had ttc > MIN_CASCADE_TTC_S
        "is_cascade_component":        component in CASCADE_COMPONENTS,
        "timing_fractions":            timing_summary,
        "quality_score":               quality_score,
        "variable_attention_ranking":  var_ranking_agg,
        "first_cascade_t_median_s":    round(float(np.median(first_cascade_times)), 1)
                                       if first_cascade_times else None,
        "most_common_time_between_failures": most_common_tbf,
        "most_common_cascade_events":  [
            {"event": k, "fraction": round(v / n, 2)}
            for k, v in cascade_events_seen.most_common(5)
        ],
        "kill_zone":                   kill_zone_ranges,
        "crash_probability_in_zone":   crash_prob,
        "recovery_probability_in_zone": recovery_prob,
        "induced_rule":                induced_rule,
    }


def _build_context1_rule(
    component: str,
    kill_zone: dict,
    timing: dict,
    var_ranking: List[dict],
    crash_prob: Optional[float],
    recovery_prob: Optional[float],
    tbf: Optional[str],
    cascade_events: Counter,
    quality: float,
    n: int,
) -> dict:
    """
    Produce the exact output format from context1 [Layer 3: Relationship Extraction]:

      {
        "induced_relationship": "pitch_angle + voltage_drop timing interaction",
        "critical_state_combination": {
          "voltage_range": [20, 24],
          "pitch_rate_range": [5, 20],
          "fuel_flow_range": [0, 3],
          "time_between_failures": "<2 minutes"
        },
        "crash_probability_in_this_zone": 0.87,
        "recovery_probability_in_this_zone": 0.13,
        "interpretation": "..."
      }
    """
    if not kill_zone:
        return {"status": "INSUFFICIENT_DATA — no non-terminal high-attention states found"}

    v   = kill_zone.get("v",        {})
    p   = kill_zone.get("pump_eff", {})
    pt  = kill_zone.get("pitch",    {})
    a   = kill_zone.get("alt",      {})
    ff  = kill_zone.get("ff",       {})
    t   = kill_zone.get("time_window_s", {})

    early_mid = timing["EARLY_CASCADE"] + timing["MID_CASCADE"]

    # Build induced_relationship name from top 2 variables
    top_vars = [e["variable"] for e in var_ranking[:2]] if var_ranking else ["voltage"]
    induced_rel = " + ".join(top_vars) + " interaction"

    # Build critical_state_combination
    critical_combo = {}
    if v:
        critical_combo["voltage_range_V"] = [v.get("p25"), v.get("p75")]
    if pt:
        critical_combo["pitch_range_deg"] = [pt.get("p25"), pt.get("p75")]
    if ff:
        critical_combo["fuel_flow_range_gph"] = [ff.get("p25"), ff.get("p75")]
    if a:
        critical_combo["altitude_range_ft"] = [a.get("p25"), a.get("p75")]
    if tbf:
        critical_combo["time_between_failures"] = tbf

    # Interpretation per context1 style
    if early_mid > 0.5:
        attn_char = "INDUCED — model attends to precursor states BEFORE cascade is obvious"
    elif early_mid > 0.25:
        attn_char = "PARTIAL — mixed precursor and reactive attention"
    else:
        attn_char = "REACTIVE — model attends primarily to post-cascade terminal states"

    top_var_str = ", ".join(
        f"{e['variable']}(attn={e['mean_attention']:.2f})"
        for e in var_ranking[:3]
    ) if var_ranking else "unknown"

    interp = (
        f"For '{component}' failures: when "
        f"voltage={v.get('p25','?')}–{v.get('p75','?')} V, "
        f"pitch={pt.get('p25','?')}°–{pt.get('p75','?')}°, "
        f"fuel_flow={ff.get('p25','?')}–{ff.get('p75','?')} GPH, "
        f"altitude={a.get('p25','?')}–{a.get('p75','?')} ft "
        f"(window t={t.get('p25','?')}–{t.get('p75','?')} s): "
    )
    if crash_prob is not None:
        interp += (
            f"crash probability is {crash_prob:.0%}. "
            f"Top attention variables: {top_var_str}. "
            f"Attention character: {attn_char}. "
            f"n={n} trajectories (all ttc>{MIN_CASCADE_TTC_S:.0f}s)."
        )
    else:
        interp += (
            f"Attention character: {attn_char}. "
            f"Top attention variables: {top_var_str}. "
            f"n={n} trajectories (all ttc>{MIN_CASCADE_TTC_S:.0f}s)."
        )

    return {
        "induced_relationship":           induced_rel,
        "primary_failure":                component,
        "critical_state_combination":     critical_combo,
        "crash_probability_in_this_zone": crash_prob,
        "recovery_probability_in_this_zone": recovery_prob,
        "interpretation":                 interp,
        "attention_quality":              attn_char,
        "quality_score":                  quality,
        "top_attention_variables":        var_ranking[:5],
        "most_common_cascade_events":     [k for k, _ in cascade_events.most_common(3)],
        "validation_status":              "PENDING — Method 2 (physics) + Method 3 (NTSB)",
        "source":                         "attention_weight_analysis_v3",
    }


# ── Phase 6: Cross-component patterns ─────────────────────────────────────────

def _cross_component_patterns(per_component_results: dict) -> List[dict]:
    """
    Find voltage danger zones shared across multiple failure components.
    Per context1: structural aircraft vulnerability, not component-specific.
    """
    patterns = []
    voltage_ranges = {}
    for comp, res in per_component_results.items():
        kz = res.get("kill_zone", {})
        v  = kz.get("v", {})
        if v.get("p25") is not None and v.get("p75") is not None:
            voltage_ranges[comp] = (v["p25"], v["p75"])

    comps = list(voltage_ranges.keys())
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            c1, c2    = comps[i], comps[j]
            lo1, hi1  = voltage_ranges[c1]
            lo2, hi2  = voltage_ranges[c2]
            overlap_lo = max(lo1, lo2)
            overlap_hi = min(hi1, hi2)
            if overlap_hi > overlap_lo and (overlap_hi - overlap_lo) < 3.0:
                patterns.append({
                    "components":            [c1, c2],
                    "shared_voltage_zone_V": [round(overlap_lo, 1), round(overlap_hi, 1)],
                    "interpretation": (
                        f"'{c1}' and '{c2}' both produce dangerous state combinations "
                        f"in voltage {overlap_lo:.1f}–{overlap_hi:.1f} V. "
                        "This voltage band is structurally dangerous regardless of "
                        "which component initiates the failure — it represents a "
                        "shared electrical system vulnerability per context1 [Layer 4]."
                    ),
                    "validation_status": "PENDING",
                })

    return sorted(patterns, key=lambda p: p["shared_voltage_zone_V"][0])


# ── Global variable importance ─────────────────────────────────────────────────

def _global_variable_importance(per_component_results: dict) -> List[dict]:
    """
    Aggregate variable attention rankings across all components.
    Produces the global ranking: which state variables matter most
    for failure prediction across all aircraft system failures?
    """
    var_scores: Dict[str, list] = defaultdict(list)
    for res in per_component_results.values():
        for entry in res.get("variable_attention_ranking", []):
            var_scores[entry["variable"]].append(entry["mean_attention"])

    return sorted(
        [
            {
                "variable":       var,
                "mean_attention": round(float(np.mean(scores)), 3),
                "n_components":   len(scores),
            }
            for var, scores in var_scores.items()
        ],
        key=lambda x: -x["mean_attention"],
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def extract_attention(
    model_path: Path = MODEL_PATH,
    trajectories_path: Path = TRAJECTORIES_PATH,
    output_path: Path = OUTPUT_PATH,
    n_per_cascade: int = N_PER_CASCADE,
    n_per_other: int   = N_PER_OTHER,
    seed: int = 42,
) -> dict:
    """
    Full stratified attention extraction per context1 [Layer 3] spec.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_layer3.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(seed)

    # Load model
    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt["model_config"])
    model_cfg["use_gradient_checkpointing"] = False

    model = FailureScopeTransformer(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"[Extractor v3] Loaded  val_acc={ckpt['val_acc']:.4f} "
        f"epoch={ckpt['epoch']}  params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"[Extractor v3] Min cascade TTC filter: {MIN_CASCADE_TTC_S:.0f}s")

    # Phase 1: Build index
    print("\n[Phase 1] Indexing val set (catastrophic ttc>300 + recoverable) …")
    cat_index, rec_lines = _build_index(trajectories_path)

    if not cat_index:
        raise RuntimeError(
            f"No catastrophic trajectories with ttc_s > {MIN_CASCADE_TTC_S}s found in val set."
        )

    # Phase 2: Stratified sample
    print(f"\n[Phase 2] Stratified sampling …")
    sampled_cat, sampled_rec = _stratified_sample(cat_index, rec_lines, rng)
    total_cat = sum(len(v) for v in sampled_cat.values())
    print(f"  Catastrophic : {total_cat} across {len(sampled_cat)} components")
    print(f"  Recoverable  : {len(sampled_rec)} (contrast sample)")

    # Phase 3: Load records (single scan)
    print(f"\n[Phase 3] Loading records (single gz scan) …")
    cat_records_by_comp, rec_records = _load_lines(
        trajectories_path, sampled_cat, sampled_rec
    )
    print(f"  Loaded {sum(len(v) for v in cat_records_by_comp.values())} catastrophic "
          f"+ {len(rec_records)} recoverable records")

    # Phase 4: Attention analysis per component
    ds = TrajectoryDataset.__new__(TrajectoryDataset)
    ds.max_seq_len = model_cfg["max_seq_len"]

    per_component_results: Dict[str, dict] = {}

    for comp, recs in sorted(cat_records_by_comp.items()):
        print(f"\n[Phase 4] {comp} ({len(recs)} trajectories, ttc>{MIN_CASCADE_TTC_S:.0f}s) …")
        analyses = []
        for i, rec in enumerate(recs):
            res = _analyse_one(model, rec, device, ds)
            if res:
                analyses.append(res)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(recs)}")

        if analyses:
            agg = _aggregate_component(
                analyses, comp,
                cat_records_by_comp.get(comp, []),
                rec_records,
            )
            per_component_results[comp] = agg
            tf = agg.get("timing_fractions", {})
            cp = agg.get("crash_probability_in_zone")
            print(
                f"  quality={agg.get('quality_score',0):.3f}  "
                f"EARLY={tf.get('EARLY_CASCADE',0):.2f}  "
                f"MID={tf.get('MID_CASCADE',0):.2f}  "
                f"TERMINAL={tf.get('TERMINAL',0):.2f}  "
                f"kill_zone_crash_prob={cp}"
            )

    # Phase 5: Cross-component + global importance
    print("\n[Phase 5] Cross-component pattern mining …")
    cross_patterns = _cross_component_patterns(per_component_results)
    global_var_importance = _global_variable_importance(per_component_results)

    print(f"  {len(cross_patterns)} shared voltage-zone patterns")
    print(f"\n  Global variable importance ranking:")
    for entry in global_var_importance[:5]:
        print(f"    {entry['variable']:20s}  attn={entry['mean_attention']:.3f}  "
              f"across {entry['n_components']} components")

    # Build emergent rules (quality > 0.1, sorted by score)
    emergent_rules = [
        res["induced_rule"]
        for res in per_component_results.values()
        if res.get("induced_rule", {}).get("quality_score", 0) > 0.1
    ]
    emergent_rules.sort(key=lambda r: -(r.get("quality_score") or 0))

    # Component index summary
    component_index_summary = {
        comp: {
            "total_long_cascade_in_val": len(entries),
            "sampled": len(sampled_cat.get(comp, [])),
        }
        for comp, entries in cat_index.items()
    }

    output = {
        "meta": {
            "model_val_acc":          ckpt["val_acc"],
            "model_epoch":            ckpt["epoch"],
            "extractor_version":      "v3",
            "min_cascade_ttc_s":      MIN_CASCADE_TTC_S,
            "total_cat_analysed":     sum(len(v) for v in cat_records_by_comp.values()),
            "total_rec_contrast":     len(rec_records),
            "n_components":           len(per_component_results),
            "n_emergent_rules":       len(emergent_rules),
            "n_cross_patterns":       len(cross_patterns),
            "layer3_status":          "complete",
            "next_step":              "Layer 4 — Failure Mode Discovery + NTSB cross-reference",
        },
        "component_index":             component_index_summary,
        "variable_importance_global":  global_var_importance,
        "per_component":               per_component_results,
        "cross_component_patterns":    cross_patterns,
        "emergent_rules":              emergent_rules,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Extractor v3] ✓ Saved → {output_path}")
    print(f"  {len(emergent_rules)} emergent rules  |  "
          f"{len(cross_patterns)} cross-component patterns  |  "
          f"{len(per_component_results)} components")

    print("\n── Top emergent rules by quality score ──")
    for r in emergent_rules[:6]:
        cp = r.get("crash_probability_in_this_zone")
        cp_str = f"  crash_prob={cp:.2f}" if cp is not None else ""
        print(f"  {r['primary_failure']:25s}  quality={r['quality_score']:.3f}"
              f"{cp_str}  [{r.get('attention_quality','')[:35]}]")

    return output