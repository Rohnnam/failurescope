"""
FailureScope Layer 3 — Shared Constants
=========================================
Single source of truth for all constants used across Layer 3.
All other Layer 3 files import from here.

Data facts (confirmed from diagnostic runs):
  - 500K total trajectories: 168,982 catastrophic (33.8%), 331,018 recoverable (66.2%)
  - Cascade duration (ttc_s - inject_t): mean=64s, median=33s, max=3416s
  - cascade > 30s:  97.1% of catastrophic (~163K)
  - cascade > 100s:  1.5% of catastrophic (~2.5K) — slow multi-step cascades
  - inject_t distribution: 63.1% injected at t>300s (late injection)
  - Sequence lengths in shards: 9-400 steps (after cascade window extraction)

Ground truth anchor (per PROGRESS.md §4 Layer 1):
  14 cascade rules (R001–R014) from POH + FAA documents.
  Induced relationships must be cross-referenced against these rules.
  Relationships consistent with known rules → CONFIRMED
  Relationships inconsistent → NOVEL_CANDIDATE (requires Layer 4 validation)
  Relationships not covered → INDUCED (primary discovery target)
"""

from typing import List, Dict

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAJECTORIES_PATH_STR = (
    "/opt/failurescope_data/simulation_trajectories/trajectories.jsonl.gz"
)
DATA_ROOT = "/opt/failurescope_data"
LAYER3_MODELS_DIR = "/opt/failurescope_data/models"

# ── State variable keys (from Layer 2 TrajectoryRecord) ───────────────────────

STATE_KEYS: List[str] = [
    "alt",      # altitude MSL (ft)      — range: 0–16500
    "spd",      # airspeed (knots)       — range: 0–200
    "v",        # electrical voltage (V) — range: 0–32  ← KEY cascade variable
    "rpm",      # engine RPM             — range: 0–3000
    "ff",       # fuel flow (GPH)        — range: 0–15  ← KEY cascade variable
    "fuel_l",   # left tank (gal)        — range: 0–50
    "fuel_r",   # right tank (gal)       — range: 0–50
    "bat",      # battery charge (Ah)    — range: 0–33
    "pump_eff", # fuel pump efficiency   — range: 0–1   ← KEY cascade variable
    "pitch",    # pitch angle (deg)      — range: -45–45 ← KEY recovery maneuver
    "roll",     # roll angle (deg)       — range: -90–90
    "vs",       # vertical speed (FPM)   — range: -2000–2000
    "t",        # time (seconds)         — range: 0–3600
]
N_STATE_KEYS = len(STATE_KEYS)  # 13

# Physical normalisation scales — known ranges, not learned statistics
# Enables stable normalisation at inference without a separate stats pass
STATE_SCALES: List[float] = [
    16500.0,  # alt
    200.0,    # spd
    32.0,     # v
    3000.0,   # rpm
    15.0,     # ff
    50.0,     # fuel_l
    50.0,     # fuel_r
    33.0,     # bat
    1.0,      # pump_eff
    45.0,     # pitch
    90.0,     # roll
    2000.0,   # vs
    3600.0,   # t
]

# ── Failure components (from Layer 0 FailureComponent enum) ───────────────────
# Per PROGRESS.md §4 Layer 0 — 23 components exactly

FAILURE_COMPONENTS: List[str] = [
    "engine_left", "engine_right", "engine_both",
    "alternator", "battery", "voltage_regulator",
    "fuel_pump", "fuel_selector", "fuel_system",
    "fuel_tank_left", "fuel_tank_right",
    "carburetor", "magnetos",
    "vacuum_pump", "gyroscope", "attitude_indicator",
    "flaps", "trim_system", "hydraulics",
    "landing_gear", "rudder", "ailerons", "elevator",
]
N_FAILURE_COMPONENTS = len(FAILURE_COMPONENTS)   # 23
FAILURE_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FAILURE_COMPONENTS)}

# Electrically-cascading components (R001–R013 in PROGRESS.md §4 Layer 1)
CASCADE_COMPONENTS = frozenset({
    "alternator", "battery", "voltage_regulator",
    "fuel_pump", "fuel_selector", "fuel_system",
    "engine_left", "engine_right", "engine_both",
    "carburetor", "magnetos",
})

# ── Layer 1 constraint rules (ground truth anchor) ────────────────────────────
# Per PROGRESS.md §4 Layer 1 — used by relationship_miner to cross-reference
# discovered patterns against known rules.

L1_RULES: List[Dict] = [
    {"id": "R001", "trigger": "alternator",        "effect": "voltage→12V",          "source": "POH §7.1"},
    {"id": "R002", "trigger": "battery_discharge",  "effect": "voltage_decay",         "source": "Service Manual D2006R4-13"},
    {"id": "R003", "trigger": "voltage<11.5V",      "effect": "pump_eff−20%/V",        "source": "FAA AC 43.13-1B"},
    {"id": "R004", "trigger": "voltage<10.5V",      "effect": "pump_fail",             "source": "FAA AC 43.13-1B"},
    {"id": "R005", "trigger": "pump_fail",          "effect": "engine_fuel_loss",      "source": "POH §7.1"},
    {"id": "R006", "trigger": "engine_fail",        "effect": "alternator_stop",       "source": "POH §7.1"},
    {"id": "R007", "trigger": "vacuum_pump_fail",   "effect": "gyros_fail",            "source": "POH §7.3"},
    {"id": "R008", "trigger": "engine_fail",        "effect": "vacuum_stop",           "source": "POH §7.3"},
    {"id": "R009", "trigger": "pump_degraded",      "effect": "engine_power_degraded", "source": "Continental O-470 Manual"},
    {"id": "R010", "trigger": "voltage<10V",        "effect": "flaps_trim_fail",       "source": "FAA AC 23.1309-1A"},
    {"id": "R011", "trigger": "carburetor_fail",    "effect": "rpm_loss_gradual",      "source": "FAA AC 61-23C"},
    {"id": "R012", "trigger": "fuel_selector_empty","effect": "fuel_starvation",       "source": "POH §4.2"},
    {"id": "R013", "trigger": "voltage_reg_fail",   "effect": "same_as_R001",          "source": "POH §7.1"},
    {"id": "R014", "trigger": "vacuum_fail+imc",    "effect": "spatial_disorientation","source": "FAA AC 61-27C"},
]

# ── Pilot / weather encodings ──────────────────────────────────────────────────

SKILL_LEVELS = ["student", "private", "commercial", "instrument", "airline"]
SKILL_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(SKILL_LEVELS)}

WEATHER_TYPES = ["vmc", "imc", "night_vmc", "vmc_turbulent"]
WEATHER_TO_IDX: Dict[str, int] = {w: i for i, w in enumerate(WEATHER_TYPES)}

# ── Dataset split ──────────────────────────────────────────────────────────────

TOTAL_TRAJECTORIES = 500_000
TRAIN_FRACTION     = 0.80

# ── Cascade window parameters ──────────────────────────────────────────────────

# Minimum cascade duration to include in Transformer training.
# cascade_duration = ttc_s - inject_t
# Data is at DT=0.1s. At 5s minimum = 50 timesteps of cascade data.
# This keeps ~96% of catastrophic trajectories.
# (Old value was 30.0s, designed for DT=1.0s data — too restrictive at DT=0.1s)
MIN_CASCADE_DURATION_S = 5.0

# Maximum cascade window length in steps.
# At DT=0.1s: 400 steps = 40 seconds of cascade window.
# Covers median cascade (33s) + room for slow cascades.
# Stored in shards; attention extractor also uses this cap.
CASCADE_MAX_STEPS = 400

# ── Snapshot features ──────────────────────────────────────────────────────────

# Features extracted from state at inject_t + init dict
# These are the "hole alignment" features for swiss cheese analysis
SNAPSHOT_FEATURE_NAMES: List[str] = [
    "alt_at_inject",       # altitude at failure injection
    "spd_at_inject",       # airspeed
    "v_at_inject",         # voltage
    "rpm_at_inject",       # engine RPM
    "ff_at_inject",        # fuel flow
    "fuel_total_at_inject",# left + right fuel
    "pump_eff_at_inject",  # pump efficiency (pre-existing degradation?)
    "pitch_at_inject",     # pitch angle
    "vs_at_inject",        # vertical speed
    "fatigue",             # pilot fatigue (from init)
    "skill_encoded",       # pilot skill level (0–4)
    "weather_encoded",     # weather (0–3)
    "failure_encoded",     # failure component (0–22)
    "inject_time_frac",    # inject_t / 3600 (how late in flight)
    "bat_at_inject",       # battery charge at injection
    "roll_at_inject",      # roll angle
]
N_SNAPSHOT_FEATURES = len(SNAPSHOT_FEATURE_NAMES)  # 16

# ── Cascade Transformer feature dimension ─────────────────────────────────────
# 13 STATE_KEYS only — NO failure one-hot
# Failure type is provided as a separate context token, not broadcast across sequence
# This forces the model to learn from state variable CHANGES, not failure identity
CASCADE_FEATURE_DIM = N_STATE_KEYS   # 13

# Context token dimension: encodes "hole alignment" at inject_t
# [alt_norm, fatigue, skill_norm, weather_norm] + failure one-hot (23) = 27
CASCADE_CONTEXT_DIM = 4 + N_FAILURE_COMPONENTS   # = 27

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 CONSTANTS — appended by Layer 4 development
# Do NOT duplicate these in any layer4/ file. Import from here.
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 4 output paths ──────────────────────────────────────────────────────

NTSB_CACHE_PATH         = "/opt/failurescope_data/ntsb_cache.json"
NTSB_CSV_PATH           = "/opt/failurescope_data/ntsb_aviation_data.csv"
NTSB_MDB_PATH           = "/opt/failurescope_data/avall.mdb"
VALIDATED_FINDINGS_PATH = "/opt/failurescope_data/models/validated_findings.json"
PHYSICS_RESULTS_PATH    = "/opt/failurescope_data/models/physics_results.json"
NTSB_RESULTS_PATH       = "/opt/failurescope_data/models/ntsb_results.json"

# ── C182 physical envelope ────────────────────────────────────────────────────
# Used by physics_validator.py to check kill zone plausibility.
# All values from Cessna 182 POH + Continental O-470 Service Manual.

C182_PHYSICS_BOUNDS: Dict[str, Dict] = {
    "alt":      {"min": 0.0,    "max": 16500.0, "source": "POH §5.1 Service Ceiling"},
    "v":        {"min": 0.0,    "max": 32.0,    "source": "Service Manual D2006R4-13"},
    "rpm":      {"min": 0.0,    "max": 2550.0,  "source": "Continental O-470 Manual §3.2"},
    "pitch":    {"min": -45.0,  "max": 45.0,    "source": "C182 flight envelope"},
    "roll":     {"min": -90.0,  "max": 90.0,    "source": "C182 flight envelope"},
    "vs":       {"min":-2000.0, "max": 2000.0,  "source": "C182 flight envelope"},
    "spd":      {"min": 0.0,    "max": 200.0,   "source": "C182 Vne=158kts + margin"},
    "fatigue":  {"min": 0.0,    "max": 1.0,     "source": "Simulation parameter [0,1]"},
    "fuel_l":   {"min": 0.0,    "max": 50.0,    "source": "POH §4.2 per-tank capacity"},
    "fuel_r":   {"min": 0.0,    "max": 50.0,    "source": "POH §4.2 per-tank capacity"},
}

# Voltage consistency thresholds — from L1 rules, used as cross-check in validator
VOLTAGE_PUMP_DEGRADE_V = 11.5   # R003: pump efficiency starts dropping below this
VOLTAGE_PUMP_FAIL_V    = 10.5   # R004: pump fails completely below this
VOLTAGE_FLAPS_FAIL_V   = 10.0   # R010: flaps/trim fail below this

# ── Scoring thresholds ────────────────────────────────────────────────────────
# Per context1 [Layer 5: Validation] and handoff §8

PHYSICS_SCORE_PASS     = 80   # Method 2: score >= 80 → physics sound
NTSB_SCORE_LIKELY_REAL = 70   # Method 3: score >= 70 → "likely real"
COMBINED_SCORE_KEEP    = 60   # Filter: combined > 60 → kept in final output

# ── fuel_pump reclassification ────────────────────────────────────────────────
# Known Layer 3 bug: fuel_pump was classified NOVEL because string matcher
# couldn't match "fuel_pump" to R003/R004/R005 triggers ("pump_fail", etc.).
# physics_validator.py fixes this before scoring.
# Handoff §6 note: "fuel_pump: Should be INDUCED, not NOVEL."

FUEL_PUMP_L1_FIX = {
    "correct_classification": "INDUCED",
    "l1_related_rules":       ["R003", "R004", "R005"],
    "l1_citations":           ["FAA AC 43.13-1B", "POH §7.1"],
    "fix_reason":             "String mismatch: fuel_pump relates to pump_fail/pump_degrade triggers",
}

# ── NTSB API configuration ────────────────────────────────────────────────────
# Confirmed format from probe: POST /Main requires "AndOr" field.
# Without it: 400 {"Error":"The Query AndOr value {0} is null"}

NTSB_API_BASE    = "https://data.ntsb.gov/carol-main-public/api/Query"
NTSB_API_TIMEOUT = 15   # seconds
NTSB_MAX_RESULTS = 100  # per query

# Bulk CSV download URL (fallback if API fails)
# Manual download: https://www.ntsb.gov/Pages/AviationDownloadData.aspx
# File: AviationData.txt (pipe-delimited, ~130MB for all records since 1962)
NTSB_BULK_CSV_URL = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip"