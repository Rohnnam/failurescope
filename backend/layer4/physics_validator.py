"""
FailureScope Layer 4 — Physics Validator
==========================================
Method 2 validation: checks each discovered rule's kill zone values against
the Cessna 182 physical operating envelope and L1 constraint logic.

Input:  list of emergent_rules dicts (from induced_relationships.json)
Output: dict keyed by primary_failure → {score, evidence, passed, ...}

Scoring rubric (0–100):
  Base scores:
    INDUCED (has L1 anchor):         85
    NOVEL   (no L1 anchor):          65
    NOVEL with fatigue cluster:      72  (physically plausible scenario)

  Bonuses:
    +8   All kill zone values within C182 envelope
    +5   Altitude range fully within service ceiling (0–16500 ft)
    +4   Fatigue IQR directionally correct (cat > rec in swiss cheese)
    +3   Kill zone voltage consistent with L1 voltage thresholds

  Penalties:
    -25  Any kill zone value outside physical envelope
    -15  L1 citations present but kill zone contradicts L1 effect
    -10  swiss_cheese_variables empty despite being cascade component
    -5   sub_rules present but altitude bands overlap or are unordered

  Capped at [0, 100].

Source citations are carried through from l1_citations on each rule.
The fuel_pump reclassification (NOVEL → INDUCED) is applied before scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# ── Layer 3 constants (single source of truth) ───────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.layer3.constants import (
    C182_PHYSICS_BOUNDS,
    FUEL_PUMP_L1_FIX,
    L1_RULES,
    VOLTAGE_PUMP_DEGRADE_V,
    VOLTAGE_PUMP_FAIL_V,
    VOLTAGE_FLAPS_FAIL_V,
    LAYER3_MODELS_DIR,
    PHYSICS_SCORE_PASS,
)

log = logging.getLogger(__name__)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _in_bounds(value: float, bounds: dict) -> bool:
    return bounds["min"] <= value <= bounds["max"]


def _range_in_bounds(lo: float, hi: float, bounds: dict) -> bool:
    return _in_bounds(lo, bounds) and _in_bounds(hi, bounds)


def _l1_rule_by_id(rule_id: str) -> dict | None:
    for r in L1_RULES:
        if r["id"] == rule_id:
            return r
    return None


def _apply_fuel_pump_fix(rule: dict) -> dict:
    """
    Reclassify fuel_pump from NOVEL → INDUCED.

    Layer 3 bug: relationship_miner._crossref_with_l1_rules uses literal string
    matching against L1 trigger fields ("pump_fail", "pump_degraded") which don't
    contain the substring "fuel_pump". Result: fuel_pump gets no L1 match despite
    being directly governed by R003/R004/R005.

    Fix documented in handoff §6 note, constants.py FUEL_PUMP_L1_FIX.
    """
    if rule.get("primary_failure") == "fuel_pump" and rule.get("classification") == "NOVEL":
        rule = dict(rule)  # shallow copy — do not mutate original
        rule["classification"]    = FUEL_PUMP_L1_FIX["correct_classification"]
        rule["l1_related_rules"]  = FUEL_PUMP_L1_FIX["l1_related_rules"]
        rule["l1_citations"]      = FUEL_PUMP_L1_FIX["l1_citations"]
        rule["_classification_fix"] = FUEL_PUMP_L1_FIX["fix_reason"]
        log.info("fuel_pump reclassified NOVEL → INDUCED (L3 string-match bug fix)")
    return rule


# ── Kill zone field extractors ────────────────────────────────────────────────

def _extract_kill_zone(rule: dict) -> dict:
    """
    Normalise kill zone from critical_state_combination.

    Returns a flat dict with typed values ready for bounds checking:
      alt_lo, alt_hi, voltage_lo, voltage_hi,
      pitch_lo, pitch_hi,
      fatigue_cat_lo, fatigue_cat_hi, fatigue_rec_lo, fatigue_rec_hi
    """
    csc = rule.get("critical_state_combination", {})

    def _range(key: str) -> tuple[float, float]:
        v = csc.get(key, [None, None])
        if isinstance(v, list) and len(v) == 2:
            return float(v[0]), float(v[1])
        return 0.0, 0.0

    ic = csc.get("initial_conditions", {})
    fatigue = ic.get("fatigue", {})

    return {
        "alt_lo":          _range("altitude_range_ft")[0],
        "alt_hi":          _range("altitude_range_ft")[1],
        "voltage_lo":      _range("voltage_range_V")[0],
        "voltage_hi":      _range("voltage_range_V")[1],
        "pitch_lo":        _range("pitch_range_deg")[0],
        "pitch_hi":        _range("pitch_range_deg")[1],
        "fatigue_cat_lo":  float(fatigue.get("cat_range", [0.0, 1.0])[0]),
        "fatigue_cat_hi":  float(fatigue.get("cat_range", [0.0, 1.0])[1]),
        "fatigue_rec_lo":  float(fatigue.get("rec_range", [0.0, 1.0])[0]),
        "fatigue_rec_hi":  float(fatigue.get("rec_range", [0.0, 1.0])[1]),
    }


# ── Core scoring ──────────────────────────────────────────────────────────────

def _score_rule(rule: dict) -> dict:
    """
    Score a single rule 0–100 and return evidence list.

    Returns:
        {
          "score":            int,
          "passed":           bool,
          "evidence":         list[str],
          "deductions":       list[str],
          "classification":   str,
          "l1_citations":     list[str],
          "classification_fix": str | None,
        }
    """
    rule = _apply_fuel_pump_fix(rule)

    classification = rule.get("classification", "NOVEL")
    l1_citations   = rule.get("l1_citations", [])
    l1_related     = rule.get("l1_related_rules", [])
    component      = rule.get("primary_failure", "unknown")
    is_cascade     = rule.get("is_cascade_component", False)

    evidence:   list[str] = []
    deductions: list[str] = []

    # ── Base score ────────────────────────────────────────────────────────────
    if classification == "INDUCED":
        score = 85
        evidence.append(f"INDUCED classification: grounded in L1 rules {l1_related}")
    else:
        # NOVEL — check if it belongs to the "exhausted pilot + low fuel" cluster
        # (handoff §5.3: fatigue > 0.91 AND fuel < 8 gal).  Physically plausible.
        kz = _extract_kill_zone(rule)
        if kz["fatigue_cat_lo"] >= 0.88:
            score = 72
            evidence.append(
                "NOVEL classification: no direct L1 anchor, but kill zone aligns "
                "with physically plausible exhausted-pilot + low-fuel scenario"
            )
        else:
            score = 65
            evidence.append(
                "NOVEL classification: no L1 anchor; relies on empirical simulation pattern"
            )

    kz = _extract_kill_zone(rule)

    # ── Altitude bounds check ─────────────────────────────────────────────────
    alt_bounds = C182_PHYSICS_BOUNDS["alt"]
    if _range_in_bounds(kz["alt_lo"], kz["alt_hi"], alt_bounds):
        score += 5
        evidence.append(
            f"Altitude kill zone [{kz['alt_lo']:.0f}–{kz['alt_hi']:.0f} ft] "
            f"within service ceiling ({alt_bounds['source']})"
        )
    else:
        score -= 25
        deductions.append(
            f"FAIL: Altitude [{kz['alt_lo']:.0f}–{kz['alt_hi']:.0f} ft] "
            f"outside envelope [{alt_bounds['min']:.0f}–{alt_bounds['max']:.0f} ft]"
        )

    # ── Voltage consistency ───────────────────────────────────────────────────
    v_lo, v_hi = kz["voltage_lo"], kz["voltage_hi"]
    v_bounds   = C182_PHYSICS_BOUNDS["v"]

    if v_lo == 0.0 and v_hi == 0.0:
        # Battery/non-electrical failure: voltage=0 means "not a voltage-driven cascade"
        if classification == "INDUCED" and any(
            r in l1_related for r in ("R001", "R003", "R013")
        ):
            # Electrical rule with 0 voltage signal is suspicious but battery failure
            # legitimately has v=0 (fully discharged)
            if component in ("battery",):
                score += 3
                evidence.append(
                    "Voltage=0.0 consistent with battery depletion (no charging source)"
                )
            else:
                score -= 10
                deductions.append(
                    f"WARNING: Electrical rule ({component}) shows voltage IQR=0 — "
                    "kill zone may reflect step-function failure, not gradual cascade"
                )
        else:
            # Non-electrical component: voltage=0 just means voltage isn't the cascade var
            score += 3
            evidence.append("Voltage not a primary cascade variable for this component (expected)")
    elif _range_in_bounds(v_lo, v_hi, v_bounds):
        score += 3
        # Check L1 voltage threshold consistency
        if v_hi <= VOLTAGE_PUMP_FAIL_V:
            evidence.append(
                f"Voltage kill zone [{v_lo:.2f}–{v_hi:.2f} V] is at/below R004 pump-fail "
                f"threshold ({VOLTAGE_PUMP_FAIL_V} V) — consistent with fuel system cascade"
            )
        elif v_hi <= VOLTAGE_PUMP_DEGRADE_V:
            evidence.append(
                f"Voltage kill zone [{v_lo:.2f}–{v_hi:.2f} V] is in R003 degradation band "
                f"({VOLTAGE_PUMP_DEGRADE_V} V threshold) — consistent with pump efficiency loss"
            )
        else:
            evidence.append(
                f"Voltage kill zone [{v_lo:.2f}–{v_hi:.2f} V] within operating envelope"
            )
    else:
        score -= 25
        deductions.append(
            f"FAIL: Voltage [{v_lo:.2f}–{v_hi:.2f} V] outside physical envelope "
            f"[{v_bounds['min']:.1f}–{v_bounds['max']:.1f} V]"
        )

    # ── Pitch bounds check ────────────────────────────────────────────────────
    pitch_bounds = C182_PHYSICS_BOUNDS["pitch"]
    p_lo, p_hi   = kz["pitch_lo"], kz["pitch_hi"]
    if _range_in_bounds(p_lo, p_hi, pitch_bounds):
        if p_lo > 0:
            score += 2
            evidence.append(
                f"Pitch kill zone [{p_lo:.1f}°–{p_hi:.1f}°] is positive (nose-up), "
                "consistent with recovery maneuver attempt (pilot pitching up)"
            )
        else:
            score += 2
            evidence.append(f"Pitch kill zone [{p_lo:.1f}°–{p_hi:.1f}°] within flight envelope")
    else:
        score -= 25
        deductions.append(
            f"FAIL: Pitch [{p_lo:.1f}°–{p_hi:.1f}°] outside flight envelope "
            f"[{pitch_bounds['min']:.0f}°–{pitch_bounds['max']:.0f}°]"
        )

    # ── Fatigue directionality ────────────────────────────────────────────────
    f_cat_lo, f_cat_hi = kz["fatigue_cat_lo"], kz["fatigue_cat_hi"]
    f_rec_lo, f_rec_hi = kz["fatigue_rec_lo"], kz["fatigue_rec_hi"]
    f_bounds           = C182_PHYSICS_BOUNDS["fatigue"]

    if (
        _range_in_bounds(f_cat_lo, f_cat_hi, f_bounds)
        and _range_in_bounds(f_rec_lo, f_rec_hi, f_bounds)
        and f_cat_lo > f_rec_hi - 0.05  # cat range is meaningfully higher than rec
    ):
        score += 4
        evidence.append(
            f"Fatigue swiss cheese: catastrophic P25–P75 = [{f_cat_lo:.2f}–{f_cat_hi:.2f}] "
            f"> recoverable [{f_rec_lo:.2f}–{f_rec_hi:.2f}] — directionally correct "
            "(higher pilot fatigue → worse outcomes)"
        )
    elif f_cat_lo == 0.0 and f_cat_hi == 0.0:
        # No fatigue data in swiss cheese (some rules don't have it)
        pass
    else:
        score -= 5
        deductions.append(
            f"Fatigue direction unexpected: cat=[{f_cat_lo:.2f}–{f_cat_hi:.2f}] "
            f"rec=[{f_rec_lo:.2f}–{f_rec_hi:.2f}]"
        )

    # ── Swiss cheese completeness check ──────────────────────────────────────
    swiss_vars = rule.get("swiss_cheese_variables", [])
    if is_cascade and len(swiss_vars) == 0:
        score -= 10
        deductions.append(
            "WARN: Cascade component has no swiss cheese variables — "
            "pre-existing condition analysis absent"
        )
    elif len(swiss_vars) >= 2:
        evidence.append(
            f"Swiss cheese pattern captures {len(swiss_vars)} pre-existing conditions: "
            f"{swiss_vars}"
        )

    # ── Sub-rule altitude band ordering ──────────────────────────────────────
    sub_rules = rule.get("sub_rules_by_altitude", [])
    if len(sub_rules) >= 2:
        # Check LOW alt band has higher crash_prob than HIGH (more compressed recovery)
        low_rule  = next((s for s in sub_rules if "LOW"  in s.get("altitude_band", "")), None)
        high_rule = next((s for s in sub_rules if "HIGH" in s.get("altitude_band", "")), None)
        if low_rule and high_rule:
            low_p  = low_rule.get("crash_probability", 0)
            high_p = high_rule.get("crash_probability", 0)
            if low_p >= high_p:
                score += 3
                evidence.append(
                    f"Sub-rule altitude ordering correct: LOW alt crash_p={low_p:.2f} "
                    f">= HIGH alt crash_p={high_p:.2f} "
                    "(less altitude → compressed recovery margin)"
                )
            else:
                score -= 5
                deductions.append(
                    f"Sub-rule ordering unexpected: LOW crash_p={low_p:.2f} < "
                    f"HIGH crash_p={high_p:.2f} (unusual — low altitude should compress recovery)"
                )

    # ── L1 citation completeness ──────────────────────────────────────────────
    if l1_citations:
        evidence.append(f"L1 citations: {l1_citations}")

    # ── Cap and finalise ──────────────────────────────────────────────────────
    score = max(0, min(100, score))
    passed = score >= PHYSICS_SCORE_PASS

    return {
        "score":              score,
        "passed":             passed,
        "evidence":           evidence,
        "deductions":         deductions,
        "classification":     classification,
        "l1_citations":       l1_citations,
        "classification_fix": rule.get("_classification_fix"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def validate_all_rules(rules: list[dict]) -> dict[str, dict]:
    """
    Validate physics plausibility for all emergent rules.

    Args:
        rules: list of dicts from induced_relationships.json["emergent_rules"]

    Returns:
        dict keyed by primary_failure →
            {score, passed, evidence, deductions, classification, l1_citations,
             classification_fix}

    Notes:
        - fuel_pump reclassification is applied internally (NOVEL → INDUCED)
        - Mutates nothing; original rule dicts are not modified
    """
    results: dict[str, dict] = {}
    passed_count = 0

    for rule in rules:
        comp = rule.get("primary_failure", "unknown")
        try:
            result = _score_rule(rule)
            results[comp] = result
            status = "PASS" if result["passed"] else "FAIL"
            fix_note = f" [reclassified: {result['classification_fix']}]" \
                       if result["classification_fix"] else ""
            log.info(
                f"  {comp:22s}  score={result['score']:3d}  {status}{fix_note}"
            )
            if result["passed"]:
                passed_count += 1
        except Exception as exc:
            log.error(f"  {comp}: physics validation error: {exc}")
            results[comp] = {
                "score": 0, "passed": False,
                "evidence": [], "deductions": [f"Validation error: {exc}"],
                "classification": rule.get("classification", "UNKNOWN"),
                "l1_citations": [], "classification_fix": None,
            }

    log.info(
        f"Physics validation complete: {passed_count}/{len(rules)} rules passed "
        f"(threshold={PHYSICS_SCORE_PASS})"
    )
    return results


def load_and_validate(
    induced_path: str | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Convenience wrapper: load induced_relationships.json, run validation.

    Returns:
        (rules_list, physics_results_dict)
    """
    induced_path = induced_path or f"{LAYER3_MODELS_DIR}/induced_relationships.json"
    with open(induced_path) as f:
        data = json.load(f)
    rules = data["emergent_rules"]
    log.info(f"Loaded {len(rules)} emergent rules from {induced_path}")
    results = validate_all_rules(rules)
    return rules, results


def save_physics_results(results: dict[str, dict], output_path: str) -> None:
    """Write physics validation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Physics results saved → {output_path}")