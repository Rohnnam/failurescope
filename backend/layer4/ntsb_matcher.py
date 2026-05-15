"""
FailureScope Layer 4 — NTSB Matcher
=====================================
Method 3 validation: for each discovered rule, search the loaded NTSB
accident records for cases that match the rule's failure component and
kill zone conditions.

Matching strategy (additive, each dimension contributes to score):

  Dimension 1 — Component keyword match (max 30 pts)
      Does the accident narrative/probable_cause mention the failure component?
      e.g. "alternator" rule → search for "alternator", "electrical failure",
           "battery discharge", related synonyms.

  Dimension 2 — Altitude match (max 20 pts)
      Does the reported altitude (if present) fall within the kill zone range?
      Many NTSB narratives mention altitude; extracted via simple regex.

  Dimension 3 — Pilot state match (max 20 pts)
      Swiss cheese variables present in narrative text:
      - Fatigue: "fatigue", "tired", "duty time", "long flight"
      - Low fuel:  "fuel exhaustion", "low fuel", "fuel starvation"

  Dimension 4 — Cascade progression match (max 20 pts)
      Does the narrative describe a chain of events consistent with the
      cascade variables (alt × rpm × vs × pitch pattern)?
      Looks for multi-step language: "resulted in", "led to", "followed by".

  Dimension 5 — Outcome severity match (max 10 pts)
      Was the outcome catastrophic (fatal injuries, aircraft destroyed)?

Total maximum: 100 pts.
Threshold for "likely real": NTSB_SCORE_LIKELY_REAL = 70 (from constants).

Important notes:
  - NTSB accident IDs in context1/context2 were illustrative. Fresh search only.
  - If no accidents are found for a component, ntsb_score = 0 (honest, not error).
  - Match quality is conservative: partial matches score < 50.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.layer3.constants import (
    C182_PHYSICS_BOUNDS,
    L1_RULES,
    NTSB_SCORE_LIKELY_REAL,
)

log = logging.getLogger(__name__)

# ── Component keyword map ─────────────────────────────────────────────────────
# Maps primary_failure component → search terms for narrative/probable_cause text.
# Synonyms drawn from NTSB narrative conventions.

COMPONENT_KEYWORDS: dict[str, list[str]] = {
    "alternator":        ["alternator", "electrical failure", "electrical system",
                           "charging system", "generator"],
    "battery":           ["battery", "electrical failure", "battery discharge",
                           "battery depletion", "electrical power"],
    "voltage_regulator": ["voltage regulator", "electrical failure", "alternator",
                           "charging system", "voltage"],
    "fuel_pump":         ["fuel pump", "fuel system", "fuel pressure",
                           "fuel starvation", "engine failure"],
    "fuel_selector":     ["fuel selector", "fuel starvation", "fuel exhaustion",
                           "improper fuel management", "fuel tank"],
    "fuel_system":       ["fuel system", "fuel starvation", "fuel exhaustion",
                           "fuel pressure", "fuel contamination"],
    "fuel_tank_left":    ["fuel", "fuel imbalance", "fuel tank", "fuel quantity"],
    "fuel_tank_right":   ["fuel", "fuel imbalance", "fuel tank", "fuel quantity"],
    "carburetor":        ["carburetor", "carburetor icing", "carb ice",
                           "engine failure", "rpm loss"],
    "magnetos":          ["magneto", "ignition", "engine roughness", "engine failure"],
    "engine_left":       ["engine failure", "engine", "power loss", "engine stop"],
    "engine_right":      ["engine failure", "engine", "power loss", "engine stop"],
    "engine_both":       ["dual engine", "engine failure", "power loss",
                           "both engines", "complete power loss"],
    "vacuum_pump":       ["vacuum pump", "vacuum system", "attitude indicator",
                           "instrument failure", "spatial disorientation"],
    "gyroscope":         ["gyroscope", "attitude indicator", "instrument failure",
                           "spatial disorientation", "vacuum"],
    "attitude_indicator":["attitude indicator", "instrument failure",
                           "spatial disorientation", "vacuum"],
    "flaps":             ["flap", "landing configuration", "control failure"],
    "trim_system":       ["trim", "pitch control", "control failure"],
    "hydraulics":        ["hydraulic", "control system failure"],
    "landing_gear":      ["landing gear", "gear failure", "gear retraction"],
    "rudder":            ["rudder", "directional control", "yaw"],
    "ailerons":          ["aileron", "roll control", "lateral control"],
    "elevator":          ["elevator", "pitch control", "longitudinal control"],
}

# Fatigue and low-fuel keywords for swiss cheese matching
FATIGUE_KEYWORDS = [
    "fatigue", "tired", "duty time", "flight time", "long flight",
    "fatigued", "rest", "sleep", "hours flown",
]
LOW_FUEL_KEYWORDS = [
    "fuel exhaustion", "fuel starvation", "low fuel", "fuel quantity",
    "improper fuel management", "fuel imbalance", "empty tank",
]
CASCADE_KEYWORDS = [
    "resulted in", "led to", "followed by", "subsequently", "which caused",
    "failure of", "loss of", "degraded", "continued to", "eventually",
]


# ── Matching helpers ──────────────────────────────────────────────────────────

def _get_text(record: dict) -> str:
    """Combine narrative + probable_cause into a single searchable string."""
    return (
        (record.get("narrative") or "")
        + " "
        + (record.get("probable_cause") or "")
    ).lower()


def _extract_altitude_from_text(text: str) -> list[float]:
    """
    Extract altitude values mentioned in accident text.
    Handles patterns like "5,000 feet", "5000 ft", "at 3500 MSL".
    Returns list of floats (may be empty).
    """
    patterns = [
        r"(\d{1,2},\d{3})\s*(?:feet|ft|foot|msl)",
        r"(\d{3,5})\s*(?:feet|ft|foot|msl)",
        r"altitude[^\d]{1,20}(\d{3,5})",
    ]
    found: list[float] = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            raw = match.group(1).replace(",", "")
            try:
                alt = float(raw)
                if 0 <= alt <= 20000:  # plausible GA altitude
                    found.append(alt)
            except ValueError:
                pass
    return found


def _score_component_match(record: dict, component: str) -> int:
    """Dimension 1: component keyword match. Returns 0–30."""
    text = _get_text(record)
    keywords = COMPONENT_KEYWORDS.get(component, [component.replace("_", " ")])

    hit_count = sum(1 for kw in keywords if kw.lower() in text)
    if hit_count == 0:
        return 0
    elif hit_count == 1:
        return 15
    elif hit_count == 2:
        return 22
    else:
        return 30


def _score_altitude_match(record: dict, alt_lo: float, alt_hi: float) -> int:
    """Dimension 2: altitude within kill zone. Returns 0–20."""
    text = _get_text(record)
    alts = _extract_altitude_from_text(text)
    if not alts:
        return 0  # No altitude mentioned — can't score
    # Any mentioned altitude within ±20% of kill zone range earns points
    margin = (alt_hi - alt_lo) * 0.20
    for alt in alts:
        if (alt_lo - margin) <= alt <= (alt_hi + margin):
            return 20
    # Altitude mentioned but outside range — partial credit
    return 5


def _score_swiss_cheese_match(
    record: dict, swiss_vars: list[str], kill_zone: dict
) -> int:
    """Dimension 3: swiss cheese variables (fatigue/low fuel) in narrative. Returns 0–20."""
    text = _get_text(record)
    score = 0
    matched: list[str] = []

    if "fatigue" in swiss_vars:
        if any(kw in text for kw in FATIGUE_KEYWORDS):
            score += 10
            matched.append("fatigue")

    # Low fuel pattern: fuel_total cat_range low from kill zone initial_conditions
    ic = kill_zone.get("initial_conditions", {})
    fuel_cat = ic.get("fuel_total_at_inject", {}).get("cat_range", [99, 99])
    if isinstance(fuel_cat, list) and len(fuel_cat) == 2:
        if float(fuel_cat[1]) < 15.0:  # Cat P75 < 15 gal → low fuel scenario
            if any(kw in text for kw in LOW_FUEL_KEYWORDS):
                score += 10
                matched.append("low_fuel")

    if matched:
        log.debug(f"    Swiss cheese match: {matched}")
    return score


def _score_cascade_match(record: dict, cascade_vars: list[str]) -> int:
    """Dimension 4: cascade progression language. Returns 0–20."""
    text = _get_text(record)
    cascade_keyword_hits = sum(1 for kw in CASCADE_KEYWORDS if kw in text)
    # Also look for variable names
    var_hits = sum(1 for v in cascade_vars if v in text)

    if cascade_keyword_hits >= 3 and var_hits >= 1:
        return 20
    elif cascade_keyword_hits >= 2:
        return 12
    elif cascade_keyword_hits >= 1:
        return 6
    return 0


def _score_outcome_match(record: dict, crash_probability: float) -> int:
    """Dimension 5: outcome severity consistent with crash probability. Returns 0–10."""
    fatalities = record.get("total_fatalities", "")
    severity   = (record.get("injury_severity") or "").lower()

    is_fatal = (
        (isinstance(fatalities, (int, float)) and int(fatalities) > 0)
        or (isinstance(fatalities, str) and fatalities.isdigit() and int(fatalities) > 0)
        or "fatal" in severity
        or "destroyed" in (record.get("aircraft_damage") or "").lower()
    )

    if crash_probability >= 0.9 and is_fatal:
        return 10
    elif crash_probability >= 0.5 and is_fatal:
        return 7
    elif crash_probability < 0.5 and not is_fatal:
        return 7  # Low probability, non-fatal — also consistent
    else:
        return 3  # Partial match


def _score_single_accident(
    record: dict,
    rule: dict,
) -> dict:
    """
    Score a single accident record against a rule.

    Returns:
        {accident_id, total_score, dimension_scores, text_snippet}
    """
    component      = rule.get("primary_failure", "unknown")
    cascade_vars   = rule.get("cascade_variables", [])
    swiss_vars     = rule.get("swiss_cheese_variables", [])
    crash_prob     = rule.get("crash_probability_in_this_zone", 0.0)
    csc            = rule.get("critical_state_combination", {})

    alt_range = csc.get("altitude_range_ft", [0, 16500])
    alt_lo    = float(alt_range[0]) if isinstance(alt_range, list) else 0.0
    alt_hi    = float(alt_range[1]) if isinstance(alt_range, list) else 16500.0

    d1 = _score_component_match(record, component)
    d2 = _score_altitude_match(record, alt_lo, alt_hi)
    d3 = _score_swiss_cheese_match(record, swiss_vars, csc)
    d4 = _score_cascade_match(record, cascade_vars)
    d5 = _score_outcome_match(record, crash_prob)

    total = d1 + d2 + d3 + d4 + d5

    # Short narrative snippet for the output report
    narrative = record.get("narrative") or record.get("probable_cause") or ""
    snippet   = narrative[:200].strip() + ("…" if len(narrative) > 200 else "")

    return {
        "accident_id":       record.get("accident_id", "UNKNOWN"),
        "event_date":        record.get("event_date", ""),
        "total_score":       total,
        "dimension_scores":  {"component": d1, "altitude": d2,
                               "swiss_cheese": d3, "cascade": d4, "outcome": d5},
        "text_snippet":      snippet,
        "aircraft_model":    record.get("aircraft_model", ""),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def match_rules_to_accidents(
    rules: list[dict],
    accidents_per_component: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Match each rule against its component's NTSB accident records.

    Args:
        rules:                    list of emergent_rule dicts
        accidents_per_component:  {component: [ntsb_record, ...]}
                                  from ntsb_loader.load_ntsb_per_component()

    Returns:
        dict keyed by primary_failure →
            {
              "ntsb_score":          int (0–100, best single match),
              "status":              str ("VALIDATED" / "POSSIBLE" / "UNCONFIRMED"),
              "n_accidents_searched":int,
              "n_matches":           int (score >= 40),
              "top_matches":         list[dict] (top 3 by score),
              "likely_real":         bool,
            }
    """
    results: dict[str, dict] = {}

    for rule in rules:
        comp      = rule.get("primary_failure", "unknown")
        accidents = accidents_per_component.get(comp, [])

        if not accidents:
            log.info(f"  {comp:22s}  ntsb_score=  0  (no accident data)")
            results[comp] = {
                "ntsb_score":           0,
                "status":               "UNCONFIRMED",
                "n_accidents_searched": 0,
                "n_matches":            0,
                "top_matches":          [],
                "likely_real":          False,
            }
            continue

        # Score all accidents for this rule
        scored = [_score_single_accident(acc, rule) for acc in accidents]
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        best_score   = scored[0]["total_score"] if scored else 0
        n_matches    = sum(1 for s in scored if s["total_score"] >= 40)
        top_matches  = scored[:3]

        # Classify
        if best_score >= NTSB_SCORE_LIKELY_REAL:
            status = "VALIDATED"
        elif best_score >= 50:
            status = "POSSIBLE"
        else:
            status = "UNCONFIRMED"

        likely_real = best_score >= NTSB_SCORE_LIKELY_REAL

        log.info(
            f"  {comp:22s}  ntsb_score={best_score:3d}  {status}  "
            f"(searched={len(accidents)}, matches={n_matches})"
        )

        results[comp] = {
            "ntsb_score":           best_score,
            "status":               status,
            "n_accidents_searched": len(accidents),
            "n_matches":            n_matches,
            "top_matches":          top_matches,
            "likely_real":          likely_real,
        }

    validated_count = sum(1 for r in results.values() if r["status"] == "VALIDATED")
    log.info(
        f"NTSB matching complete: {validated_count}/{len(rules)} rules validated "
        f"(threshold={NTSB_SCORE_LIKELY_REAL})"
    )
    return results


def match_rules_with_single_pool(
    rules: list[dict],
    all_accidents: list[dict],
) -> dict[str, dict]:
    """
    Match rules against a single pool of accidents (when per-component
    fetching wasn't used — e.g. a single broad query was run instead).

    Filters the pool by component keywords before scoring.
    """
    # Partition by component
    per_component: dict[str, list[dict]] = {
        rule["primary_failure"]: [] for rule in rules
    }

    for acc in all_accidents:
        text = _get_text(acc)
        for rule in rules:
            comp = rule.get("primary_failure", "")
            kws  = COMPONENT_KEYWORDS.get(comp, [comp.replace("_", " ")])
            if any(kw.lower() in text for kw in kws):
                per_component[comp].append(acc)

    return match_rules_to_accidents(rules, per_component)