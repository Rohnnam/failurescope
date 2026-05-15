"""
FailureScope Layer 4 — Discovery Ranker
=========================================
Combines physics (Method 2) and NTSB (Method 3) scores into a single
combined confidence score, filters by threshold, and ranks for output.

Scoring formula (from handoff §8):
    combined_score = (physics_score × ntsb_score) / 100

This is a geometric product, not an arithmetic mean. It penalises rules
where either source is weak. Examples:
    physics=95, ntsb=90  →  combined=85.5  (both strong, high confidence)
    physics=90, ntsb=50  →  combined=45.0  (NTSB uncertain, fails threshold)
    physics=80, ntsb=70  →  combined=56.0  (borderline — just below threshold)
    physics=85, ntsb=75  →  combined=63.8  (passes threshold)

Filter threshold: COMBINED_SCORE_KEEP = 60 (from constants).

Output also includes a human-readable validation_status:
    combined >= 80:  "VALIDATED"       — strong support from both methods
    combined >= 60:  "LIKELY_REAL"     — passes threshold, some uncertainty
    combined >= 40:  "POSSIBLE"        — below threshold, not included by default
    combined <  40:  "INSUFFICIENT_EVIDENCE"

Rules with ntsb_score=0 (no accident data) are NOT automatically failed.
They receive an adjusted combined score using physics_score × 0.5 and
validation_status = "PHYSICS_ONLY" — they may still be included if
physics_score is high enough that combined > 60 (requires physics > 120,
which is impossible, so they are always filtered out but documented).

Design rationale: if we have no NTSB data, we can't claim Method 3 support.
These rules appear in the output as "PHYSICS_ONLY — requires historical validation".
"""

from __future__ import annotations

import logging
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.layer3.constants import (
    COMBINED_SCORE_KEEP,
    NTSB_SCORE_LIKELY_REAL,
    PHYSICS_SCORE_PASS,
)

log = logging.getLogger(__name__)


def _compute_combined(physics_score: int, ntsb_score: int) -> float:
    """
    Combined confidence = (physics × ntsb) / 100.
    If ntsb_score = 0 (no data), returns 0.0 regardless of physics.
    """
    if ntsb_score == 0:
        return 0.0
    return (physics_score * ntsb_score) / 100.0


def _validation_status(
    combined: float,
    physics_score: int,
    ntsb_score: int,
) -> str:
    if ntsb_score == 0:
        return "PHYSICS_ONLY"
    if combined >= 80:
        return "VALIDATED"
    if combined >= COMBINED_SCORE_KEEP:
        return "LIKELY_REAL"
    if combined >= 40:
        return "POSSIBLE"
    return "INSUFFICIENT_EVIDENCE"


def rank_and_filter(
    rules: list[dict],
    physics_results: dict[str, dict],
    ntsb_results: dict[str, dict],
    include_below_threshold: bool = False,
) -> list[dict]:
    """
    Combine scores, filter, and rank all rules.

    Args:
        rules:                   list of emergent_rule dicts from Layer 3
        physics_results:         {component: {score, passed, evidence, ...}}
                                 from physics_validator.validate_all_rules()
        ntsb_results:            {component: {ntsb_score, top_matches, ...}}
                                 from ntsb_matcher.match_rules_to_accidents()
        include_below_threshold: If True, include rules with combined < 60
                                 (tagged "POSSIBLE" or "INSUFFICIENT_EVIDENCE").
                                 Used for the full output report.

    Returns:
        list[dict] sorted by combined_score descending. Each entry:
        {
          "primary_failure":    str,
          "classification":     str,
          "physics_score":      int,
          "ntsb_score":         int,
          "combined_score":     float,
          "validation_status":  str,
          "passed_threshold":   bool,
          "physics_evidence":   list[str],
          "physics_deductions": list[str],
          "l1_citations":       list[str],
          "ntsb_top_matches":   list[dict],
          "n_accidents_searched": int,
          "likely_real":        bool,
          "rule":               dict,   (original rule dict, for report generation)
        }
    """
    ranked: list[dict] = []

    for rule in rules:
        comp = rule.get("primary_failure", "unknown")

        phys = physics_results.get(comp, {
            "score": 0, "passed": False, "evidence": [], "deductions": [],
            "classification": rule.get("classification", "UNKNOWN"),
            "l1_citations": [],
        })
        ntsb = ntsb_results.get(comp, {
            "ntsb_score": 0, "top_matches": [], "n_accidents_searched": 0,
            "likely_real": False,
        })

        p_score    = phys.get("score", 0)
        n_score    = ntsb.get("ntsb_score", 0)
        combined   = _compute_combined(p_score, n_score)
        status     = _validation_status(combined, p_score, n_score)
        passes     = combined >= COMBINED_SCORE_KEEP

        entry = {
            "primary_failure":      comp,
            "classification":       phys.get("classification", rule.get("classification")),
            "physics_score":        p_score,
            "ntsb_score":           n_score,
            "combined_score":       round(combined, 1),
            "validation_status":    status,
            "passed_threshold":     passes,
            "physics_evidence":     phys.get("evidence", []),
            "physics_deductions":   phys.get("deductions", []),
            "l1_citations":         phys.get("l1_citations", []),
            "classification_fix":   phys.get("classification_fix"),
            "ntsb_top_matches":     ntsb.get("top_matches", []),
            "n_accidents_searched": ntsb.get("n_accidents_searched", 0),
            "likely_real":          ntsb.get("likely_real", False),
            "rule":                 rule,
        }
        ranked.append(entry)

    # Sort: combined_score descending; break ties by physics_score
    ranked.sort(key=lambda x: (x["combined_score"], x["physics_score"]), reverse=True)

    # Report
    above  = sum(1 for r in ranked if r["passed_threshold"])
    below  = len(ranked) - above
    log.info(
        f"\nRanking summary:\n"
        f"  Total rules:          {len(ranked)}\n"
        f"  Above threshold (≥{COMBINED_SCORE_KEEP}): {above}\n"
        f"  Below threshold:      {below}\n"
        f"  Threshold:            combined = physics × ntsb / 100 > {COMBINED_SCORE_KEEP}"
    )
    for r in ranked[:5]:
        log.info(
            f"  {r['primary_failure']:22s}  "
            f"physics={r['physics_score']:3d}  ntsb={r['ntsb_score']:3d}  "
            f"combined={r['combined_score']:5.1f}  {r['validation_status']}"
        )

    if not include_below_threshold:
        filtered = [r for r in ranked if r["passed_threshold"]]
        log.info(f"  After filter:         {len(filtered)} rules kept")
        return filtered

    return ranked