"""
FailureScope Layer 4 — Validation Report Generator
====================================================
Produces validated_findings.json in the output format specified in the handoff.

Output format per handoff §11:
    {
      "failure_mode":             str,  e.g. "alternator_cascade_induced"
      "primary_failure":          str,
      "classification":           str,  "INDUCED" | "NOVEL" | "PHYSICS_ONLY"
      "physics_score":            int,
      "ntsb_score":               int,
      "combined_confidence":      float,
      "validation_status":        str,
      "passed_threshold":         bool,
      "citations":                list[str],
      "ntsb_matching_accidents":  list[str],  accident_ids
      "match_quality":            dict,       {accident_id: score}
      "interpretation":           str,
      "cascade_variables":        list[str],
      "swiss_cheese_variables":   list[str],
      "kill_zone":                dict,
      "sub_rules_by_altitude":    list[dict],
      "physics_evidence":         list[str],
      "physics_deductions":       list[str],
      "classification_fix":       str | null,
    }

Also writes a summary section at the top level:
    {
      "meta": {...},
      "summary": {n_total, n_validated, n_likely_real, n_physics_only, ...},
      "findings": [list of per-rule output dicts, sorted by combined_confidence],
    }

All rules appear in "findings" (not just those above threshold).
Rules below threshold are tagged with validation_status and passed_threshold=False
so the frontend can display them with appropriate caveats.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.layer3.constants import (
    COMBINED_SCORE_KEEP,
    VALIDATED_FINDINGS_PATH,
)

log = logging.getLogger(__name__)


def _failure_mode_id(rule: dict, classification: str) -> str:
    """
    Build a human-readable failure_mode identifier.
    e.g. "alternator_cascade_induced", "carburetor_cascade_novel"
    """
    comp = rule.get("primary_failure", "unknown")
    cls  = classification.lower().replace("_only", "")
    return f"{comp}_cascade_{cls}"


def _build_finding(ranked_entry: dict) -> dict:
    """
    Convert a ranked entry (from discovery_ranker) to the output report format.
    """
    rule  = ranked_entry["rule"]
    comp  = ranked_entry["primary_failure"]
    cls   = ranked_entry["classification"]

    # Accident IDs + match quality from top NTSB matches
    top_matches    = ranked_entry.get("ntsb_top_matches", [])
    accident_ids   = [m["accident_id"] for m in top_matches if m.get("accident_id")]
    match_quality  = {
        m["accident_id"]: m["total_score"]
        for m in top_matches
        if m.get("accident_id") and m["accident_id"] != "UNKNOWN"
    }

    # Kill zone: pass through from critical_state_combination
    csc = rule.get("critical_state_combination", {})

    # Interpretation: prefer rule's own interpretation, else build one
    interpretation = rule.get("interpretation", "")
    if not interpretation:
        cascade_vars = rule.get("cascade_variables", [])
        swiss_vars   = rule.get("swiss_cheese_variables", [])
        cp           = rule.get("crash_probability_in_this_zone", 0)
        interpretation = (
            f"For '{comp}' failures: cascade dynamics across "
            f"{' × '.join(cascade_vars)} determine outcome, modulated by "
            f"pre-existing conditions ({', '.join(swiss_vars)}). "
            f"Crash probability in this state zone: {cp:.0%}."
        )

    return {
        "failure_mode":            _failure_mode_id(rule, cls),
        "primary_failure":         comp,
        "classification":          cls,
        "physics_score":           ranked_entry["physics_score"],
        "ntsb_score":              ranked_entry["ntsb_score"],
        "combined_confidence":     ranked_entry["combined_score"],
        "validation_status":       ranked_entry["validation_status"],
        "passed_threshold":        ranked_entry["passed_threshold"],
        "citations":               ranked_entry["l1_citations"],
        "ntsb_matching_accidents": accident_ids,
        "match_quality":           match_quality,
        "interpretation":          interpretation,
        "cascade_variables":       rule.get("cascade_variables", []),
        "swiss_cheese_variables":  rule.get("swiss_cheese_variables", []),
        "kill_zone":               csc,
        "crash_probability":       rule.get("crash_probability_in_this_zone", 0),
        "sub_rules_by_altitude":   rule.get("sub_rules_by_altitude", []),
        "n_sub_rules":             rule.get("n_sub_rules", 0),
        "induced_relationship":    rule.get("induced_relationship", ""),
        "l1_related_rules":        rule.get("l1_related_rules", []),
        "physics_evidence":        ranked_entry["physics_evidence"],
        "physics_deductions":      ranked_entry["physics_deductions"],
        "classification_fix":      ranked_entry.get("classification_fix"),
        "n_accidents_searched":    ranked_entry.get("n_accidents_searched", 0),
        "source":                  rule.get("source", "cascade_transformer + snapshot_xgboost"),
    }


def _build_summary(findings: list[dict]) -> dict:
    """Build the summary section from all findings."""
    n_total         = len(findings)
    n_validated     = sum(1 for f in findings if f["validation_status"] == "VALIDATED")
    n_likely_real   = sum(1 for f in findings if f["validation_status"] == "LIKELY_REAL")
    n_physics_only  = sum(1 for f in findings if f["validation_status"] == "PHYSICS_ONLY")
    n_possible      = sum(1 for f in findings if f["validation_status"] == "POSSIBLE")
    n_insufficient  = sum(1 for f in findings if f["validation_status"] == "INSUFFICIENT_EVIDENCE")
    n_above         = sum(1 for f in findings if f["passed_threshold"])
    n_induced       = sum(1 for f in findings if f["classification"] == "INDUCED")
    n_novel         = sum(1 for f in findings if f["classification"] == "NOVEL")

    return {
        "n_total_rules":            n_total,
        "n_above_threshold":        n_above,
        "n_validated":              n_validated,
        "n_likely_real":            n_likely_real,
        "n_physics_only":           n_physics_only,
        "n_possible":               n_possible,
        "n_insufficient_evidence":  n_insufficient,
        "n_induced_classification": n_induced,
        "n_novel_classification":   n_novel,
        "combined_threshold":       COMBINED_SCORE_KEEP,
        "note": (
            "Findings include all rules regardless of threshold. "
            "'passed_threshold' field indicates whether combined_confidence >= "
            f"{COMBINED_SCORE_KEEP}. 'PHYSICS_ONLY' rules have ntsb_score=0 — "
            "physics validation passed but no historical accident confirmation found."
        ),
    }


def generate_validated_findings(
    ranked_entries: list[dict],
    output_path: str | None = None,
    *,
    include_all: bool = True,
) -> dict:
    """
    Generate and write validated_findings.json.

    Args:
        ranked_entries:  Output of discovery_ranker.rank_and_filter()
                         (pass include_below_threshold=True to get all rules).
        output_path:     Destination path (default: VALIDATED_FINDINGS_PATH).
        include_all:     If True, all rules appear in output (above+below threshold).
                         Rules below threshold are clearly tagged.

    Returns:
        The full output dict (also written to disk).
    """
    output_path = output_path or VALIDATED_FINDINGS_PATH

    findings = [_build_finding(entry) for entry in ranked_entries]

    # Sort: validated first, then by combined_confidence descending
    findings.sort(
        key=lambda f: (
            f["passed_threshold"],
            f["combined_confidence"],
            f["physics_score"],
        ),
        reverse=True,
    )

    summary = _build_summary(findings)

    output = {
        "meta": {
            "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "layer":           "Layer 4 — Failure Mode Discovery + NTSB Validation",
            "source_model":    "cascade_transformer + snapshot_xgboost (Layer 3)",
            "aircraft":        "Cessna 182",
            "n_trajectories":  500_000,
            "physics_threshold":  80,
            "ntsb_threshold":     70,
            "combined_threshold": COMBINED_SCORE_KEEP,
            "validation_methods": ["Method 2: Physics Soundness", "Method 3: Historical Fit"],
            "ntsb_disclaimer": (
                "NTSB accident IDs in this file were matched against live NTSB data. "
                "Accident IDs cited in early project documentation (WPR13FA089, "
                "ANC12WA034, SFO08CA123) were illustrative examples and have not been "
                "verified. All matches here are from fresh live searches."
            ),
        },
        "summary": summary,
        "findings": findings,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(
        f"\nValidated findings written → {output_path}\n"
        f"  Total rules:        {summary['n_total_rules']}\n"
        f"  Above threshold:    {summary['n_above_threshold']}\n"
        f"  Validated (both):   {summary['n_validated']}\n"
        f"  Likely real:        {summary['n_likely_real']}\n"
        f"  Physics only:       {summary['n_physics_only']}\n"
    )

    return output


def print_findings_table(output: dict) -> None:
    """Print a human-readable summary table of findings to stdout."""
    findings = output.get("findings", [])
    print(
        f"\n{'='*88}\n"
        f"  FailureScope Layer 4 — Validated Findings\n"
        f"{'='*88}\n"
        f"  {'Component':22s}  {'Class':8s}  {'Phys':4s}  {'NTSB':4s}  "
        f"{'Comb':5s}  {'Status'}\n"
        f"  {'-'*80}"
    )
    for f in findings:
        marker = "✓" if f["passed_threshold"] else " "
        print(
            f"  {marker} {f['primary_failure']:21s}  "
            f"{f['classification']:8s}  "
            f"{f['physics_score']:3d}   "
            f"{f['ntsb_score']:3d}   "
            f"{f['combined_confidence']:5.1f}  "
            f"{f['validation_status']}"
        )
    print(f"{'='*88}")

    summary = output.get("summary", {})
    print(
        f"\n  Above threshold (combined ≥ {COMBINED_SCORE_KEEP}):  "
        f"{summary.get('n_above_threshold', 0)} / {summary.get('n_total_rules', 0)} rules\n"
        f"  Validated (strong evidence):  {summary.get('n_validated', 0)}\n"
        f"  Likely real:                  {summary.get('n_likely_real', 0)}\n"
        f"  Physics only (no NTSB data):  {summary.get('n_physics_only', 0)}\n"
    )