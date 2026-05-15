"""
FailureScope Layer 5 — L4 Findings Matcher
============================================
Looks up the relevant validated finding from Layer 4's output for a given
failure component and scenario, and formats it for API consumption.

validated_findings.json is loaded once at server startup and cached.
The lookup is a simple dict keyed by primary_failure — O(1).

What it returns to the frontend:
  - failure_mode identifier
  - validation_status (VALIDATED / LIKELY_REAL / PHYSICS_ONLY)
  - combined_confidence score
  - citations (POH/FAA sources)
  - NTSB matching accident IDs + match quality
  - interpretation text
  - kill_zone (state variable ranges that characterise the danger zone)
  - induced_relationship (the ML-discovered pattern)
  - sub_rules_by_altitude (altitude-specific variants)
  - swiss_cheese_variables (which pre-existing conditions matter)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FINDINGS_PATH = Path("/opt/failurescope_data/models/validated_findings.json")

# Module-level cache — loaded once at startup
_findings_by_component: dict[str, dict] = {}
_findings_loaded = False


def load_findings() -> bool:
    """
    Load validated_findings.json into module cache.
    Called once at server startup.
    """
    global _findings_by_component, _findings_loaded

    if _findings_loaded:
        return True

    if not _FINDINGS_PATH.exists():
        log.warning(f"  validated_findings.json not found at {_FINDINGS_PATH}")
        return False

    try:
        with open(_FINDINGS_PATH) as f:
            data = json.load(f)

        for finding in data.get("findings", []):
            comp = finding.get("primary_failure", "")
            if comp:
                _findings_by_component[comp] = finding

        _findings_loaded = True
        log.info(
            f"  L4 findings loaded: {len(_findings_by_component)} components "
            f"from {_FINDINGS_PATH}"
        )
        return True

    except Exception as e:
        log.warning(f"  Failed to load validated_findings.json: {e}")
        return False


def match(component: str) -> dict[str, Any]:
    """
    Return the validated finding for the given failure component.

    Args:
        component:  primary_failure string, e.g. "alternator"

    Returns:
        dict with finding fields formatted for API response, or
        a 'not_found' dict if the component has no validated finding.
    """
    if not _findings_loaded:
        load_findings()

    finding = _findings_by_component.get(component)
    if not finding:
        return {
            "found":              False,
            "component":          component,
            "validation_status":  "NOT_IN_FINDINGS",
            "message":            (
                f"No validated finding for '{component}'. "
                "This component may not have been in the Layer 3 training set."
            ),
        }

    return {
        "found":                   True,
        "failure_mode":            finding.get("failure_mode"),
        "primary_failure":         component,
        "classification":          finding.get("classification"),
        "validation_status":       finding.get("validation_status"),
        "passed_threshold":        finding.get("passed_threshold"),
        "combined_confidence":     finding.get("combined_confidence"),
        "physics_score":           finding.get("physics_score"),
        "ntsb_score":              finding.get("ntsb_score"),
        "citations":               finding.get("citations", []),
        "ntsb_matching_accidents": finding.get("ntsb_matching_accidents", []),
        "match_quality":           finding.get("match_quality", {}),
        "interpretation":          finding.get("interpretation", ""),
        "induced_relationship":    finding.get("induced_relationship", ""),
        "cascade_variables":       finding.get("cascade_variables", []),
        "swiss_cheese_variables":  finding.get("swiss_cheese_variables", []),
        "crash_probability":       finding.get("crash_probability"),
        "kill_zone":               finding.get("kill_zone", {}),
        "sub_rules_by_altitude":   finding.get("sub_rules_by_altitude", []),
        "l1_related_rules":        finding.get("l1_related_rules", []),
        "n_accidents_searched":    finding.get("n_accidents_searched", 0),
    }


def match_for_response(component: str) -> dict[str, Any]:
    """
    Alias for match() — kept for readability at call sites.
    """
    return match(component)