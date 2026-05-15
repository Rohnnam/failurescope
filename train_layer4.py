#!/usr/bin/env python3
"""
FailureScope — Layer 4 Runner
================================
Validates discovered failure modes via physics soundness (Method 2)
and historical accident correspondence (Method 3 / NTSB).

Pipeline:
  1. validate-physics  Load induced_relationships.json. Score each rule 0–100
                       against C182 physical envelope + L1 constraint logic.
                       Applies fuel_pump NOVEL→INDUCED reclassification fix.
                       Output: /opt/failurescope_data/models/physics_results.json

  2. search-ntsb       POST to NTSB CAROL API (one query per failure component).
                       Falls back to local CSV if API unavailable.
                       Caches results to avoid re-fetching.
                       Output: /opt/failurescope_data/ntsb_cache_<component>.json

  3. match-ntsb        Score each accident record against each rule (5 dimensions).
                       Output: /opt/failurescope_data/models/ntsb_results.json

  4. rank              Combine physics × ntsb / 100. Filter combined > 60.
                       Sort by combined confidence descending.
                       (step is embedded inside --report; no separate output file)

  5. report            Generate validated_findings.json in the spec format.
                       Print findings table to stdout.
                       Output: /opt/failurescope_data/models/validated_findings.json

Usage:
  # Full pipeline (recommended):
  python train_layer4.py --all

  # Step by step:
  python train_layer4.py --validate-physics
  python train_layer4.py --search-ntsb
  python train_layer4.py --match-ntsb
  python train_layer4.py --report

  # Re-run with fresh NTSB data:
  python train_layer4.py --search-ntsb --force-refresh
  python train_layer4.py --match-ntsb --report

  # Override input path (if induced_relationships.json is elsewhere):
  python train_layer4.py --all --input /path/to/induced_relationships.json

Notes:
  - Layer 3 outputs must exist before running Layer 4:
      /opt/failurescope_data/models/induced_relationships.json
      /opt/failurescope_data/models/snapshot_analysis.json
      /opt/failurescope_data/models/cascade_attention.json
  - Run: python train_layer3.py --mine  if they are missing.
  - NTSB API uses POST with AndOr=AND field (400 without it — confirmed from probe).
  - NTSB scores default to 0 if API and CSV are both unavailable (non-blocking).
  - Do NOT retrain Layer 3 models; Layer 4 is read-only w.r.t. Layer 3 outputs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Shared constants (loaded once, used across cmds) ──────────────────────────
from backend.layer3.constants import (
    LAYER3_MODELS_DIR,
    NTSB_CACHE_PATH,
    NTSB_RESULTS_PATH,
    PHYSICS_RESULTS_PATH,
    VALIDATED_FINDINGS_PATH,
)

INDUCED_RELATIONSHIPS_PATH = f"{LAYER3_MODELS_DIR}/induced_relationships.json"


def _load_induced_relationships(args) -> list[dict]:
    """Load emergent_rules from induced_relationships.json (or --input override)."""
    path = getattr(args, "input", None) or INDUCED_RELATIONSHIPS_PATH
    if not Path(path).exists():
        log.error(
            f"induced_relationships.json not found at: {path}\n"
            "Run: python train_layer3.py --mine"
        )
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    rules = data["emergent_rules"]
    log.info(f"Loaded {len(rules)} emergent rules from {path}")
    return rules


def _check_layer3_outputs() -> None:
    """Fail fast if Layer 3 hasn't been run."""
    required = [
        f"{LAYER3_MODELS_DIR}/induced_relationships.json",
        f"{LAYER3_MODELS_DIR}/snapshot_analysis.json",
        f"{LAYER3_MODELS_DIR}/cascade_attention.json",
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        log.error(
            "Layer 3 outputs missing:\n" +
            "\n".join(f"  {p}" for p in missing) +
            "\nRun: python train_layer3.py --all"
        )
        sys.exit(1)


# ── Step 1: Physics validation ────────────────────────────────────────────────

def cmd_validate_physics(args) -> dict:
    """
    Method 2: Score each rule against C182 physical envelope + L1 consistency.
    Applies fuel_pump NOVEL→INDUCED reclassification.
    Writes: physics_results.json
    """
    log.info("\n── Step 1: Physics Validation (Method 2) ──────────────────────")
    from backend.layer4.physics_validator import load_and_validate, save_physics_results

    rules, physics_results = load_and_validate(
        induced_path=getattr(args, "input", None) or INDUCED_RELATIONSHIPS_PATH
    )

    save_physics_results(physics_results, PHYSICS_RESULTS_PATH)

    passed = sum(1 for r in physics_results.values() if r["passed"])
    log.info(f"\n✓ Physics validation complete.")
    log.info(f"  {passed}/{len(physics_results)} rules passed (threshold=80)")
    log.info(f"  Results → {PHYSICS_RESULTS_PATH}")

    return physics_results


# ── Step 2: NTSB data fetch ───────────────────────────────────────────────────

def cmd_search_ntsb(args) -> dict[str, list]:
    """
    Fetch NTSB accident data per failure component.
    Sequential (one query per component, ~13 total).
    Falls back to local CSV or empty if API unavailable.
    Writes: ntsb_cache_<component>.json (per component)
    """
    log.info("\n── Step 2: NTSB Accident Search (Method 3 data fetch) ─────────")

    try:
        import requests  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "requests",
             "--break-system-packages", "-q"], check=True
        )

    from backend.layer4.ntsb_loader import load_ntsb_per_component

    rules = _load_induced_relationships(args)
    components = [r["primary_failure"] for r in rules]

    force_refresh = getattr(args, "force_refresh", False)
    accidents_per_component = load_ntsb_per_component(
        components=components,
        force_refresh=force_refresh,
    )

    total_records = sum(len(v) for v in accidents_per_component.values())
    log.info(f"\n✓ NTSB search complete.")
    log.info(f"  {total_records} total records across {len(components)} components")
    if total_records == 0:
        log.warning(
            "  No NTSB records found. NTSB scores will be 0 for all rules.\n"
            "  Physics validation (Method 2) still runs normally.\n"
            "  See ntsb_loader.py for manual CSV download instructions."
        )

    return accidents_per_component


# ── Step 3: NTSB matching ─────────────────────────────────────────────────────

def cmd_match_ntsb(args) -> dict:
    """
    Score each NTSB accident against each rule (5 dimensions).
    Reads cached NTSB data from disk. If no cache exists, runs search first.
    Writes: ntsb_results.json
    """
    log.info("\n── Step 3: NTSB Matching (Method 3) ───────────────────────────")
    from backend.layer4.ntsb_loader import load_ntsb_per_component
    from backend.layer4.ntsb_matcher import match_rules_to_accidents

    rules = _load_induced_relationships(args)
    components = [r["primary_failure"] for r in rules]

    # Load from cache (force_refresh=False — cache was built in step 2)
    accidents_per_component = load_ntsb_per_component(
        components=components,
        force_refresh=False,
    )

    ntsb_results = match_rules_to_accidents(rules, accidents_per_component)

    # Write ntsb_results.json
    Path(NTSB_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(NTSB_RESULTS_PATH, "w") as f:
        json.dump(ntsb_results, f, indent=2)

    validated = sum(1 for r in ntsb_results.values() if r["status"] == "VALIDATED")
    log.info(f"\n✓ NTSB matching complete.")
    log.info(f"  {validated}/{len(ntsb_results)} rules have NTSB validation")
    log.info(f"  Results → {NTSB_RESULTS_PATH}")

    return ntsb_results


# ── Step 4+5: Rank + Report ───────────────────────────────────────────────────

def cmd_report(args) -> None:
    """
    Load physics_results.json + ntsb_results.json, rank, filter, write output.
    If physics_results.json is missing, runs physics validation first.
    If ntsb_results.json is missing, uses ntsb_score=0 for all rules (non-blocking).
    Writes: validated_findings.json
    Prints: findings table to stdout
    """
    log.info("\n── Step 4+5: Ranking + Report ──────────────────────────────────")
    from backend.layer4.discovery_ranker import rank_and_filter
    from backend.layer4.validation_report import generate_validated_findings, print_findings_table

    rules = _load_induced_relationships(args)

    # Load physics results (run validation if missing)
    if not Path(PHYSICS_RESULTS_PATH).exists():
        log.info("physics_results.json not found — running physics validation first")
        physics_results = cmd_validate_physics(args)
    else:
        with open(PHYSICS_RESULTS_PATH) as f:
            physics_results = json.load(f)
        log.info(f"Loaded physics results from {PHYSICS_RESULTS_PATH}")

    # Load NTSB results (default to empty if missing — non-blocking)
    if not Path(NTSB_RESULTS_PATH).exists():
        log.warning(
            "ntsb_results.json not found — NTSB scores default to 0.\n"
            "Run: python train_layer4.py --search-ntsb --match-ntsb  to populate."
        )
        ntsb_results: dict = {}
    else:
        with open(NTSB_RESULTS_PATH) as f:
            ntsb_results = json.load(f)
        log.info(f"Loaded NTSB results from {NTSB_RESULTS_PATH}")

    # Rank all rules (include below threshold for complete report)
    ranked_all = rank_and_filter(
        rules=rules,
        physics_results=physics_results,
        ntsb_results=ntsb_results,
        include_below_threshold=True,
    )

    # Generate output
    output = generate_validated_findings(
        ranked_entries=ranked_all,
        output_path=VALIDATED_FINDINGS_PATH,
        include_all=True,
    )

    print_findings_table(output)

    summary = output["summary"]
    log.info(f"\n✓ Layer 4 complete.")
    log.info(f"  {summary['n_above_threshold']} rules above threshold (combined ≥ 60)")
    log.info(f"  {summary['n_validated']} VALIDATED  "
             f"{summary['n_likely_real']} LIKELY_REAL  "
             f"{summary['n_physics_only']} PHYSICS_ONLY")
    log.info(f"  Results → {VALIDATED_FINDINGS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FailureScope Layer 4 — Failure Mode Discovery + NTSB Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_layer4.py --all\n"
            "  python train_layer4.py --validate-physics\n"
            "  python train_layer4.py --search-ntsb --force-refresh\n"
            "  python train_layer4.py --match-ntsb --report\n"
            "  python train_layer4.py --report  (physics-only, skip NTSB)\n"
        ),
    )

    parser.add_argument("--all",              action="store_true",
                        help="Run full pipeline: validate-physics → search-ntsb → match-ntsb → report")
    parser.add_argument("--validate-physics", action="store_true",
                        help="Method 2: score rules against C182 physical envelope")
    parser.add_argument("--search-ntsb",      action="store_true",
                        help="Fetch NTSB accident data per component (API + CSV fallback)")
    parser.add_argument("--match-ntsb",       action="store_true",
                        help="Method 3: score accidents against each rule")
    parser.add_argument("--report",           action="store_true",
                        help="Rank rules, generate validated_findings.json")
    parser.add_argument("--force-refresh",    action="store_true",
                        help="Ignore NTSB cache, re-fetch from API/CSV")
    parser.add_argument("--input",            type=str, default=None,
                        help=f"Path to induced_relationships.json (default: {INDUCED_RELATIONSHIPS_PATH})")

    args = parser.parse_args()

    _check_layer3_outputs()

    if args.all:
        cmd_validate_physics(args)
        cmd_search_ntsb(args)
        cmd_match_ntsb(args)
        cmd_report(args)
        return

    ran = False
    if args.validate_physics: cmd_validate_physics(args); ran = True
    if args.search_ntsb:      cmd_search_ntsb(args);      ran = True
    if args.match_ntsb:       cmd_match_ntsb(args);       ran = True
    if args.report:           cmd_report(args);            ran = True

    if not ran:
        parser.print_help()
        print("\nQuick start:  python train_layer4.py --all")
        print("Physics only: python train_layer4.py --validate-physics --report")


if __name__ == "__main__":
    main()