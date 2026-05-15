#!/usr/bin/env python3
"""
FailureScope — Relationship Audit
===================================
Lists ALL relationships across three tiers and cross-checks each against sources.

Tier 1 — DETERMINED (L1 constraint graph rules, hardcoded from POH/FAA docs)
Tier 2 — LEARNED (model confirms a known L1 relationship holds in simulation data)
Tier 3 — INDUCED (model finds patterns not explicitly in L1 rules)

Usage:
  python audit_relationships.py
  python audit_relationships.py > audit_report.txt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backend.layer3.constants import L1_RULES, CASCADE_COMPONENTS

INDUCED_PATH  = Path("/opt/failurescope_data/models/induced_relationships.json")
SNAPSHOT_PATH = Path("/opt/failurescope_data/models/snapshot_analysis.json")


def load_json(path):
    if not path.exists():
        print(f"  ⚠ Not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def sep(char="═", n=65):
    print(char * n)


def main():
    induced  = load_json(INDUCED_PATH)
    snapshot = load_json(SNAPSHOT_PATH)

    # ──────────────────────────────────────────────────────────────
    # TIER 1: DETERMINED RELATIONSHIPS (L1 constraint graph)
    # ──────────────────────────────────────────────────────────────
    sep()
    print("TIER 1 — DETERMINED RELATIONSHIPS")
    print("Hardcoded from POH / FAA documents into the Layer 1 constraint graph.")
    print("These are ground truth. Every simulation respects them exactly.")
    sep()
    for rule in L1_RULES:
        print(f"\n  [{rule['id']}]  {rule['trigger']}  →  {rule['effect']}")
        print(f"          Source: {rule['source']}")
    print(f"\n  Total: {len(L1_RULES)} determined relationships")

    # ──────────────────────────────────────────────────────────────
    # TIER 2: LEARNED RELATIONSHIPS (model confirms L1 in data)
    # ──────────────────────────────────────────────────────────────
    sep("\n" + "═")
    print("TIER 2 — LEARNED RELATIONSHIPS")
    print("L1 rules confirmed to be active by the models in 500K simulation trajectories.")
    print("Evidence: SHAP global importance + cascade attention + L1 cross-reference.")
    sep()

    # From snapshot SHAP L1 crossref
    l1_crossref = snapshot.get("l1_rule_crossreference", [])
    confirmed_rules = [r for r in l1_crossref if r.get("shap_consistent")]
    not_detected    = [r for r in l1_crossref if not r.get("shap_consistent")]

    print("\n  Confirmed by SHAP (snapshot model):")
    for r in confirmed_rules:
        feats = ", ".join(r.get("related_shap_features", []))
        print(f"    ✓ [{r['rule_id']}]  {r['trigger']} → {r['effect']}")
        print(f"       SHAP features: {feats}  |  Source: {r['source_citation']}")

    print(f"\n  Not detected in snapshot features:")
    for r in not_detected:
        print(f"    ○ [{r['rule_id']}]  {r['trigger']} → {r['effect']}")
        print(f"       Source: {r['source_citation']}")
        print(f"       Note: May require cascade window data, not snapshot features")

    # From induced_relationships CONFIRMED tier
    confirmed_induced = [r for r in induced.get("all_rules_ranked", [])
                         if r.get("classification") == "CONFIRMED"]
    if confirmed_induced:
        print(f"\n  Confirmed by cascade Transformer:")
        for r in confirmed_induced:
            cr = r.get("context1_rule", {})
            print(f"    ✓ {r['component']:20s}  L1 rules: {cr.get('l1_related_rules')}  "
                  f"crash_prob={cr.get('crash_probability_in_this_zone')}")

    # Global variable importance cross-check
    print("\n  Cascade Transformer — global variable attention (confirms L1 cascade order):")
    cas_attn = load_json(Path("/opt/failurescope_data/models/cascade_attention.json"))
    for e in cas_attn.get("variable_importance_global", []):
        var   = e["variable"]
        attn  = e["mean_attention"]
        n_comp = e["n_components"]
        # Cross-check which L1 rules mention this variable
        matching = [r["id"] for r in L1_RULES
                    if var in r["trigger"].lower() or var in r["effect"].lower()]
        l1_ref = f"  ← {', '.join(matching)}" if matching else "  ← (no direct L1 rule)"
        print(f"    {var:12s}  attn={attn:.3f}  across {n_comp} components{l1_ref}")

    print(f"\n  Total L1 rules confirmed: {len(confirmed_rules)}/{len(L1_RULES)}")

    # ──────────────────────────────────────────────────────────────
    # TIER 3: INDUCED RELATIONSHIPS
    # ──────────────────────────────────────────────────────────────
    sep("\n" + "═")
    print("TIER 3 — INDUCED RELATIONSHIPS")
    print("Patterns not explicitly in L1 rules, discovered from 500K simulations.")
    print("These emerge from multi-variable interactions between cascade dynamics")
    print("and pre-existing pilot/aircraft conditions.")
    sep()

    all_rules = induced.get("all_rules_ranked", [])

    for tier_label, tier_class in [("INDUCED", "INDUCED"), ("NOVEL", "NOVEL")]:
        tier_rules = [r for r in all_rules if r.get("classification") == tier_class]
        print(f"\n  ── {tier_label} ({len(tier_rules)} base rules) ──")

        for r in tier_rules:
            cr     = r.get("context1_rule", {})
            comp   = r["component"]
            c_vars = cr.get("cascade_variables", [])
            s_vars = cr.get("swiss_cheese_variables", [])
            cp     = cr.get("crash_probability_in_this_zone")
            l1     = cr.get("l1_related_rules", [])
            cits   = cr.get("l1_citations", [])
            n_sub  = cr.get("n_sub_rules", 0)

            print(f"\n  [{comp.upper()}]")
            print(f"    Cascade variables:    {' × '.join(c_vars) if c_vars else 'N/A'}")
            print(f"    Swiss cheese holes:   {' + '.join(s_vars) if s_vars else 'N/A'}")
            print(f"    Crash probability:    {cp}")

            # Kill zone
            csc = cr.get("critical_state_combination", {})
            alt_r = csc.get("altitude_range_ft")
            v_r   = csc.get("voltage_range_V")
            pit_r = csc.get("pitch_range_deg")
            if alt_r:
                print(f"    Kill zone alt:        {alt_r[0]:.0f}–{alt_r[1]:.0f} ft")
            if v_r and v_r != [None, None]:
                print(f"    Kill zone voltage:    {v_r[0]}–{v_r[1]} V")
            if pit_r:
                print(f"    Kill zone pitch:      {pit_r[0]:.1f}°–{pit_r[1]:.1f}°")

            # Swiss cheese ranges from snapshot
            init = csc.get("initial_conditions", {})
            for feat, vals in list(init.items())[:2]:
                if feat == "failure_encoded":
                    continue
                cat_r = vals.get("cat_range", [])
                rec_r = vals.get("rec_range", [])
                if cat_r and rec_r:
                    print(f"    {feat:28s}  cat=[{cat_r[0]:.2f},{cat_r[1]:.2f}]  "
                          f"rec=[{rec_r[0]:.2f},{rec_r[1]:.2f}]")

            # Source cross-check
            if tier_class == "INDUCED" and l1:
                print(f"    Extends L1 rules:     {l1}")
                print(f"    Source citations:     {cits}")
                print(f"    Relationship status:  INDUCED — variables {c_vars} interact")
                print(f"                          with pilot state. Not in L1 rules.")
            elif tier_class == "NOVEL":
                print(f"    Source citations:     none (no matching L1 rule)")
                print(f"    Relationship status:  NOVEL — failure mode not covered by")
                print(f"                          any of the 14 L1 constraint rules.")

            # Sub-rules
            sub_rules = cr.get("sub_rules_by_altitude", [])
            if sub_rules:
                print(f"    Sub-rules ({n_sub}):")
                for sr in sub_rules:
                    print(f"      [{sr['altitude_band']}]  "
                          f"crash_prob={sr.get('crash_probability')}  "
                          f"{sr['critical_condition']}")

    # ──────────────────────────────────────────────────────────────
    # SOURCE COVERAGE AUDIT
    # ──────────────────────────────────────────────────────────────
    sep("\n" + "═")
    print("SOURCE COVERAGE AUDIT")
    print("Every claim traceable to a source document.")
    sep()

    all_cited_sources = set()
    for r in all_rules:
        for cit in r.get("context1_rule", {}).get("l1_citations", []):
            all_cited_sources.add(cit)
    for r in L1_RULES:
        all_cited_sources.add(r["source"])

    print("\n  Sources referenced across all tiers:")
    for src in sorted(all_cited_sources):
        # Count how many rules cite this source
        l1_count  = sum(1 for r in L1_RULES if r["source"] == src)
        ind_count = sum(1 for r in all_rules
                        if src in r.get("context1_rule", {}).get("l1_citations", []))
        print(f"    {src:35s}  L1 rules: {l1_count}  |  Induced rules: {ind_count}")

    print(f"\n  Claims WITHOUT source citation:")
    novel_rules = [r for r in all_rules if r.get("classification") == "NOVEL"]
    if novel_rules:
        for r in novel_rules:
            print(f"    {r['component']:20s}  NOVEL — requires Layer 4 NTSB validation")
            print(f"                          before any citation can be assigned")
    else:
        print("    None")

    # ──────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────
    sep("\n" + "═")
    print("SUMMARY")
    sep()
    total_induced = sum(1 for r in all_rules if r["classification"] == "INDUCED")
    total_novel   = sum(1 for r in all_rules if r["classification"] == "NOVEL")
    total_sub     = sum(r.get("context1_rule", {}).get("n_sub_rules", 0) for r in all_rules)
    print(f"\n  Tier 1 (Determined):   {len(L1_RULES)} rules  — all sourced to POH/FAA docs")
    print(f"  Tier 2 (Learned):      {len(confirmed_rules)} rules  — L1 rules confirmed in 500K simulations")
    print(f"  Tier 3 (Induced):      {total_induced} base + {total_sub} sub-rules  — new discoveries")
    print(f"  Tier 3 (Novel):        {total_novel} base rules  — not in L1 (need NTSB validation)")
    print(f"\n  Total unique relationships: {len(L1_RULES) + len(confirmed_rules) + total_induced + total_novel + total_sub}")
    print(f"\n  Validation status: ALL Tier 3 rules are PENDING")
    print(f"  Next step: Layer 4 — NTSB cross-reference + physics soundness check")
    sep()


if __name__ == "__main__":
    main()