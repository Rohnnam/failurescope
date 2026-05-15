"""
FailureScope Edge Case Test Suite
==================================
Tests all meaningful scenario combinations against /api/run_full.
Flags cases where:
  - outcome doesn't match physical expectations
  - L3 crash_probability contradicts L2 outcome
  - TTC is physically implausible
  - L4 not found for a primary training component
  - API errors

Usage:
  python edge_case_tests.py --url http://localhost:8000 --verbose
"""
import argparse
import json
import time
import sys
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import urllib.error

# ── Expected outcomes ──────────────────────────────────────────────────────────
# (component, altitude_ft, fatigue, weather) → expected_outcome
# "any" = not deterministically known (accept either)
EXPECTED = {
    # Known recoverable: trained commercial pilot, good conditions, long recovery window
    ("alternator", 8000, 0.2, "vmc"):       "recoverable",
    ("alternator", 5000, 0.3, "vmc"):       "recoverable",
    ("vacuum_pump", 5000, 0.3, "vmc"):      "recoverable",  # VMC — gyros not needed
    # Known catastrophic: student, low alt, bad conditions
    ("alternator", 1000, 0.9, "imc"):       "catastrophic",
    ("vacuum_pump", 3000, 0.8, "imc"):      "catastrophic",  # IMC + vacuum = R014
    ("engine_left", 500,  0.9, "vmc"):      "catastrophic",  # 500ft — no altitude margin
    ("carburetor",  500,  0.7, "vmc"):      "catastrophic",
    # Dual failures — expect catastrophic except at very high altitude
    # "any" outcomes — acceptable either way
    ("fuel_selector", 5000, 0.5, "vmc"):    "any",
    ("magnetos",      5000, 0.5, "vmc"):    "any",
}

# All 14 training components — L4 MUST be found for these
L4_REQUIRED_COMPONENTS = {
    "alternator", "battery", "voltage_regulator",
    "carburetor", "engine_left", "magnetos",
    "fuel_pump", "fuel_selector", "fuel_tank_left", "fuel_tank_right",
    "vacuum_pump", "attitude_indicator", "flaps", "trim_system",
}

# ── Scenario library ───────────────────────────────────────────────────────────
SCENARIOS = [
    # ── Electrical failures ──
    ("C182 at 8000ft, fresh commercial pilot, VMC, alternator fails at cruise",
     {"expected_outcome":"recoverable", "expected_comp":"alternator"}),
    ("Cessna 182 at 5000ft, tired commercial pilot, VMC, alternator fails",
     {"expected_outcome":"recoverable", "expected_comp":"alternator"}),
    ("Cessna 182 at 2000ft, exhausted student pilot, IMC, alternator fails",
     {"expected_outcome":"catastrophic", "expected_comp":"alternator"}),
    ("C182 at 8000ft, commercial pilot, alternator and vacuum pump fail simultaneously, IMC",
     {"expected_outcome":"catastrophic", "expected_comp":"alternator"}),
    ("C182 at 5000ft, fresh pilot, battery fails at cruise",
     {"expected_outcome":"recoverable", "expected_comp":"battery"}),
    ("C182 at 1500ft approach, tired pilot, voltage regulator fails, night VMC",
     {"expected_outcome":"any", "expected_comp":"voltage_regulator"}),

    # ── Engine failures ──
    ("Cessna 182 cruise at 9000ft, commercial pilot, engine failure",
     {"expected_outcome":"any", "expected_comp":"engine_left"}),
    ("C182 500ft after takeoff, student pilot, engine failure",
     {"expected_outcome":"catastrophic", "expected_comp":"engine_left"}),
    ("C182 at 6000ft, carburetor icing, commercial pilot, moderate fatigue",
     {"expected_outcome":"any", "expected_comp":"carburetor"}),
    ("Cessna 182 low altitude 800ft, magneto failure, fatigued pilot",
     {"expected_outcome":"any", "expected_comp":"magnetos"}),

    # ── Fuel failures ──
    ("C182 at 5000ft, pilot switches to empty tank, commercial, VMC",
     {"expected_outcome":"any", "expected_comp":"fuel_selector"}),
    ("Cessna 182 cruise, fuel pump fails, fresh commercial pilot",
     {"expected_outcome":"any", "expected_comp":"fuel_pump"}),
    ("C182 at 7000ft, left fuel tank empty due to imbalance, commercial pilot",
     {"expected_outcome":"any", "expected_comp":"fuel_tank_left"}),

    # ── Vacuum/instrument failures ──
    ("C182 at 8000ft, IMC, vacuum pump failure, commercial pilot, moderate fatigue",
     {"expected_outcome":"catastrophic", "expected_comp":"vacuum_pump"}),
    ("C182 at 5000ft, VMC, vacuum pump failure, fresh instrument-rated pilot",
     {"expected_outcome":"recoverable", "expected_comp":"vacuum_pump"}),
    ("C182 attitude indicator fails, IMC cruise, tired pilot",
     {"expected_outcome":"catastrophic", "expected_comp":"attitude_indicator"}),

    # ── Controls ──
    ("C182 on approach, flap failure, commercial pilot, VMC",
     {"expected_outcome":"any", "expected_comp":"flaps"}),
    ("Cessna 182 cruise, trim system failure, commercial pilot",
     {"expected_outcome":"any", "expected_comp":"trim_system"}),

    # ── Edge cases ──
    # Very short TTC (low alt + catastrophic)
    ("Cessna 182 at 300ft final approach, engine failure, student pilot",
     {"expected_outcome":"catastrophic", "expected_comp":"engine_left", "max_ttc":60}),
    # Should be long TTC (high alt + electrical)
    ("C182 at 14000ft cruise, fresh commercial, VMC, alternator fails",
     {"expected_outcome":"recoverable", "expected_comp":"alternator"}),
    # Dual failure
    ("C182 at 5000ft, simultaneous engine failure and alternator failure, commercial pilot",
     {"expected_outcome":"catastrophic", "expected_comp":"engine_left"}),
    # Exhaustion edge
    ("Cessna 182 at 5000ft, pilot hasn't slept in 30 hours, VMC, alternator fails",
     {"expected_outcome":"any", "expected_comp":"alternator"}),
    # IMC without vacuum — spatial disorientation R014
    ("Cessna 182 in clouds at 6000ft, vacuum pump fails, no backup instruments, commercial pilot",
     {"expected_outcome":"catastrophic", "expected_comp":"vacuum_pump"}),
]

# ── HTTP helper (no requests dependency) ──────────────────────────────────────
def get_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return None, str(e)

def post_json(url, body, timeout=90):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return None, str(e)

# ── Result checker ────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    scenario: str
    passed: bool
    issues: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    outcome: Optional[str] = None
    ttc: Optional[float] = None
    l4_found: bool = True
    crash_prob: Optional[float] = None

def check(scenario_text, expectations, data, verbose):
    r = data.get("result", {})
    outcome = r.get("outcome")
    ttc = r.get("time_to_criticality_s")
    duration = r.get("duration_s", 0)
    l4 = r.get("l4_findings", {})
    l3 = r.get("l3_analysis", {})
    failure_chain = r.get("failure_chain", [])
    initial = r.get("initial", {})

    issues = []
    notes = []

    # 1. Expected outcome
    exp_out = expectations.get("expected_outcome")
    if exp_out and exp_out != "any" and outcome != exp_out:
        issues.append(f"OUTCOME MISMATCH: expected={exp_out} got={outcome}")

    # 2. Max TTC check
    max_ttc = expectations.get("max_ttc")
    if max_ttc and ttc and ttc > max_ttc:
        issues.append(f"TTC TOO HIGH: {ttc:.0f}s > max_expected={max_ttc}s")

    # 3. TTC plausibility — catastrophic must have TTC, recoverable must not
    if outcome == "catastrophic" and ttc is None:
        issues.append("CATASTROPHIC with no TTC recorded")
    if outcome == "recoverable" and ttc is not None:
        issues.append(f"RECOVERABLE but TTC={ttc:.0f}s was recorded")

    # 4. L4 required for training components
    exp_comp = expectations.get("expected_comp")
    if exp_comp and exp_comp in L4_REQUIRED_COMPONENTS and not l4.get("found"):
        issues.append(f"L4 NOT FOUND for training component '{exp_comp}'")

    # 5. L3 crash_prob vs L2 outcome (snapshot model, fast cascades only)
    crash_prob = l3.get("snapshot", {}).get("crash_probability") if l3.get("available") else None
    if crash_prob is not None and duration is not None and duration < 300:
        if outcome == "recoverable" and crash_prob > 0.8:
            issues.append(f"L3/L2 DISAGREEMENT: L2=recoverable but L3 crash_prob={crash_prob:.2f}")
        if outcome == "catastrophic" and crash_prob < 0.2:
            issues.append(f"L3/L2 DISAGREEMENT: L2=catastrophic but L3 crash_prob={crash_prob:.2f}")

    # 6. TTC physical plausibility — TTC < 5s is suspicious (physics takes time)
    if ttc is not None and ttc < 5:
        issues.append(f"SUSPICIOUSLY SHORT TTC: {ttc:.1f}s (< 5s)")

    # 7. Failure chain not empty
    if not failure_chain:
        issues.append("EMPTY FAILURE CHAIN")

    # Notes (informational, not failures)
    if ttc: notes.append(f"TTC={ttc:.0f}s")
    if crash_prob is not None: notes.append(f"crash_prob={crash_prob:.3f}")
    if l4.get("validation_status"): notes.append(f"L4={l4['validation_status']} {l4.get('combined_confidence','')}")
    if initial.get("fatigue"): notes.append(f"fatigue={initial['fatigue']:.1f}")

    return CheckResult(
        scenario=scenario_text[:70],
        passed=len(issues)==0,
        issues=issues, notes=notes,
        outcome=outcome, ttc=ttc,
        l4_found=l4.get("found", False),
        crash_prob=crash_prob,
    )

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()

    # Health check
    health_data, err = get_json(f"{args.url.rstrip('/')}/health", timeout=5)
    if err or not health_data:
        print(f"✗ Backend not reachable at {args.url}: {err}")
        sys.exit(1)

    layers = health_data.get("layers", {})
    print(f"Backend: {args.url}")
    print(f"Layers: {', '.join(k for k,v in layers.items() if v)}")
    print(f"Full pipeline: {health_data.get('full_pipeline_available')}")
    print(f"\nRunning {len(SCENARIOS)} test scenarios...\n")
    print("=" * 80)

    passed = 0
    failed = 0
    errors = 0
    results = []

    for i, (text, expectations) in enumerate(SCENARIOS):
        print(f"[{i+1:02d}/{len(SCENARIOS)}] {text[:65]}...")

        data, err = post_json(f"{args.url.rstrip('/')}/api/run_full", {"text": text})

        if err or not data:
            print(f"  ✗ API ERROR: {err}")
            errors += 1
            results.append(CheckResult(scenario=text[:70], passed=False, issues=[f"API error: {err}"]))
            time.sleep(args.delay)
            continue

        cr = check(text, expectations, data, args.verbose)
        results.append(cr)

        status = "✓ PASS" if cr.passed else "✗ FAIL"
        outcome_str = f"{cr.outcome or '?':12}"
        ttc_str = f"TTC={cr.ttc:.0f}s" if cr.ttc else "TTC=None  "
        print(f"  {status} | {outcome_str} | {ttc_str:12} | {' | '.join(cr.notes[:3])}")

        if not cr.passed:
            for issue in cr.issues:
                print(f"         ↳ {issue}")
            failed += 1
        else:
            passed += 1

        if args.verbose and cr.passed:
            for note in cr.notes:
                print(f"         · {note}")

        time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} PASS  {failed} FAIL  {errors} ERRORS  ({len(SCENARIOS)} total)")
    print("=" * 80)

    if failed > 0 or errors > 0:
        print("\nFAILED CASES:")
        for r in results:
            if not r.passed:
                print(f"\n  Scenario: {r.scenario}")
                for issue in r.issues:
                    print(f"    ✗ {issue}")

    # L4 coverage report
    l4_missing = [r for r in results if r.passed and not r.l4_found]
    if l4_missing:
        print(f"\nL4 NOT FOUND ({len(l4_missing)} scenarios — not failures, just gaps):")
        for r in l4_missing:
            print(f"  · {r.scenario}")

    return 0 if (failed == 0 and errors == 0) else 1

if __name__ == "__main__":
    sys.exit(main())