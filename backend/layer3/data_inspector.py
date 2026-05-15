"""
FailureScope Layer 3 — Data Inspector
=======================================
Run this BEFORE any preprocessing or training.

Validates:
  1. Cascade duration distribution (are there enough slow cascades?)
  2. State variable change during cascade (does voltage actually drop?)
  3. Failure chain completeness (do downstream events get recorded?)
  4. Class balance and failure type distribution
  5. Swiss cheese candidates (same failure, different outcomes)

Usage:
  python inspect_data.py

This produces:
  /opt/failurescope_data/models/data_inspection.json
  Plus printed summary with go/no-go recommendation for training
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.layer3.constants import (
    CASCADE_COMPONENTS, FAILURE_COMPONENTS, FAILURE_TO_IDX,
    STATE_KEYS, TRAJECTORIES_PATH_STR,
)

OUTPUT_PATH = Path("/opt/failurescope_data/models/data_inspection.json")
SAMPLE_SIZE = 50_000   # inspect first 50K for speed


def inspect(n: int = SAMPLE_SIZE) -> dict:
    path = Path(TRAJECTORIES_PATH_STR)
    if not path.exists():
        raise FileNotFoundError(f"Trajectories not found: {path}")

    print(f"[Inspector] Scanning first {n:,} trajectories …")

    # ── Accumulators ──────────────────────────────────────────────────────────
    outcomes        = Counter()
    failure_counts  = Counter()
    cascade_durs    = {"cat": [], "rec": []}
    inject_times    = []
    chain_lengths   = []
    state_change_at_cascade: Dict[str, list] = defaultdict(list)  # var → |change|
    swiss_cheese: Dict[str, Dict] = defaultdict(lambda: {"cat": 0, "rec": 0})
    slow_cascade_examples = []  # trajectories with cascade > 200s

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= n:
                break
            if line_idx % 5000 == 0:
                print(f"\r  Scanning: {line_idx:,}/{n:,} ({line_idx/n*100:.0f}%)", end="", flush=True)
            try:
                rec     = json.loads(line)
                outcome = rec.get("outcome", "recoverable")
                fails   = rec.get("failures", [])
                states  = rec.get("states", [])
                ttc_s   = float(rec.get("ttc_s") or 0)
                dur_s   = float(rec.get("duration_s") or 0)

                outcomes[outcome] += 1

                if not fails or not states:
                    continue

                comp     = fails[0].get("component", "unknown")
                inject_t = float(fails[0].get("trigger_time_s", 0))
                failure_counts[comp] += 1
                inject_times.append(inject_t)
                chain_lengths.append(len(rec.get("failure_chain", [])))

                # Cascade duration
                if outcome == "catastrophic":
                    cascade = ttc_s - inject_t
                    cascade_durs["cat"].append(cascade)

                    # Swiss cheese: same component, different outcomes
                    init = rec.get("init", {})
                    alt_bin = int(float(init.get("altitude_ft", 5000)) // 2000) * 2000
                    key = f"{comp}_alt{alt_bin}"
                    swiss_cheese[key]["cat"] += 1

                    # State change during cascade
                    # Find state at inject_t and at ttc_s
                    inject_state = None
                    for s in states:
                        if float(s.get("t", 0)) >= inject_t:
                            inject_state = s
                            break
                    if inject_state and len(states) > 0:
                        final_state = states[-1]
                        for var in ("v", "ff", "pump_eff", "rpm", "alt"):
                            v0 = float(inject_state.get(var, 0))
                            v1 = float(final_state.get(var, 0))
                            state_change_at_cascade[var].append(abs(v1 - v0))

                    # Collect slow cascade examples
                    if cascade > 200 and len(slow_cascade_examples) < 3:
                        slow_cascade_examples.append({
                            "id":        rec.get("scenario_id"),
                            "comp":      comp,
                            "inject_t":  inject_t,
                            "ttc_s":     ttc_s,
                            "cascade_s": cascade,
                            "chain":     rec.get("failure_chain", []),
                            "alt_init":  float(rec.get("init", {}).get("altitude_ft", 0)),
                            "state_at_inject": {
                                k: float(inject_state.get(k, 0))
                                for k in ("v", "ff", "pump_eff", "rpm")
                            } if inject_state else {},
                            "state_at_crash": {
                                k: float(states[-1].get(k, 0))
                                for k in ("v", "ff", "pump_eff", "rpm")
                            } if states else {},
                        })

                else:
                    cascade = dur_s - inject_t
                    cascade_durs["rec"].append(max(cascade, 0))

                    key = f"{comp}_alt{int(float(rec.get('init',{}).get('altitude_ft',5000))//2000)*2000}"
                    swiss_cheese[key]["rec"] += 1

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    # ── Compute summaries ──────────────────────────────────────────────────────
    cat_c = np.array(cascade_durs["cat"])
    rec_c = np.array(cascade_durs["rec"])
    inj   = np.array(inject_times)
    clen  = np.array(chain_lengths)

    cascade_summary = {
        "catastrophic": {
            "n": len(cat_c),
            "mean": round(float(cat_c.mean()), 1) if len(cat_c) else 0,
            "median": round(float(np.median(cat_c)), 1) if len(cat_c) else 0,
            "max": round(float(cat_c.max()), 1) if len(cat_c) else 0,
            "pct_over_30s":  round(float((cat_c > 30).mean() * 100), 1),
            "pct_over_60s":  round(float((cat_c > 60).mean() * 100), 1),
            "pct_over_200s": round(float((cat_c > 200).mean() * 100), 1),
            "pct_over_300s": round(float((cat_c > 300).mean() * 100), 1),
        },
        "recoverable": {
            "n": len(rec_c),
            "mean": round(float(rec_c.mean()), 1) if len(rec_c) else 0,
        },
    }

    state_change_summary = {
        var: {
            "mean_change": round(float(np.mean(vals)), 3) if vals else 0,
            "pct_nonzero": round(float((np.array(vals) > 0.01).mean() * 100), 1) if vals else 0,
        }
        for var, vals in state_change_at_cascade.items()
    }

    # Swiss cheese: find comp/altitude combos where BOTH outcomes exist
    swiss_cheese_candidates = [
        {
            "scenario": k,
            "catastrophic": v["cat"],
            "recoverable": v["rec"],
            "ratio": round(v["cat"] / max(v["rec"], 1), 2),
        }
        for k, v in swiss_cheese.items()
        if v["cat"] > 5 and v["rec"] > 5
    ]
    swiss_cheese_candidates.sort(key=lambda x: -(x["catastrophic"] + x["recoverable"]))

    inject_summary = {
        "pct_at_t0":     round(float((inj < 1).mean() * 100), 1),
        "pct_before_100s": round(float((inj < 100).mean() * 100), 1),
        "pct_after_300s":  round(float((inj > 300).mean() * 100), 1),
        "median_inject_t": round(float(np.median(inj)), 1),
    }

    chain_summary = {
        str(v): int((clen == v).sum())
        for v in range(7)
    }

    # ── Go / No-Go assessment ──────────────────────────────────────────────────
    n_slow_cat = int((cat_c > 60).sum()) if len(cat_c) else 0
    projected_full = int(n_slow_cat * (500_000 / n))

    go_nogo = {
        "snapshot_model": {
            "go": True,
            "reason": f"All {outcomes['catastrophic']:,} cat + {outcomes['recoverable']:,} rec trajectories usable",
        },
        "cascade_transformer": {
            "go": n_slow_cat > 100,
            "n_usable": projected_full,
            "reason": (
                f"~{projected_full:,} trajectories with cascade>60s projected in full dataset. "
                + ("✓ Sufficient for Transformer training."
                   if projected_full > 5000
                   else "⚠ May be insufficient — consider lowering MIN_CASCADE_DURATION_S.")
            ),
        },
        "state_variables_change": {
            "go": state_change_summary.get("v", {}).get("pct_nonzero", 0) > 10,
            "voltage_changes_in_pct": state_change_summary.get("v", {}).get("pct_nonzero", 0),
            "pump_eff_changes_in_pct": state_change_summary.get("pump_eff", {}).get("pct_nonzero", 0),
        },
    }

    result = {
        "n_inspected":            n,
        "outcome_counts":         dict(outcomes),
        "failure_type_counts":    dict(failure_counts.most_common()),
        "cascade_duration":       cascade_summary,
        "state_change_during_cascade": state_change_summary,
        "inject_time_distribution":    inject_summary,
        "failure_chain_lengths":       chain_summary,
        "swiss_cheese_candidates":     swiss_cheese_candidates[:20],
        "slow_cascade_examples":       slow_cascade_examples,
        "go_nogo":                     go_nogo,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print()  # newline after progress counter
    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DATA INSPECTION RESULTS")
    print(f"{'='*60}")
    print(f"Outcomes: {dict(outcomes)}")
    print(f"\nCascade duration (catastrophic):")
    cs = cascade_summary["catastrophic"]
    print(f"  mean={cs['mean']}s  median={cs['median']}s  max={cs['max']}s")
    print(f"  >30s: {cs['pct_over_30s']}%  >60s: {cs['pct_over_60s']}%  >200s: {cs['pct_over_200s']}%")
    print(f"\nState variable changes DURING cascade:")
    for var, chg in state_change_summary.items():
        print(f"  {var:12s}: mean_change={chg['mean_change']:.3f}  "
              f"pct_nonzero={chg['pct_nonzero']:.1f}%")
    print(f"\nSwiss cheese candidates (same failure, different outcomes): "
          f"{len(swiss_cheese_candidates)}")
    print(f"\nGO/NO-GO:")
    for model, status in go_nogo.items():
        icon   = "✓" if status.get("go") else "✗"
        reason = status.get("reason") or str({k: v for k, v in status.items() if k != "go"})
        print(f"  {icon} {model}: {reason}")
    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"{'='*60}")

    if slow_cascade_examples:
        print(f"\nSLOW CASCADE EXAMPLES (cascade > 200s):")
        for ex in slow_cascade_examples:
            print(f"\n  {ex['comp']} | inject_t={ex['inject_t']:.0f}s | "
                  f"ttc_s={ex['ttc_s']:.0f}s | cascade={ex['cascade_s']:.0f}s")
            print(f"  chain: {ex['chain']}")
            si = ex.get('state_at_inject', {})
            sc = ex.get('state_at_crash', {})
            if si and sc:
                print(f"  v:        {si.get('v',0):.1f} → {sc.get('v',0):.1f} V")
                print(f"  ff:       {si.get('ff',0):.1f} → {sc.get('ff',0):.1f} GPH")
                print(f"  pump_eff: {si.get('pump_eff',0):.3f} → {sc.get('pump_eff',0):.3f}")
                print(f"  rpm:      {si.get('rpm',0):.0f} → {sc.get('rpm',0):.0f}")

    return result


if __name__ == "__main__":
    inspect()