"""
Layer 2: Scenario Generator
Generates scenarios for the 500K simulation sweep.

PILOT SKILL DISTRIBUTION:
  Source: FAA Civil Airmen Statistics 2025, Table 1
  URL: https://www.faa.gov/data_research/aviation_data_statistics/civil_airmen_statistics
  Date: December 31, 2025

  Airplane certificate holders used as GA proxy:
    Student:    370,286  → 43.8%
    Private:    174,155  → 20.6%
    Commercial: 118,314  → 14.0%
    ATP:        181,742  → 21.5%
  Total: 844,497

FATIGUE:
  Uniform sampling over [0.0, 1.0].
  No certificate-specific ranges assumed — simulation physics
  determine outcomes, not pre-assigned fatigue bands.

ALTITUDE:
  [500, 16500]ft — VS0 floor to C182 service ceiling (POH p4)

FUEL:
  [5, 61]gal — FAA reserve minimum to full standard tanks (POH p9)

TIMING:
  [0, 900]s — failure can occur anywhere in 0–15 min window

SAMPLING: Latin Hypercube Sampling (McKay et al. 1979, Technometrics 21:239)
  Guarantees full coverage with no gaps.
"""

import hashlib
import itertools
import json
import random
from typing import Generator


FAILURE_COMPONENTS = [
    "alternator", "battery", "fuel_pump", "engine_left",
    "vacuum_pump", "fuel_selector", "magnetos", "carburetor",
    "voltage_regulator", "fuel_tank_left", "fuel_tank_right",
    "attitude_indicator", "flaps", "trim_system",
]

# FAA Civil Airmen Statistics 2025, Table 1
_FAA_2025_COUNTS = {
    "student":    370_286,
    "private":    174_155,
    "commercial": 118_314,
    "atp":        181_742,
}
_TOTAL_PILOTS = sum(_FAA_2025_COUNTS.values())
SKILL_WEIGHTS = {k: v / _TOTAL_PILOTS for k, v in _FAA_2025_COUNTS.items()}

WEATHER_CONDITIONS = ["vmc", "imc", "vmc_turbulent", "night_vmc"]
WEATHER_FATIGUE_PENALTY = {
    "vmc": 0.0, "imc": 0.2, "vmc_turbulent": 0.1, "night_vmc": 0.15
}

# Altitude to airspeed mapping
def _airspeed(alt):
    if alt < 3000:   return 75.0
    if alt < 8000:   return 100.0
    return 120.0


def _lhs(n: int, lo: float, hi: float, seed: int) -> list:
    """1D Latin Hypercube: n stratified samples in [lo, hi]."""
    rng = random.Random(seed)
    interval = (hi - lo) / n
    pts = [lo + (i + rng.random()) * interval for i in range(n)]
    rng.shuffle(pts)
    return pts


def _scenario_id(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def is_plausible_combination(primary, secondary, tertiary=None):
    parts = [c for c in [primary, secondary, tertiary] if c]
    if len(parts) != len(set(parts)):
        return False
    impossible = {
        frozenset(["battery", "alternator"]),
        frozenset(["engine_left", "fuel_pump"]),
        frozenset(["engine_left", "alternator"]),
    }
    return frozenset([primary, secondary]) not in impossible


def generate_scenarios(
    max_scenarios: int = 500_000,
    single_failure_only: bool = False,
) -> Generator[dict, None, None]:
    """
    Generate scenarios using LHS sampling.
    Pilot skill weighted by FAA 2025 certificate distribution.
    Fatigue uniform [0.0, 1.0] — no certificate-specific assumptions.
    """
    n_comp = len(FAILURE_COMPONENTS)
    n_per  = max(1, max_scenarios // n_comp)  # min 1 per component

    # Pre-generate LHS samples for continuous params (per component, same seed offset)
    count = 0
    for comp_idx, component in enumerate(FAILURE_COMPONENTS):
        seed_base = comp_idx * 1000

        alts     = _lhs(n_per, 500.0,   16500.0, seed_base)
        fuels    = _lhs(n_per, 5.0,     61.0,    seed_base + 1)
        timings  = _lhs(n_per, 0.0,     900.0,   seed_base + 2)
        fatigues = _lhs(n_per, 0.0,     1.0,     seed_base + 3)

        # Pilot skills weighted by FAA 2025
        rng = random.Random(seed_base + 4)
        skills_list = list(SKILL_WEIGHTS.keys())
        weights_list = list(SKILL_WEIGHTS.values())
        skills = rng.choices(skills_list, weights=weights_list, k=n_per)

        # Weather cycles for coverage
        weathers = [WEATHER_CONDITIONS[i % len(WEATHER_CONDITIONS)]
                    for i in range(n_per)]

        for i in range(n_per):
            weather = weathers[i]
            fatigue = min(1.0, fatigues[i] + WEATHER_FATIGUE_PENALTY[weather])
            alt     = alts[i]

            p = {
                "c": component, "i": comp_idx * n_per + i,
                "a": round(alt, 0), "t": round(timings[i], 0),
            }
            yield {
                "scenario_id":          _scenario_id(p),
                "initial_altitude_ft":  round(alt, 1),
                "initial_airspeed_kts": _airspeed(alt),
                "initial_fuel_gal":     round(fuels[i], 1),
                "pilot_skill":          skills[i],
                "pilot_fatigue":        round(fatigue, 3),
                "weather":              weather,
                "failures": [{
                    "component":      component,
                    "trigger_time_s": round(timings[i], 1),
                    "severity":       "catastrophic",
                }],
            }
            count += 1
            if count >= max_scenarios:
                return

    if single_failure_only:
        return

    # Dual failure scenarios
    dual_count = 0
    for primary in FAILURE_COMPONENTS[:8]:
        for secondary in FAILURE_COMPONENTS[:8]:
            if not is_plausible_combination(primary, secondary):
                continue
            seed_d = hash((primary, secondary)) % 100000
            n_dual = 200
            alts_d     = _lhs(n_dual, 500.0,  16500.0, seed_d)
            fuels_d    = _lhs(n_dual, 5.0,    61.0,    seed_d+1)
            timings_d  = _lhs(n_dual, 0.0,    600.0,   seed_d+2)
            timings_d2 = _lhs(n_dual, 0.0,    900.0,   seed_d+3)
            fatigues_d = _lhs(n_dual, 0.0,    1.0,     seed_d+4)
            rng_d = random.Random(seed_d+5)
            skills_d = rng_d.choices(skills_list, weights=weights_list, k=n_dual)

            for i in range(n_dual):
                t1, t2 = sorted([timings_d[i], timings_d2[i]])
                weather = WEATHER_CONDITIONS[i % 4]
                fatigue = min(1.0, fatigues_d[i] + WEATHER_FATIGUE_PENALTY[weather])
                alt = alts_d[i]
                p = {"p": primary, "s": secondary, "i": dual_count + i}
                yield {
                    "scenario_id":          _scenario_id(p),
                    "initial_altitude_ft":  round(alt, 1),
                    "initial_airspeed_kts": _airspeed(alt),
                    "initial_fuel_gal":     round(fuels_d[i], 1),
                    "pilot_skill":          skills_d[i],
                    "pilot_fatigue":        round(fatigue, 3),
                    "weather":              weather,
                    "failures": [
                        {"component": primary,   "trigger_time_s": round(t1, 1), "severity": "catastrophic"},
                        {"component": secondary, "trigger_time_s": round(t2, 1), "severity": "catastrophic"},
                    ],
                }
                count += 1
                if count >= max_scenarios:
                    return
            dual_count += n_dual


def count_scenarios(single_failure_only: bool = False) -> dict:
    n_comp   = len(FAILURE_COMPONENTS)
    n_per    = 500_000 // n_comp
    single   = n_comp * n_per
    dual_est = 8 * 7 * 200
    return {
        "single_failure": single,
        "dual_failure":   dual_est,
        "total":          single + (0 if single_failure_only else dual_est),
        "single_lhs":     single,
        "total_lhs":      single + dual_est,
        "single_grid":    0,
    }