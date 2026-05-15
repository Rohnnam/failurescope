"""
Layer 0: Scenario Validator
Validates parsed scenario JSON against physical constraints.
Source: context1 [Layer 0: Hallucination Prevention]
Source: context2 [Stage 1: All values validated for physical plausibility]

Risk level: NONE — deterministic range checks, no ML involved.
"""

from loguru import logger
from backend.layer0.defaults import AIRCRAFT_DEFAULTS, VALIDATION_RANGES
from backend.layer0.schemas import ScenarioInput, ValidationResult, ValidationError


class ScenarioValidator:
    """
    Validates a parsed ScenarioInput against physical constraints.

    All checks are deterministic. Either a value is in range or it isn't.
    No inference. No learning. No hallucination possible.
    """

    def validate(self, scenario: ScenarioInput) -> ValidationResult:
        errors = []
        warnings = []

        aircraft_key = scenario.aircraft_type
        defaults = AIRCRAFT_DEFAULTS.get(aircraft_key)

        if defaults is None:
            errors.append(ValidationError(
                field="aircraft_type",
                issue=f"Unknown aircraft type: {aircraft_key}",
                value=aircraft_key
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        s = scenario.initial_state

        # ── Altitude checks ──────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["altitude_ft"]
        if not (lo <= s.altitude_ft <= hi):
            errors.append(ValidationError(
                field="altitude_ft",
                issue=f"Altitude {s.altitude_ft} ft outside physical range [{lo}, {hi}]",
                value=s.altitude_ft
            ))

        max_alt = defaults["max_altitude_ft"]
        if s.altitude_ft > max_alt:
            warnings.append(
                f"Altitude {s.altitude_ft} ft exceeds {aircraft_key} "
                f"service ceiling {max_alt} ft. Simulation may behave unusually."
            )

        # ── Airspeed checks ───────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["airspeed_knots"]
        if not (lo <= s.airspeed_knots <= hi):
            errors.append(ValidationError(
                field="airspeed_knots",
                issue=f"Airspeed {s.airspeed_knots} kts outside range [{lo}, {hi}]",
                value=s.airspeed_knots
            ))

        vne = defaults["vne_knots"]
        vs0 = defaults["vs0_knots"]
        if s.airspeed_knots > vne:
            errors.append(ValidationError(
                field="airspeed_knots",
                issue=f"Airspeed {s.airspeed_knots} kts exceeds Vne ({vne} kts) for {aircraft_key}",
                value=s.airspeed_knots
            ))
        if 0 < s.airspeed_knots < vs0 and s.altitude_ft > 500:
            warnings.append(
                f"Airspeed {s.airspeed_knots} kts below Vs0 ({vs0} kts) — "
                f"aircraft would be stalled at this altitude."
            )

        # ── Attitude checks ───────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["pitch_deg"]
        if not (lo <= s.pitch_deg <= hi):
            errors.append(ValidationError(
                field="pitch_deg",
                issue=f"Pitch {s.pitch_deg}° outside physical range [{lo}, {hi}]",
                value=s.pitch_deg
            ))

        lo, hi = VALIDATION_RANGES["roll_deg"]
        if not (lo <= s.roll_deg <= hi):
            errors.append(ValidationError(
                field="roll_deg",
                issue=f"Roll {s.roll_deg}° outside physical range [{lo}, {hi}]",
                value=s.roll_deg
            ))

        # ── Fuel checks ───────────────────────────────────────────────
        max_left = defaults["fuel_capacity_left_gal"]
        max_right = defaults["fuel_capacity_right_gal"]

        if s.fuel_left_gal > max_left:
            errors.append(ValidationError(
                field="fuel_left_gal",
                issue=f"Left fuel {s.fuel_left_gal} gal exceeds tank capacity {max_left} gal",
                value=s.fuel_left_gal
            ))
        if s.fuel_right_gal > max_right:
            errors.append(ValidationError(
                field="fuel_right_gal",
                issue=f"Right fuel {s.fuel_right_gal} gal exceeds tank capacity {max_right} gal",
                value=s.fuel_right_gal
            ))

        total_fuel = s.fuel_left_gal + s.fuel_right_gal
        fuel_flow = defaults["cruise_fuel_flow_gph"]
        endurance_hrs = total_fuel / fuel_flow if fuel_flow > 0 else 0
        if endurance_hrs < 0.25 and s.altitude_ft > 1000:
            warnings.append(
                f"Total fuel {total_fuel:.1f} gal gives only "
                f"{endurance_hrs * 60:.0f} min endurance at cruise. "
                f"Simulation window may be very short."
            )

        # ── Electrical checks ─────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["electrical_voltage"]
        if not (lo <= s.electrical_voltage <= hi):
            errors.append(ValidationError(
                field="electrical_voltage",
                issue=f"Voltage {s.electrical_voltage}V outside range [{lo}, {hi}]",
                value=s.electrical_voltage
            ))

        lo, hi = VALIDATION_RANGES["battery_charge"]
        if not (lo <= s.battery_charge <= hi):
            errors.append(ValidationError(
                field="battery_charge",
                issue=f"Battery charge {s.battery_charge}% outside range [{lo}, {hi}]",
                value=s.battery_charge
            ))

        # ── RPM checks ────────────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["engine_rpm"]
        if not (lo <= s.engine_rpm_left <= hi):
            errors.append(ValidationError(
                field="engine_rpm_left",
                issue=f"RPM {s.engine_rpm_left} outside range [{lo}, {hi}]",
                value=s.engine_rpm_left
            ))

        # ── Pilot state checks ────────────────────────────────────────
        lo, hi = VALIDATION_RANGES["fatigue"]
        fatigue = scenario.pilot_state.fatigue
        if not (lo <= fatigue <= hi):
            errors.append(ValidationError(
                field="fatigue",
                issue=f"Fatigue {fatigue} outside range [{lo}, {hi}]",
                value=fatigue
            ))

        # ── Failure event checks ──────────────────────────────────────
        lo, hi = VALIDATION_RANGES["trigger_time_seconds"]
        for i, failure in enumerate(scenario.failures):
            if not (lo <= failure.trigger_time_seconds <= hi):
                errors.append(ValidationError(
                    field=f"failures[{i}].trigger_time_seconds",
                    issue=f"Trigger time {failure.trigger_time_seconds}s outside range [{lo}, {hi}]",
                    value=failure.trigger_time_seconds
                ))

        # ── Multi-engine consistency ──────────────────────────────────
        if defaults["engine_count"] == 1 and s.engine_rpm_right > 0:
            warnings.append(
                f"{aircraft_key} is a single-engine aircraft. "
                f"engine_rpm_right will be ignored in simulation."
            )

        is_valid = len(errors) == 0
        logger.debug(f"Validation result: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )