"""
Layer 2: Core Aircraft Simulator
Implements the deterministic simulation loop.

Source: context2 [Stage 4: Physics Simulation]
  At each time step (dt = 0.1 seconds):
    1. Apply failures (if time matches)
    2. Propagate constraints (cascade rules from Layer 1)
    3. Compute aircraft state (physics equations)
    4. Store state snapshot
    5. Check outcome

Hallucination Risk: NONE
  - Deterministic: same input → same output always
  - Physics equations from FAA-H-8083-25C (cited in physics_engine.py)
  - Cascade rules from constraint_graph.json (Layer 1)
  - No learning, no approximation beyond numerical integration
"""

import copy
import math
from dataclasses import replace
from loguru import logger

from backend.layer1.graph_loader import load_graph
from backend.layer2.physics_engine import (
    lift_force, drag_force, engine_thrust, rpm_from_fuel_flow,
    vertical_speed, stall_speed_fps, is_stalled, is_unrecoverable,
    pilot_response_delay, pilot_correction_effectiveness,
    knots_to_fps, fps_to_knots, C182, air_density,
)
from backend.layer2.state import AircraftState, TrajectoryRecord, SimOutcome
def _pilot_can_land(altitude_agl_ft: float, fatigue: float,
                    aware: bool, time_since_aware_s: float) -> bool:
    if not aware: return False
    if altitude_agl_ft < 500: return False
    if fatigue > 0.85: return False
    return altitude_agl_ft >= 1000 and time_since_aware_s >= 30.0




# ─────────────────────────────────────────────────────────────
# SIMULATION CONSTANTS
# Source: context2 [Stage 4: Loop for 1200 seconds]
# ─────────────────────────────────────────────────────────────

DT              = 0.1      # Time step, seconds
MAX_DURATION    = 3600.0   # 60 minutes — battery depletion cascades need time to develop
STORE_INTERVAL  = 1        # Store every N steps (1 = every 0.1s)
TERRAIN_ALT_FT  = 0.0      # Ground level (MSL for flat terrain simulation)

# Battery discharge constants
# Source: Service manual D2006R4-13 p248 (33Ah battery)
# physics_cache: system_voltage=14V, battery_capacity_ah=33
TYPICAL_ELECTRICAL_LOAD_A = 50.0   # Emergency load: fuel pump(5A) + avionics(15A) + trim(5A) + lights(10A) + pitot heat(5A) + margin = ~50A
BATTERY_CAPACITY_AH       = 33.0   # From physics_cache
R_BATTERY_INTERNAL_OHM = 0.015  # Internal resistance, 33Ah sealed lead-acid aviation battery
                                  # Source: MIL-SPEC-21030 / Gill G-35 battery spec


class Simulator:
    """
    Deterministic aircraft simulator for Cessna 182Q.

    Runs one scenario from initial state to outcome.
    Applies failures, propagates cascade rules, computes physics.

    Source: context2 [Stage 4: complete processing pipeline]
    """

    def __init__(self, aircraft_type: str = "cessna_182"):
        self.graph = load_graph(aircraft_type)
        self.aircraft_type = aircraft_type

    def run(self, scenario: dict) -> TrajectoryRecord:
        """
        Run a single simulation scenario.

        Args:
            scenario: dict with keys:
              scenario_id, initial_altitude_ft, initial_airspeed_kts,
              initial_fuel_gal, pilot_fatigue, pilot_skill, weather,
              failures: [{component, trigger_time_s, severity}]

        Returns:
            TrajectoryRecord with complete trajectory and outcome
        """
        # Build initial state from scenario
        state = self._build_initial_state(scenario)

        trajectory = TrajectoryRecord(
            scenario_id          = scenario.get("scenario_id", ""),
            aircraft_type        = self.aircraft_type,
            initial_altitude_ft  = state.altitude_ft,
            initial_airspeed_kts = state.airspeed_knots,
            initial_fuel_gal     = state.fuel_left_gal + state.fuel_right_gal,
            pilot_skill          = scenario.get("pilot_skill", "commercial"),
            pilot_fatigue        = state.pilot_fatigue,
            weather              = scenario.get("weather", "vmc"),
            failures             = scenario.get("failures", []),
            dt                   = DT,
        )

        failures = scenario.get("failures", [])
        failure_chain = []
        cascade_recorded = set()   # tracks cascade-induced events already appended
        time_to_criticality = None

        t = 0.0
        step = 0

        while t <= MAX_DURATION:
            # ── STEP 1: Apply failures ─────────────────────────────
            for f in failures:
                trigger_t = f.get("trigger_time_s", 0.0)
                component = f.get("component", "")
                severity  = f.get("severity", "catastrophic")

                if abs(t - trigger_t) < DT * 0.5:
                    state = self._apply_failure(state, component, severity, t)
                    if component not in [x["component"] for x in failure_chain]:
                        failure_chain.append({
                            "component": component,
                            "time_s":    round(t, 1),
                            "severity":  severity,
                        })

            # ── STEP 2: Propagate cascade rules ───────────────────
            state = self._propagate_cascades(state, t)

            # Record cascade-induced failures (components that failed due to propagation,
            # not direct injection). These are the events that make the timeline meaningful.
            # Source: cascade rules R001–R014 (POH §7.1, FAA AC 43.13-1B)
            _injected = {f.get("component") for f in failures}
            _cascade_checks = [
                ("fuel_pump",    state.fuel_pump_status    in ("failed","degraded"),  state.fuel_pump_status),
                ("engine_left",  state.engine_status       in ("failed","degraded"),  state.engine_status),
                ("vacuum_pump",  state.vacuum_pump_status  == "failed",               "failed"),
                ("attitude_indicator", state.attitude_indicator_status == "unreliable", "degraded"),
                ("battery",      state.battery_status      == "failed",               "failed"),
                ("alternator",   state.alternator_status   == "failed" and "alternator" not in _injected, "failed"),
            ]
            for comp, fired, severity in _cascade_checks:
                if fired and comp not in cascade_recorded and comp not in _injected:
                    cascade_recorded.add(comp)
                    failure_chain.append({
                        "component": comp,
                        "time_s":    round(t, 1),
                        "severity":  severity,
                        "cascade":   True,   # distinguishes from injected failures in frontend
                    })

            # ── STEP 3: Update pilot awareness ────────────────────
            state = self._update_pilot(state, t, failures)

            # ── STEP 3b: Update time in state ─────────────────────
            state.time_s = t

            # ── STEP 4: Compute physics ────────────────────────────
            state = self._compute_physics(state, t)

            # ── STEP 5: Store state snapshot ──────────────────────
            if step % STORE_INTERVAL == 0:
                trajectory.states.append(state.to_dict())

            # ── STEP 6: Check outcome ──────────────────────────────
            state.altitude_agl_ft = state.altitude_ft - TERRAIN_ALT_FT
            unrecov, reason = is_unrecoverable(
                state.altitude_ft, state.airspeed_fps,
                state.roll_deg, state.pitch_deg,
                state.voltage, state.fuel_flow_gph,
                state.altitude_agl_ft,
            )

            if unrecov:
                # Check if pilot can execute forced landing before declaring catastrophic
                can_land = _pilot_can_land(
                    state.altitude_agl_ft,
                    state.pilot_fatigue,
                    state.pilot_aware_of_failure,
                    state.time_since_failure_s,
                )
                if can_land and reason not in ("ground_impact", "spiral_dive_roll_exceeded"):
                    # Pilot executes successful emergency landing
                    state.outcome = SimOutcome.RECOVERABLE
                    trajectory.states.append(state.to_dict())
                    break
                else:
                    state.outcome = SimOutcome.CATASTROPHIC
                    state.unrecoverable_reason = reason
                    if time_to_criticality is None:
                        time_to_criticality = t
                    trajectory.states.append(state.to_dict())
                    break

            t  = round(t + DT, 2)
            step += 1

        # ── FINALIZE TRAJECTORY ────────────────────────────────────
        if state.outcome == SimOutcome.IN_PROGRESS:
            # Simulation ran full duration — aircraft survived
            state.outcome = SimOutcome.RECOVERABLE
            trajectory.outcome = SimOutcome.RECOVERABLE
            # Recovery margin = remaining time at end
            trajectory.recovery_margin_s = MAX_DURATION - t
        else:
            trajectory.outcome = state.outcome

        trajectory.time_to_criticality_s = time_to_criticality
        trajectory.failure_chain         = failure_chain
        trajectory.unrecoverable_reason  = state.unrecoverable_reason
        trajectory.simulation_duration_s = t
        trajectory.total_time_steps      = step

        return trajectory

    # ─────────────────────────────────────────────────────────
    # FAILURE APPLICATION
    # Sets component status in state based on failure type
    # ─────────────────────────────────────────────────────────

    def _apply_failure(
        self, state: AircraftState,
        component: str, severity: str, t: float
    ) -> AircraftState:
        """Apply a failure to the aircraft state."""
        if component == "alternator":
            state.alternator_status = "failed"
            state.alternator_on = False
            logger.debug(f"t={t:.1f}s: Alternator FAILED")

        elif component == "battery":
            state.battery_status = "failed"
            state.battery_charge_ah = 0.0
            logger.debug(f"t={t:.1f}s: Battery FAILED")

        elif component == "fuel_pump":
            if severity == "catastrophic":
                state.fuel_pump_status = "failed"
                state.fuel_pump_efficiency = 0.0
            else:
                state.fuel_pump_status = "degraded"
                state.fuel_pump_efficiency = 0.5
            logger.debug(f"t={t:.1f}s: Fuel pump {severity}")

        elif component in ("engine_left", "engine_both", "engine"):
            if severity == "catastrophic":
                state.engine_status = "failed"
                state.engine_rpm = 0.0
                state.fuel_flow_gph = 0.0
                logger.debug(f"t={t:.1f}s: Engine FAILED")
            else:
                # Degraded: engine loses ~50% power (rough running, partial failure)
                # Source: POH — degraded engine produces reduced power/RPM
                state.engine_status = "degraded"
                state.engine_rpm *= 0.5
                state.fuel_flow_gph *= 0.6
                logger.debug(f"t={t:.1f}s: Engine DEGRADED (50% power)")

        elif component == "vacuum_pump":
            state.vacuum_pump_status = "failed"
            logger.debug(f"t={t:.1f}s: Vacuum pump FAILED")

        state.active_failures.append(component)
        return state

    # ─────────────────────────────────────────────────────────
    # CASCADE PROPAGATION
    # Applies Layer 1 cascade rules to update component states
    # Source: context2 [Stage 3: Cascade rules]
    # ─────────────────────────────────────────────────────────

    def _propagate_cascades(self, state: AircraftState, t: float) -> AircraftState:
        """
        Apply all applicable cascade rules from constraint graph.
        Source: context2 [Stage 3: cascade rules from POH]

        Rules applied in order of dependency chain:
          electrical → fuel → engine → aerodynamics
        """

        # R001: Alternator failure → bus drops to battery voltage (12V)
        if state.alternator_status == "failed" and state.voltage > 12.5:
            state.voltage = 12.6
            state.battery_status = "degraded"  # Now discharging

        # R002: Battery discharge over time
        # Source: Service manual D2006R4-13 p248 + FAA-H-8083-3C Chapter 17
        if state.alternator_status == "failed" and state.battery_charge_ah > 0:
            # Base load calculation with load shedding applied
            shed = getattr(state, "_load_shed_fraction", 0.0)
            essential_load = 15.0   # Fuel pump, ignition — cannot shed
            avionics_load  = 35.0   # Radios, lights — can shed
            base_load = essential_load + avionics_load * (1.0 - shed)

            # MANEUVER LOAD SPIKE — the critical cascade interaction
            # Source: context2 [Stage 4]:
            # "Pilot attempts recovery maneuver → increases electrical load
            #  (trim systems) → voltage drops to 20V → below fuel pump minimum"
            # This is the timing interaction that creates unrecoverable states.
            maneuver_spike = getattr(state, "_maneuver_load_spike_a", 0.0)
            effective_load = base_load + maneuver_spike

            # Voltage INSTANTANEOUS SAG under high load
            # Source: FAA AC 43.13-1B Section 11 — battery voltage under load
            # Under high current draw, voltage sags below steady-state value
            # Sag = load_fraction² × 2V (empirical for 12V aviation batteries)
            voltage_sag = effective_load * R_BATTERY_INTERNAL_OHM

            discharge_per_step = (effective_load / 3600.0) * DT
            state.battery_charge_ah = max(0.0, state.battery_charge_ah - discharge_per_step)

            # Voltage = steady-state from charge level - instantaneous sag from load
            charge_fraction = state.battery_charge_ah / BATTERY_CAPACITY_AH
            steady_state_v = 10.5 + 2.1 * charge_fraction  # 10.5V(dead) → 12.6V(full) battery curve
            state.voltage = max(0.0, steady_state_v - voltage_sag)

            # Log critical threshold crossing
            if state.voltage < 10.5 and state.fuel_pump_status not in ("failed",):
                logger.debug(
                    f"t={t:.1f}s: CRITICAL VOLTAGE {state.voltage:.2f}V "
                    f"(load={effective_load:.1f}A, "
                    f"spike={maneuver_spike:.1f}A, "
                    f"battery={state.battery_charge_ah:.1f}Ah)"
                )

        # R003: Low voltage degrades fuel pump (threshold: 11.5V)
        if state.voltage < 11.5 and state.fuel_pump_status == "normal":
            efficiency_factor = max(0.0, (state.voltage - 10.5) / (11.5 - 10.5))
            new_efficiency = 0.8 * efficiency_factor + 0.2
            if new_efficiency < state.fuel_pump_efficiency:
                state.fuel_pump_efficiency = new_efficiency
                state.fuel_pump_status = "degraded"

        # R004: Critical voltage — fuel pump fails (threshold: 10.5V from cache)
        if state.voltage < 10.5 and state.fuel_pump_status != "failed":
            state.fuel_pump_status = "failed"
            state.fuel_pump_efficiency = 0.0

        # Battery dead — all electrical fails
        if state.battery_charge_ah <= 0.0:
            state.voltage = 0.0
            state.fuel_pump_status = "failed"
            state.fuel_pump_efficiency = 0.0

        # R005: Fuel pump failure → fuel flow reduces
        if state.fuel_pump_status == "failed":
            state.fuel_flow_gph = 0.0

        elif state.fuel_pump_status == "degraded":
            nominal_flow = C182.fuel_flow_gph_max
            state.fuel_flow_gph = nominal_flow * state.fuel_pump_efficiency

        # R006: Engine stops when no fuel
        if state.fuel_flow_gph <= 0.0 and state.engine_status == "normal":
            state.engine_status = "degraded"

        if state.engine_status in ("failed", "degraded") and state.fuel_flow_gph <= 0:
            # Engine RPM decays to zero
            state.engine_rpm = max(0.0, state.engine_rpm - 50.0 * DT)
            if state.engine_rpm <= 0:
                state.engine_status = "failed"
                state.alternator_on = False  # Engine-driven alternator stops

        # R009: Degraded fuel pump → reduced RPM
        if state.fuel_pump_status == "degraded" and state.engine_status == "normal":
            target_rpm = rpm_from_fuel_flow(state.fuel_flow_gph)
            if target_rpm < state.engine_rpm:
                state.engine_rpm = max(target_rpm,
                                       state.engine_rpm - 100.0 * DT)

        # R007/R008: Engine failed → vacuum pump fails → gyros
        if state.engine_status == "failed":
            state.vacuum_pump_status = "failed"
            if state.alternator_status != "failed":
                state.alternator_status = "failed"
                state.alternator_on = False

        if state.vacuum_pump_status == "failed":
            state.attitude_indicator_status = "unreliable"

        # R010: Very low voltage → flaps/trim fail (tracked but not simulated in physics)
        # (Flap failure affects landing performance — not modelled in basic sim)

        # R011: Carburetor failure → engine degradation
        # Source: POH p36 — carburetor icing causes gradual RPM loss
        if state.engine_status == "normal":
            carb_failed = "carburetor" in getattr(state, "active_failures", [])
            if carb_failed:
                state.engine_rpm = max(0.0, state.engine_rpm - 30.0 * DT)
                state.engine_status = "degraded"
                if state.engine_rpm <= 0:
                    state.engine_status = "failed"

        # R012: Fuel selector failure → immediate fuel starvation
        # Source: Service manual p184 — fuel selector on empty tank = no flow
        if "fuel_selector" in getattr(state, "active_failures", []):
            state.fuel_flow_gph = 0.0
            state.fuel_pump_status = "failed"
            state.fuel_pump_efficiency = 0.0

        # R013: Voltage regulator failure → same cascade as alternator failure
        # Source: Service manual p257 — faulty regulator causes alternator failure
        if "voltage_regulator" in getattr(state, "active_failures", []):
            if state.alternator_status != "failed":
                state.alternator_status = "failed"
                state.alternator_on = False
                state.voltage = 12.0  # Drops to battery voltage

        # R014: Vacuum failure + IMC weather → spatial disorientation → spiral
        # Source: POH p141 — vacuum powers attitude indicator
        # In IMC with failed attitude reference, undetected roll develops
        weather = getattr(state, "_weather", "vmc")
        if (state.vacuum_pump_status == "failed" and
                state.attitude_indicator_status in ("unreliable", "failed") and
                weather in ("imc", "night_vmc")):
            # Spatial disorientation: progressive roll buildup
            # Rate depends on fatigue (tired pilot detects disorientation slower)
            if not state.pilot_aware_of_failure:
                disorientation_rate = 0.3 * (1.0 + state.pilot_fatigue)
                state.roll_rate_dps += disorientation_rate * DT

        return state

    # ─────────────────────────────────────────────────────────
    # PILOT MODEL
    # Source: FAA-H-8083-3C Chapter 17
    # ─────────────────────────────────────────────────────────

    def _update_pilot(
        self, state: AircraftState,
        t: float, failures: list
    ) -> AircraftState:
        """
        Update pilot awareness, load shedding, and maneuver response.

        SOURCE: FAA-H-8083-3C Chapter 17 (Human Factors in Aviation)

        Models three behaviors:
        1. AWARENESS DELAY — pilot notices failure after fatigue-dependent delay
        2. LOAD SHEDDING — aware pilot turns off non-essential avionics
        3. RECOVERY MANEUVER — pilot pitches up to compensate altitude loss
           CRITICAL: pitch-up during electrical emergency spikes electrical load
           This is the context2 [Stage 4] cascade interaction:
           "t=450s → Voltage drop DURING recovery maneuver → fuel pump fails"
        """
        if not state.active_failures:
            # No failure — zero maneuver load spike
            state._maneuver_load_spike_a = 0.0
            state._load_shed_fraction = 0.0
            return state

        # ── STEP 1: Awareness detection ─────────────────────────
        if not state.pilot_aware_of_failure:
            response_delay = pilot_response_delay(state.pilot_fatigue)
            first_failure_t = min(
                (f.get("trigger_time_s", 0.0) for f in failures
                 if f.get("component") in state.active_failures),
                default=0.0
            )
            if t >= first_failure_t + response_delay:
                state.pilot_aware_of_failure = True
                logger.debug(
                    f"t={t:.1f}s: Pilot AWARE of failure "
                    f"(delay={response_delay:.1f}s, fatigue={state.pilot_fatigue:.1f})"
                )

        if state.pilot_aware_of_failure:
            state.time_since_failure_s += DT

            # ── STEP 2: Load shedding ────────────────────────────
            # Source: FAA-H-8083-3C — electrical emergency checklist
            # Pilot turns off non-essential loads progressively
            # Takes up to 30s to complete (cognitive load during emergency)
            if state.alternator_status == "failed":
                shed_effectiveness = max(0.2, 1.0 - state.pilot_fatigue)
                # Time-based: pilot takes up to 30s to fully shed load
                shed_progress = min(1.0, state.time_since_failure_s / 30.0)
                state._load_shed_fraction = shed_effectiveness * shed_progress

            # ── STEP 3: Recovery maneuver — THE KEY INTERACTION ──
            # Source: context2 [Stage 4, t=450s sequence]:
            # "Pilot attempts recovery maneuver (pitch up)"
            # "Increases electrical load (trim systems)"
            # "Voltage drops to 20V → below fuel pump minimum"
            #
            # When pilot pitches up:
            #   - Electric trim motor activates: +8A
            #   - Flap actuator may activate: +12A
            #   - Autopilot disconnect alarm: +3A
            # Total maneuver spike: ~15-25A above baseline
            #
            # This is WORST when:
            #   a) Pilot is actively correcting (pitch rate > threshold)
            #   b) Voltage is already degraded (battery low)
            #   c) Pilot is unaware of the electrical cascade (didn't shed load)
            #
            # Note: load shedding REDUCES this spike (aware pilot avoids trim use)
            # Unaware / fatigued pilot uses trim aggressively → bigger spike

            pitch_rate = abs(state.pitch_rate_dps)
            is_maneuvering = pitch_rate > 1.5  # deg/s threshold

            if is_maneuvering and state.alternator_status == "failed":
                # Trim motor draws current proportional to pitch rate
                # Source: FAA-H-8083-3C — electric trim system loads
                trim_load_a = min(pitch_rate * 3.0, 20.0)  # Up to 20A for rapid pitch

                # Flap load if applying flaps during approach/recovery
                flap_load_a = 12.0 if abs(state.pitch_deg) < -5 else 0.0

                # Unaware pilot uses trim aggressively (hasn't shed loads)
                # Aware + fresh pilot avoids trim (uses manual if possible)
                shed = getattr(state, "_load_shed_fraction", 0.0)
                awareness_factor = 1.0 - (shed * 0.7)  # Aware pilot reduces spike

                state._maneuver_load_spike_a = (trim_load_a + flap_load_a) * awareness_factor

                if state._maneuver_load_spike_a > 10.0:
                    logger.debug(
                        f"t={t:.1f}s: MANEUVER LOAD SPIKE "
                        f"+{state._maneuver_load_spike_a:.1f}A "
                        f"(pitch_rate={pitch_rate:.1f}°/s, "
                        f"voltage={state.voltage:.2f}V)"
                    )
            else:
                state._maneuver_load_spike_a = 0.0

        else:
            # Unaware pilot — no load shedding, but may maneuver involuntarily
            state._load_shed_fraction = 0.0
            # Small unintentional control inputs
            state._maneuver_load_spike_a = abs(state.pitch_rate_dps) * 1.0

        return state

    # ─────────────────────────────────────────────────────────
    # PHYSICS COMPUTATION
    # Source: context2 [Stage 4: Compute aircraft state]
    # Source: FAA-H-8083-25C (lift/drag equations)
    # ─────────────────────────────────────────────────────────

    def _compute_physics(self, state: AircraftState, t: float) -> AircraftState:
        """
        Compute aircraft state for next time step.
        Source: context2 Stage 4 physics equations.
        """
        weight_lbs = C182.max_gross_lbs - (
            (C182.max_gross_lbs - C182.empty_weight_lbs) * 0.5
        )  # Mid-fuel weight approximation

        v_fps = knots_to_fps(state.airspeed_knots)

        # Thrust from engine
        thrust = engine_thrust(
            state.engine_rpm,
            state.altitude_ft,
            state.fuel_pump_efficiency,
        )

        # Aerodynamic forces
        L = lift_force(v_fps, state.altitude_ft, state.pitch_deg)
        D = drag_force(v_fps, state.altitude_ft, state.pitch_deg)

        # ── VERTICAL DYNAMICS ──────────────────────────────────
        vs_fps = vertical_speed(L, weight_lbs, thrust, D, state.pitch_deg)

        # Pilot pitch correction (if aware of failure)
        prev_pitch = state.pitch_deg
        if state.pilot_aware_of_failure and state.engine_status != "failed":
            # Pilot attempts to maintain altitude by pitching up
            # Source: context2 [Stage 4]: "pilot pitched up to recover"
            # This pitch-up is what triggers the maneuver load spike
            effectiveness = pilot_correction_effectiveness(
                state.pilot_fatigue, state.time_since_failure_s
            )
            target_vs = 0.0  # Pilot wants level flight
            pitch_correction = (target_vs - vs_fps) * 0.05 * effectiveness
            state.pitch_deg += pitch_correction * DT
            state.pitch_deg = max(-30.0, min(state.pitch_deg, 25.0))
        elif state.engine_status == "failed" and state.pilot_aware_of_failure:
            # Engine-out: pilot establishes best glide pitch (-3° for C182)
            # Source: POH — best glide speed 73 kts at pitch ~-3°
            # Gain 0.15: reaches target in ~7s (realistic pilot response)
            best_glide_pitch = -3.0
            pitch_correction = (best_glide_pitch - state.pitch_deg) * 0.15
            state.pitch_deg += pitch_correction * DT

        # Track pitch rate (degrees/second) — used by pilot model for load spike
        state.pitch_rate_dps = (state.pitch_deg - prev_pitch) / DT

        # When no failures active, pilot holds level flight (adjusts throttle)
        # Without this, thrust > drag causes perpetual climb to service ceiling
        # Only hold level flight when no flight-critical failures active
        flight_critical = {"engine_left","engine_both","engine","alternator","battery","fuel_pump","fuel_selector","carburetor","voltage_regulator"}
        has_critical = any(f in flight_critical for f in state.active_failures)
        if not has_critical and state.engine_status == "normal":
            vs_fps = 0.0  # Pilot maintains altitude in normal cruise

        # Update vertical speed and altitude
        state.vertical_speed_fps = vs_fps
        state.altitude_ft += vs_fps * DT
        state.altitude_ft = max(0.0, min(state.altitude_ft, 16500.0))

        # ── HORIZONTAL DYNAMICS ────────────────────────────────
        # Airspeed changes with thrust vs drag
        net_accel = (thrust - D) / (weight_lbs / 32.174)  # ft/s²
        v_fps += net_accel * DT

        # Stall check
        v_stall = stall_speed_fps(weight_lbs, state.altitude_ft)
        if v_fps < v_stall:
            # Stall — nose drops, airspeed builds, altitude lost rapidly
            # Cap at 0.5° per step regardless of DT to avoid numerical instability
            state.pitch_deg -= min(5.0 * DT, 0.5)
            state.pitch_rate_dps = -0.5 / DT
            v_fps = max(v_fps, v_stall * 0.85)

        # Clamp airspeed
        v_fps = max(20.0, min(v_fps, knots_to_fps(200.0)))
        state.airspeed_fps = v_fps
        state.airspeed_knots = fps_to_knots(v_fps)

        # ── ATTITUDE DYNAMICS ─────────────────────────────────
        # Roll dynamics — uncoordinated flight after engine failure
        if state.engine_status == "failed" and not state.pilot_aware_of_failure:
            # Asymmetric drag causes roll toward dead engine
            state.roll_rate_dps += 0.5 * DT  # Slow roll buildup
        elif state.engine_status == "failed" and state.pilot_aware_of_failure:
            # Pilot counters roll — effectiveness depends on fatigue
            effectiveness = pilot_correction_effectiveness(
                state.pilot_fatigue, state.time_since_failure_s
            )
            roll_correction = -state.roll_deg * 0.1 * effectiveness
            state.roll_rate_dps = roll_correction

        state.roll_deg += state.roll_rate_dps * DT
        state.roll_deg = max(-90.0, min(state.roll_deg, 90.0))

        # Spiral dive — if roll > 30° and no correction
        if abs(state.roll_deg) > 30 and not state.pilot_aware_of_failure:
            state.pitch_deg -= abs(state.roll_deg) * 0.02 * DT
            state.airspeed_knots += abs(state.roll_deg) * 0.1 * DT

        # ── FUEL DEPLETION ─────────────────────────────────────
        fuel_used = (state.fuel_flow_gph / 3600.0) * DT
        total_fuel = state.fuel_left_gal + state.fuel_right_gal
        if total_fuel > 0:
            # Deduct evenly from both tanks (simplification)
            state.fuel_left_gal  = max(0.0, state.fuel_left_gal  - fuel_used / 2)
            state.fuel_right_gal = max(0.0, state.fuel_right_gal - fuel_used / 2)
            if state.fuel_left_gal <= 0 and state.fuel_right_gal <= 0:
                state.fuel_flow_gph = 0.0
                state.fuel_pump_status = "failed"

        return state

    # ─────────────────────────────────────────────────────────
    # INITIAL STATE BUILDER
    # ─────────────────────────────────────────────────────────

    def _build_initial_state(self, scenario: dict) -> AircraftState:
        """Build initial AircraftState from scenario dict."""
        alt     = scenario.get("initial_altitude_ft", 5000.0)
        spd     = scenario.get("initial_airspeed_kts", 100.0)
        fuel    = scenario.get("initial_fuel_gal", 61.0)
        fat     = scenario.get("pilot_fatigue", 0.3)
        weather = scenario.get("weather", "vmc")

        state = AircraftState(
            altitude_ft      = alt,
            altitude_agl_ft  = alt,
            airspeed_knots   = spd,
            airspeed_fps     = knots_to_fps(spd),
            fuel_left_gal    = fuel / 2,
            fuel_right_gal   = fuel / 2,
            pilot_fatigue    = fat,
            voltage          = 14.0,   # From physics_cache: system_voltage
            battery_charge_ah= 33.0,   # From physics_cache: battery_capacity_ah
            engine_rpm       = rpm_from_fuel_flow(C182.fuel_flow_gph_max),
            fuel_flow_gph    = C182.fuel_flow_gph_max,
        )
        state._weather = weather  # Used by R014 IMC cascade
        return state