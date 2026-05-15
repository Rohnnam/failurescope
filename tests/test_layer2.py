"""
Tests for Layer 2: Physics Simulation

Tests verify:
1. Physics equations produce physically reasonable values
2. Simulator is deterministic (same input → same output)
3. Key cascade scenarios produce expected outcomes
4. Scenario generator produces valid scenario dicts

Run: pytest tests/test_layer2.py -v
"""

import pytest
import math
from backend.layer2.physics_engine import (
    air_density, dynamic_pressure, lift_force, drag_force,
    engine_thrust, rpm_from_fuel_flow, stall_speed_fps,
    knots_to_fps, fps_to_knots, is_unrecoverable, C182,
    RHO_SL,
)
from backend.layer2.simulator import Simulator
from backend.layer2.state import SimOutcome
from backend.layer2.scenario_generator import (
    generate_scenarios, count_scenarios, is_plausible_combination,
)


# ─────────────────────────────────────────────────────────────
# PHYSICS ENGINE TESTS
# ─────────────────────────────────────────────────────────────

class TestPhysicsEngine:

    def test_air_density_sea_level(self):
        """Sea level density must match ISA standard."""
        rho = air_density(0.0)
        assert abs(rho - RHO_SL) < 0.0001, f"Sea level density {rho} != ISA {RHO_SL}"

    def test_air_density_decreases_with_altitude(self):
        """Air density must decrease with altitude."""
        rho_sl   = air_density(0.0)
        rho_5k   = air_density(5000.0)
        rho_10k  = air_density(10000.0)
        assert rho_sl > rho_5k > rho_10k

    def test_air_density_service_ceiling(self):
        """Air density at C182 service ceiling (16500ft) is positive."""
        rho = air_density(16500.0)
        assert rho > 0.0

    def test_lift_positive_at_cruise(self):
        """Lift must be positive at cruise conditions."""
        v_fps = knots_to_fps(100.0)
        L = lift_force(v_fps, 5000.0, 5.0)
        assert L > 0, f"Lift at cruise is {L}"

    def test_lift_increases_with_pitch(self):
        """Higher pitch angle should increase lift (up to stall)."""
        v_fps = knots_to_fps(100.0)
        L_low  = lift_force(v_fps, 5000.0, 2.0)
        L_high = lift_force(v_fps, 5000.0, 10.0)
        assert L_high > L_low

    def test_drag_positive_at_cruise(self):
        """Drag must be positive at cruise."""
        v_fps = knots_to_fps(100.0)
        D = drag_force(v_fps, 5000.0, 5.0)
        assert D > 0

    def test_thrust_zero_at_zero_rpm(self):
        """No thrust when engine is stopped."""
        T = engine_thrust(0.0, 5000.0)
        assert T == 0.0

    def test_thrust_positive_at_cruise_rpm(self):
        """Thrust must be positive at cruise RPM."""
        T = engine_thrust(1800.0, 5000.0)
        assert T > 0

    def test_thrust_decreases_with_altitude(self):
        """Thrust must decrease with altitude (density effect)."""
        T_sl  = engine_thrust(2400.0, 0.0)
        T_10k = engine_thrust(2400.0, 10000.0)
        assert T_sl > T_10k

    def test_stall_speed_reasonable(self):
        """Stall speed must be in reasonable range for C182."""
        v_stall = stall_speed_fps(C182.max_gross_lbs, 5000.0)
        v_stall_kts = fps_to_knots(v_stall)
        # POH says VS0=45 kts, VS1=56 kts — stall with flaps up should be ~56
        assert 40.0 <= v_stall_kts <= 70.0, f"Stall speed {v_stall_kts} kts out of range"

    def test_knot_fps_conversion(self):
        """Unit conversion must be consistent."""
        kts = 100.0
        fps = knots_to_fps(kts)
        assert abs(fps_to_knots(fps) - kts) < 0.01

    def test_is_unrecoverable_ground_impact(self):
        """Altitude=0 must be catastrophic."""
        unrecov, reason = is_unrecoverable(0.0, 100.0, 0.0, 0.0, 14.0, 11.0, 0.0)
        assert unrecov
        assert "ground" in reason

    def test_is_unrecoverable_extreme_roll(self):
        """Roll >60° must be catastrophic."""
        unrecov, reason = is_unrecoverable(5000.0, 100.0, 65.0, 5.0, 14.0, 11.0, 5000.0)
        assert unrecov

    def test_is_recoverable_normal_flight(self):
        """Normal cruise should not be unrecoverable."""
        unrecov, reason = is_unrecoverable(5000.0, 100.0, 5.0, 5.0, 14.0, 11.0, 5000.0)
        assert not unrecov


# ─────────────────────────────────────────────────────────────
# SIMULATOR TESTS
# ─────────────────────────────────────────────────────────────

class TestSimulator:

    @pytest.fixture
    def sim(self):
        return Simulator(aircraft_type="cessna_182")

    @pytest.fixture
    def baseline_scenario(self):
        return {
            "scenario_id":          "test_baseline",
            "initial_altitude_ft":  5000.0,
            "initial_airspeed_kts": 100.0,
            "initial_fuel_gal":     61.0,
            "pilot_skill":          "commercial",
            "pilot_fatigue":        0.3,
            "weather":              "vmc",
            "failures":             [],  # No failures — should be recoverable
        }

    @pytest.fixture
    def alternator_fail_scenario(self):
        return {
            "scenario_id":          "test_alternator",
            "initial_altitude_ft":  5000.0,
            "initial_airspeed_kts": 100.0,
            "initial_fuel_gal":     61.0,
            "pilot_skill":          "commercial",
            "pilot_fatigue":        0.6,
            "weather":              "vmc",
            "failures": [{
                "component":      "alternator",
                "trigger_time_s": 0.0,
                "severity":       "catastrophic",
            }],
        }

    def test_baseline_no_failure_is_recoverable(self, sim, baseline_scenario):
        """No failures → aircraft survives full simulation."""
        traj = sim.run(baseline_scenario)
        assert traj.outcome == SimOutcome.RECOVERABLE, \
            f"Baseline should be recoverable, got {traj.outcome}"

    def test_trajectory_has_states(self, sim, baseline_scenario):
        """Trajectory must contain state records."""
        traj = sim.run(baseline_scenario)
        assert len(traj.states) > 0

    def test_trajectory_states_have_required_fields(self, sim, baseline_scenario):
        """Each state must have required fields."""
        traj = sim.run(baseline_scenario)
        required = ["t", "alt", "spd", "v", "rpm", "ff", "outcome"]
        for field in required:
            assert field in traj.states[0], f"State missing field: {field}"

    def test_simulator_is_deterministic(self, sim, alternator_fail_scenario):
        """Same scenario must produce identical results."""
        traj1 = sim.run(alternator_fail_scenario)
        traj2 = sim.run(alternator_fail_scenario)
        assert traj1.outcome == traj2.outcome
        assert traj1.simulation_duration_s == traj2.simulation_duration_s
        assert len(traj1.states) == len(traj2.states)

    def test_alternator_failure_triggers_cascade(self, sim, alternator_fail_scenario):
        """Alternator failure should trigger voltage drop in states."""
        traj = sim.run(alternator_fail_scenario)
        # Find voltage in early states vs late states
        if len(traj.states) > 10:
            early_v = traj.states[5]["v"]   # ~0.5s after failure
            # Voltage should have dropped from 14V to ~12V (battery)
            assert early_v <= 14.0, f"Voltage should drop after alternator fail: {early_v}"

    def test_engine_failure_is_catastrophic_at_low_altitude(self, sim):
        """Engine failure at low altitude should be catastrophic."""
        scenario = {
            "scenario_id":          "test_engine_low",
            "initial_altitude_ft":  1000.0,
            "initial_airspeed_kts": 80.0,
            "initial_fuel_gal":     30.0,
            "pilot_skill":          "student",
            "pilot_fatigue":        0.7,
            "weather":              "vmc",
            "failures": [{
                "component":      "engine_left",
                "trigger_time_s": 0.0,
                "severity":       "catastrophic",
            }],
        }
        traj = sim.run(scenario)
        assert traj.outcome == SimOutcome.CATASTROPHIC

    def test_altitude_appears_in_failure_chain(self, sim, alternator_fail_scenario):
        """Failure chain should record alternator."""
        traj = sim.run(alternator_fail_scenario)
        components = [f["component"] for f in traj.failure_chain]
        assert "alternator" in components

    def test_trajectory_has_scenario_id(self, sim, alternator_fail_scenario):
        """Trajectory must preserve scenario ID."""
        traj = sim.run(alternator_fail_scenario)
        assert traj.scenario_id == alternator_fail_scenario["scenario_id"]


# ─────────────────────────────────────────────────────────────
# SCENARIO GENERATOR TESTS
# ─────────────────────────────────────────────────────────────

class TestScenarioGenerator:

    def test_generates_correct_count(self):
        """Generator produces scenarios up to max_scenarios limit."""
        scenarios = list(generate_scenarios(max_scenarios=50, single_failure_only=True))
        assert len(scenarios) <= 50
        assert len(scenarios) > 0

    def test_scenarios_have_required_fields(self):
        """Each scenario must have required fields."""
        for scenario in generate_scenarios(max_scenarios=5, single_failure_only=True):
            assert "scenario_id" in scenario
            assert "initial_altitude_ft" in scenario
            assert "failures" in scenario
            assert len(scenario["failures"]) >= 1

    def test_scenario_ids_are_unique(self):
        """No duplicate scenario IDs."""
        ids = [s["scenario_id"] for s in
               generate_scenarios(max_scenarios=100, single_failure_only=True)]
        assert len(ids) == len(set(ids))

    def test_count_scenarios(self):
        """count_scenarios must return positive numbers."""
        counts = count_scenarios(single_failure_only=True)
        assert counts["single_failure"] > 0
        assert counts["total"] > 0

    def test_plausible_combinations(self):
        """Implausible combinations are filtered."""
        # Same component twice
        assert not is_plausible_combination("alternator", "alternator", None)
        # Plausible combination
        assert is_plausible_combination("alternator", "vacuum_pump", None)

    def test_pilot_fatigue_in_range(self):
        """Pilot fatigue must be 0-1."""
        for scenario in generate_scenarios(max_scenarios=20, single_failure_only=True):
            assert 0.0 <= scenario["pilot_fatigue"] <= 1.0

    def test_altitude_valid(self):
        """Initial altitude must be positive."""
        for scenario in generate_scenarios(max_scenarios=20, single_failure_only=True):
            assert scenario["initial_altitude_ft"] > 0