"""
Layer 2: Aircraft State and Trajectory Schemas
Data structures for simulation state snapshots and trajectory output.

Source: context2 [Stage 4: STORE STATE - trajectory format]
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SimOutcome(str, Enum):
    RECOVERABLE   = "recoverable"
    CATASTROPHIC  = "catastrophic"
    IN_PROGRESS   = "in_progress"
    TIME_EXCEEDED = "time_exceeded"  # Simulation ran full duration without crash


@dataclass
class AircraftState:
    """
    Complete aircraft state at one time step.
    Source: context2 [Stage 4: trajectory state format]
    """
    # Time
    time_s:           float = 0.0

    # Position
    altitude_ft:      float = 5000.0
    altitude_agl_ft:  float = 5000.0   # Above ground level

    # Airspeed
    airspeed_knots:   float = 100.0
    airspeed_fps:     float = 168.78   # feet per second

    # Attitude
    pitch_deg:        float = 5.0
    roll_deg:         float = 0.0
    yaw_deg:          float = 0.0

    # Rates
    pitch_rate_dps:   float = 0.0      # deg/sec
    roll_rate_dps:    float = 0.0
    vertical_speed_fps: float = 0.0

    # Engine
    engine_rpm:       float = 1800.0
    fuel_flow_gph:    float = 11.0
    fuel_left_gal:    float = 30.5
    fuel_right_gal:   float = 30.5
    throttle:         float = 0.75     # 0.0-1.0

    # Electrical
    voltage:          float = 14.0
    battery_charge_ah: float = 33.0   # Remaining Ah (from cache: 33Ah)
    alternator_on:    bool = True

    # Component statuses (mirror constraint graph nodes)
    alternator_status:  str = "normal"
    battery_status:     str = "normal"
    fuel_pump_efficiency: float = 1.0  # 0.0-1.0
    fuel_pump_status:   str = "normal"
    engine_status:      str = "normal"
    vacuum_pump_status: str = "normal"
    attitude_indicator_status: str = "normal"

    # Pilot
    pilot_fatigue:    float = 0.3
    pilot_aware_of_failure: bool = False
    time_since_failure_s: float = 0.0

    # Outcome tracking
    outcome:          SimOutcome = SimOutcome.IN_PROGRESS
    unrecoverable_reason: str = ""
    active_failures:  list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "t":          round(self.time_s, 1),
            "alt":        round(self.altitude_ft, 1),
            "alt_agl":    round(self.altitude_agl_ft, 1),
            "spd":        round(self.airspeed_knots, 1),
            "pitch":      round(self.pitch_deg, 2),
            "roll":       round(self.roll_deg, 2),
            "yaw":        round(self.yaw_deg, 2),
            "vs":         round(self.vertical_speed_fps, 2),
            "rpm":        round(self.engine_rpm, 0),
            "ff":         round(self.fuel_flow_gph, 2),
            "fuel_l":     round(self.fuel_left_gal, 2),
            "fuel_r":     round(self.fuel_right_gal, 2),
            "v":          round(self.voltage, 2),
            "bat":        round(self.battery_charge_ah, 2),
            "pump_eff":   round(self.fuel_pump_efficiency, 3),
            "outcome":    self.outcome.value,
            "failures":   list(self.active_failures),
        }


@dataclass
class TrajectoryRecord:
    """
    Complete simulation trajectory — input scenario + state history + outcome.
    Source: context1 [Layer 2: Data Generated]
    This is what Layer 3 Transformer trains on.
    """
    # Scenario identification
    scenario_id:        str = ""
    aircraft_type:      str = "cessna_182"

    # Initial conditions
    initial_altitude_ft:   float = 5000.0
    initial_airspeed_kts:  float = 100.0
    initial_fuel_gal:      float = 61.0
    pilot_skill:           str = "commercial"
    pilot_fatigue:         float = 0.3
    weather:               str = "vmc"

    # Failure sequence
    failures:           list = field(default_factory=list)

    # Trajectory (list of state dicts — compressed for storage)
    states:             list = field(default_factory=list)

    # Outcome
    outcome:            SimOutcome = SimOutcome.IN_PROGRESS
    time_to_criticality_s: Optional[float] = None
    recovery_margin_s:  Optional[float] = None
    failure_chain:      list = field(default_factory=list)
    unrecoverable_reason: str = ""

    # Metadata
    simulation_duration_s: float = 0.0
    total_time_steps:   int = 0
    dt:                 float = 0.1

    def to_dict(self) -> dict:
        return {
            "scenario_id":    self.scenario_id,
            "aircraft":       self.aircraft_type,
            "init": {
                "altitude_ft":  self.initial_altitude_ft,
                "airspeed_kts": self.initial_airspeed_kts,
                "fuel_gal":     self.initial_fuel_gal,
                "skill":        self.pilot_skill,
                "fatigue":      self.pilot_fatigue,
                "weather":      self.weather,
            },
            "failures":       self.failures,
            "outcome":        self.outcome.value,
            "ttc_s":          self.time_to_criticality_s,
            "recovery_s":     self.recovery_margin_s,
            "failure_chain":  self.failure_chain,
            "reason":         self.unrecoverable_reason,
            "duration_s":     self.simulation_duration_s,
            "steps":          self.total_time_steps,
            "states":         self.states,
        }