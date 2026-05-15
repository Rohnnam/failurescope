"""
Layer 0: Pydantic Schemas for Scenario JSON
Defines the data contract between Layer 0 (NLP) and all downstream layers.
Source: context2 [Stage 1: LLM Parsing - Output JSON structure]
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class PilotSkillLevel(str, Enum):
    STUDENT = "student"
    PRIVATE = "private"
    COMMERCIAL = "commercial"
    INSTRUMENT = "instrument"
    AIRLINE = "airline"


class FailureSeverity(str, Enum):
    CATASTROPHIC = "catastrophic"
    DEGRADED = "degraded"
    INTERMITTENT = "intermittent"


class FailureComponent(str, Enum):
    # Engine
    ENGINE_LEFT       = "engine_left"
    ENGINE_RIGHT      = "engine_right"
    ENGINE_BOTH       = "engine_both"
    # Electrical — all have Layer 1 cascade rules
    ALTERNATOR        = "alternator"
    BATTERY           = "battery"
    VOLTAGE_REGULATOR = "voltage_regulator"   # R013: same cascade as alternator
    # Fuel system — all have Layer 1 cascade rules
    FUEL_PUMP         = "fuel_pump"
    FUEL_SELECTOR     = "fuel_selector"       # R012: starvation
    FUEL_SYSTEM       = "fuel_system"
    FUEL_TANK_LEFT    = "fuel_tank_left"
    FUEL_TANK_RIGHT   = "fuel_tank_right"
    # Engine subsystems
    CARBURETOR        = "carburetor"          # R011: gradual RPM loss
    MAGNETOS          = "magnetos"
    # Instruments
    VACUUM_PUMP       = "vacuum_pump"         # R014: IMC spatial disorientation
    GYROSCOPE         = "gyroscope"
    ATTITUDE_INDICATOR = "attitude_indicator"
    # Flight controls
    FLAPS             = "flaps"
    TRIM_SYSTEM       = "trim_system"
    HYDRAULICS        = "hydraulics"
    LANDING_GEAR      = "landing_gear"
    RUDDER            = "rudder"
    AILERONS          = "ailerons"
    ELEVATOR          = "elevator"


# ─────────────────────────────────────────────────────────────
# SUB-SCHEMAS
# ─────────────────────────────────────────────────────────────

class InitialState(BaseModel):
    """Aircraft initial state at simulation start."""

    altitude_ft: float = Field(..., ge=0, le=45000, description="Altitude in feet MSL")
    airspeed_knots: float = Field(..., ge=0, le=350, description="Indicated airspeed in knots")
    pitch_deg: float = Field(default=5.0, ge=-90, le=90, description="Pitch angle in degrees")
    roll_deg: float = Field(default=0.0, ge=-180, le=180, description="Roll angle in degrees")
    fuel_left_gal: float = Field(..., ge=0, le=200, description="Left tank fuel in gallons")
    fuel_right_gal: float = Field(..., ge=0, le=200, description="Right tank fuel in gallons")
    engine_rpm_left: float = Field(..., ge=0, le=3500, description="Left engine RPM")
    engine_rpm_right: float = Field(default=0.0, ge=0, le=3500, description="Right engine RPM (0 for single engine)")
    electrical_voltage: float = Field(default=28.0, ge=0, le=32, description="Electrical bus voltage")
    battery_charge: float = Field(default=100.0, ge=0, le=100, description="Battery charge percentage")


class PilotState(BaseModel):
    """Pilot condition at simulation start."""

    skill_level: PilotSkillLevel = Field(default=PilotSkillLevel.COMMERCIAL)
    fatigue: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fatigue level: 0.0=fully rested, 1.0=severely exhausted"
    )


class FailureEvent(BaseModel):
    """A single failure event injected into the simulation."""

    component: FailureComponent = Field(..., description="Aircraft component that fails")
    trigger_time_seconds: float = Field(
        default=0.0,
        ge=0,
        le=7200,
        description="Time in seconds from simulation start when failure occurs"
    )
    severity: FailureSeverity = Field(
        default=FailureSeverity.CATASTROPHIC,
        description="Failure severity level"
    )


# ─────────────────────────────────────────────────────────────
# PRIMARY SCENARIO SCHEMA
# ─────────────────────────────────────────────────────────────

class ScenarioInput(BaseModel):
    """
    Complete scenario definition passed to the simulator.
    This is the output of Layer 0 and the input to Layer 1.
    Source: context2 [Stage 1 Output JSON]
    """

    aircraft_type: str = Field(
        ...,
        description="Aircraft type identifier, e.g. 'cessna_182'"
    )
    initial_state: InitialState
    pilot_state: PilotState = Field(default_factory=PilotState)
    failures: List[FailureEvent] = Field(
        default_factory=list,
        description="Ordered list of failure events to inject"
    )
    weather: str = Field(default="vmc", description="Weather conditions: vmc, imc, vmc_turbulent, night_vmc")

    @field_validator("aircraft_type")
    @classmethod
    def aircraft_must_be_supported(cls, v: str) -> str:
        from backend.layer0.defaults import SUPPORTED_AIRCRAFT
        if v not in SUPPORTED_AIRCRAFT:
            raise ValueError(
                f"Aircraft '{v}' not supported. "
                f"Supported: {SUPPORTED_AIRCRAFT}"
            )
        return v


# ─────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    """POST /api/parse_scenario request body."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language scenario description"
    )


class ParsedScenarioResponse(BaseModel):
    """
    POST /api/parse_scenario response.
    Returns parsed scenario + human-readable summary for confirmation screen.
    Source: context1 [Layer 0: Step 2 Confirmation]
    """

    scenario: ScenarioInput
    summary: str = Field(..., description="Human-readable summary for confirmation screen")
    assumptions: List[str] = Field(
        default_factory=list,
        description="List of assumptions made during parsing"
    )
    parse_warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings about ambiguous input"
    )
    raw_llm_output: Optional[str] = Field(
        default=None,
        description="Raw LLM output for debugging (omitted in production)"
    )


class ValidationError(BaseModel):
    """Single validation error detail."""
    field: str
    issue: str
    value: object


class ValidationResult(BaseModel):
    """Result of scenario validation."""
    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)