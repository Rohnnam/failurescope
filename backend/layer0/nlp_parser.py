"""
Layer 0: NLP Parser
Converts natural language scenario description → structured ScenarioInput JSON
using Ollama with mistral:7b model running locally.

Source: context1 [Layer 0: NLP Input Processing]
Source: context2 [Stage 1: LLM Parsing + Stage 2: Confirmation]

Hallucination Risk: MEDIUM — mitigated by:
  1. Constrained JSON schema in system prompt
  2. User confirmation screen before simulation runs
  3. Physical range validation (ScenarioValidator)
  4. Fallback to aircraft defaults on parse failure
"""

import json
import re
from typing import Optional
from loguru import logger

import ollama

from backend.layer0.defaults import (
    AIRCRAFT_DEFAULTS,
    AIRCRAFT_ALIASES,
    FATIGUE_MAPPINGS,
    FUEL_QUANTITY_MAPPINGS,
    ALTITUDE_PHRASE_MAPPINGS,
    SKILL_MAPPINGS,
    FAILURE_COMPONENT_MAPPINGS,
    SUPPORTED_AIRCRAFT,
)
from backend.layer0.schemas import (
    ScenarioInput,
    InitialState,
    PilotState,
    FailureEvent,
    ParsedScenarioResponse,
    PilotSkillLevel,
    FailureSeverity,
    FailureComponent,
)
from backend.layer0.validator import ScenarioValidator


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Critical: tells mistral to output ONLY valid JSON, no prose.
# Includes full schema + all supported values.
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a structured data extraction engine for an aircraft safety simulator.

Your ONLY job: Convert natural language aircraft scenario descriptions into a JSON object.

OUTPUT RULES:
- Output ONLY a single valid JSON object. No explanation, no markdown, no backticks.
- If a value is not mentioned, use null and the system will apply defaults.
- Never invent values not implied by the input.

OUTPUT SCHEMA:
{
  "aircraft_type": string or null,
  "altitude_ft": number or null,
  "airspeed_knots": number or null,
  "pitch_deg": number or null,
  "roll_deg": number or null,
  "fuel_left_gal": number or null,
  "fuel_right_gal": number or null,
  "fuel_description": string or null,
  "engine_rpm_left": number or null,
  "electrical_voltage": number or null,  (bus voltage 0-32V; do NOT put SOC% here)
  "battery_charge": number or null,  (SOC as 0.0-1.0, e.g. 40% SOC = 0.4)
  "pilot_skill": string or null,
  "pilot_fatigue": number or null,
  "fatigue_description": string or null,
  "flight_phase": string or null,
  "failures": [
    {
      "component": string,
      "trigger_time_seconds": number,
      "severity": string
    }
  ]
}

AIRCRAFT TYPE VALUES (use exactly):
cessna_182, cessna_172, piper_cherokee, beechcraft_bonanza, piper_seneca, cessna_310

PILOT SKILL VALUES (use exactly):
student, private, commercial, instrument, airline

PILOT SKILL MAPPING GUIDANCE:
- "IFR rated" → instrument
- "experienced/seasoned/veteran" → commercial
- "military/military trained" → commercial
- "ATP/airline transport" → airline
- "first solo/solo student" → student

FAILURE SEVERITY VALUES (use exactly):
catastrophic, degraded, intermittent

FAILURE COMPONENT VALUES (use exactly):
engine_left, engine_right, engine_both,
alternator, battery, voltage_regulator,
fuel_pump, fuel_selector, fuel_system, fuel_tank_left, fuel_tank_right,
carburetor, magnetos,
vacuum_pump, gyroscope, attitude_indicator,
flaps, trim_system, hydraulics, landing_gear, rudder, ailerons, elevator

COMPONENT MAPPING GUIDANCE:
- "voltage regulator fault/failure" → voltage_regulator
- "carburetor ice/icing" → carburetor (use severity: degraded)
- "fuel selector on empty tank" → fuel_selector (use severity: catastrophic)
- "partial power loss" → engine_left (use severity: degraded)
- "engine roughness" → carburetor or engine_left (use severity: degraded)
- "bus voltage unstable" → voltage_regulator

FATIGUE GUIDANCE (map descriptors to 0.0-1.0):
fresh/rested = 0.1, slightly tired = 0.4, moderate = 0.5,
tired/after 3hrs = 0.6, fatigued = 0.7, exhausted = 0.9

FUEL GUIDANCE (set fuel_description if relative, e.g. "full", "half", "low"):
full tanks = 1.0 capacity, half = 0.5, low = 0.15

FLIGHT PHASE GUIDANCE (affects altitude default):
taking_off, climbing, cruising, approach, pattern, low_altitude

EXAMPLE INPUT:
"Cessna 182 cruising at 5000 feet. Tired pilot, full fuel. Alternator fails."

EXAMPLE OUTPUT:
{"aircraft_type":"cessna_182","altitude_ft":5000,"airspeed_knots":null,"pitch_deg":null,"roll_deg":null,"fuel_left_gal":null,"fuel_right_gal":null,"fuel_description":"full","engine_rpm_left":null,"electrical_voltage":null,"battery_charge":null,"pilot_skill":"commercial","pilot_fatigue":0.6,"fatigue_description":"tired","flight_phase":"cruising","failures":[{"component":"alternator","trigger_time_seconds":0,"severity":"catastrophic"}]}
"""


# ─────────────────────────────────────────────────────────────
# NLP PARSER CLASS
# ─────────────────────────────────────────────────────────────

class NLPParser:
    """
    Converts natural language → ScenarioInput using Ollama mistral:7b.

    Pipeline:
      1. Call mistral:7b with constrained system prompt
      2. Extract JSON from response
      3. Apply aircraft defaults for null fields
      4. Build ScenarioInput object
      5. Validate via ScenarioValidator
      6. Generate human-readable summary
    """

    def __init__(self, model: str = "mistral:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.validator = ScenarioValidator()
        self.client = ollama.Client(host=host)
        logger.info(f"NLPParser initialized: model={model}, host={host}")

    # ─── PUBLIC ENTRY POINT ──────────────────────────────────────

    def parse(self, text: str) -> ParsedScenarioResponse:
        """
        Main parse method.

        Args:
            text: Natural language scenario description from user

        Returns:
            ParsedScenarioResponse with scenario + summary + assumptions
        """
        logger.info(f"Parsing input: '{text[:80]}...'")
        assumptions = []
        warnings = []

        # Step 1: Call LLM
        raw_output = self._call_llm(text)
        logger.debug(f"Raw LLM output: {raw_output}")

        # Step 2: Extract JSON from LLM output
        llm_data = self._extract_json(raw_output)
        logger.debug(f"Extracted JSON: {llm_data}")

        # Step 3: Resolve aircraft type
        aircraft_type, aircraft_assumptions = self._resolve_aircraft(llm_data)
        assumptions.extend(aircraft_assumptions)
        defaults = AIRCRAFT_DEFAULTS[aircraft_type]

        # Step 4: Build InitialState with defaults for null fields
        initial_state, state_assumptions = self._build_initial_state(
            llm_data, aircraft_type, defaults
        )
        assumptions.extend(state_assumptions)

        # Step 5: Build PilotState
        pilot_state, pilot_assumptions = self._build_pilot_state(llm_data)
        assumptions.extend(pilot_assumptions)

        # Step 6: Build FailureEvents
        failures, failure_warnings = self._build_failures(llm_data)
        warnings.extend(failure_warnings)

        if not failures:
            warnings.append(
                "No failure events detected in input. "
                "Simulation will run with no injected failures."
            )

        # Step 7: Extract weather from text context
        text_lower = text.lower()
        if any(w in text_lower for w in ["imc", " clouds", "in cloud", "instrument conditions", "ifr conditions"]):
            weather = "imc"
        elif any(w in text_lower for w in ["night", "dark outside", "after dark"]):
            weather = "night_vmc"
        elif any(w in text_lower for w in ["turbulenc", "rough air", "chop"]):
            weather = "vmc_turbulent"
        else:
            weather = "vmc"

        # Step 7b: Assemble ScenarioInput
        scenario = ScenarioInput(
            aircraft_type=aircraft_type,
            initial_state=initial_state,
            pilot_state=pilot_state,
            failures=failures,
            weather=weather,
        )

        # Step 8: Validate
        validation = self.validator.validate(scenario)
        if not validation.is_valid:
            logger.warning(f"Validation errors: {validation.errors}")
            warnings.extend([e.issue for e in validation.errors])
        warnings.extend(validation.warnings)

        # Step 9: Generate confirmation summary
        summary = self._generate_summary(scenario, assumptions)

        return ParsedScenarioResponse(
            scenario=scenario,
            summary=summary,
            assumptions=assumptions,
            parse_warnings=warnings,
            raw_llm_output=raw_output,
        )

    # ─── LLM CALL ────────────────────────────────────────────────

    def _call_llm(self, text: str) -> str:
        """Call Ollama mistral:7b and return raw string response."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                options={
                    "temperature": 0.0,   # Deterministic output for structured JSON
                    "top_p": 1.0,
                    "num_predict": 512,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.host}. "
                f"Ensure Ollama is running and mistral:7b is pulled. Error: {e}"
            )

    # ─── JSON EXTRACTION ─────────────────────────────────────────

    def _extract_json(self, raw: str) -> dict:
        """
        Extract FIRST complete JSON object from LLM response.
        Uses brace matching to avoid grabbing multiple JSON objects
        when LLM outputs multiple blocks (e.g. for compound inputs).
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = cleaned.rstrip("`").strip()

        start = cleaned.find("{")
        if start == -1:
            return {}

        # Match first complete JSON object via brace depth
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == "\"":
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            logger.error("No complete JSON object found in LLM response")
            return {}

        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}. Raw: {cleaned[start:start+200]}")
            return {}

    # ─── AIRCRAFT RESOLUTION ─────────────────────────────────────

    def _resolve_aircraft(self, llm_data: dict) -> tuple[str, list[str]]:
        """
        Resolve aircraft type from LLM output.
        Falls back to cessna_182 (most common training aircraft) if unknown.
        """
        assumptions = []
        raw_type = llm_data.get("aircraft_type")

        if raw_type:
            # Direct match
            if raw_type in AIRCRAFT_DEFAULTS:
                return raw_type, assumptions

            # Alias match
            normalized = raw_type.lower().strip()
            if normalized in AIRCRAFT_ALIASES:
                resolved = AIRCRAFT_ALIASES[normalized]
                assumptions.append(
                    f"Aircraft '{raw_type}' resolved to '{resolved}'"
                )
                return resolved, assumptions

        # Default fallback
        assumptions.append(
            "Aircraft type not specified or unrecognized. "
            "Defaulting to Cessna 182 Skylane."
        )
        return "cessna_182", assumptions

    # ─── INITIAL STATE BUILDER ────────────────────────────────────

    def _build_initial_state(
        self, llm_data: dict, aircraft_type: str, defaults: dict
    ) -> tuple[InitialState, list[str]]:
        """
        Build InitialState. LLM values take priority; defaults fill nulls.
        Source: context1 [Layer 0: Intelligent Defaults]
        """
        assumptions = []

        # Altitude
        altitude = llm_data.get("altitude_ft")
        if altitude is None:
            flight_phase = (llm_data.get("flight_phase") or "").lower()
            altitude = ALTITUDE_PHRASE_MAPPINGS.get(
                flight_phase, defaults["default_altitude_ft"]
            )
            assumptions.append(
                f"Altitude not specified. Using default {altitude} ft "
                f"({'from flight phase' if flight_phase else 'aircraft default'})."
            )

        # Airspeed
        airspeed = llm_data.get("airspeed_knots")
        if airspeed is None:
            airspeed = defaults["default_airspeed_knots"]
            assumptions.append(
                f"Airspeed not specified. Using default {airspeed} kts."
            )

        # Pitch / Roll
        pitch = llm_data.get("pitch_deg") or defaults["default_pitch_deg"]
        roll = llm_data.get("roll_deg") or defaults["default_roll_deg"]

        # Fuel
        fuel_desc = (llm_data.get("fuel_description") or "").lower().strip()
        fuel_left = llm_data.get("fuel_left_gal")
        fuel_right = llm_data.get("fuel_right_gal")

        # Sanity check: LLM sometimes gives numeric fuel in wrong units (e.g. 0.5 instead of 23)
        # If numeric value is implausibly low (<5 gal) AND description exists, prefer description
        min_plausible_gal = 5.0
        if fuel_left is not None and fuel_left < min_plausible_gal and fuel_desc:
            fuel_left = None  # Fall through to description-based lookup
        if fuel_right is not None and fuel_right < min_plausible_gal and fuel_desc:
            fuel_right = None

        if fuel_left is None or fuel_right is None:
            if fuel_desc in FUEL_QUANTITY_MAPPINGS:
                ratio = FUEL_QUANTITY_MAPPINGS[fuel_desc]
                fuel_left = defaults["fuel_capacity_left_gal"] * ratio
                fuel_right = defaults["fuel_capacity_right_gal"] * ratio
                assumptions.append(
                    f"Fuel described as '{fuel_desc}' → "
                    f"{fuel_left:.0f} gal left, {fuel_right:.0f} gal right."
                )
            else:
                fuel_left = fuel_left if fuel_left is not None else defaults["default_fuel_left_gal"]
                fuel_right = fuel_right if fuel_right is not None else defaults["default_fuel_right_gal"]
                assumptions.append(
                    f"Fuel not specified. Using default "
                    f"{fuel_left:.0f} gal left, {fuel_right:.0f} gal right."
                )

        # RPM — computed from altitude (standard atmosphere approximation)
        rpm = llm_data.get("engine_rpm_left")
        if rpm is None:
            altitude_val = float(altitude)
            altitude_factor = max(0.7, 1.0 - (altitude_val / 100000))
            rpm = defaults["default_rpm"] * altitude_factor
            rpm = round(rpm, 0)
            assumptions.append(
                f"Engine RPM not specified. Estimated {rpm:.0f} RPM "
                f"based on altitude {altitude_val:.0f} ft."
            )

        # Electrical — clamp voltage to valid range
        # LLM sometimes confuses battery SOC percentage (40%) with voltage (40V)
        voltage_raw = llm_data.get("electrical_voltage")
        if voltage_raw is not None and float(voltage_raw) > 32:
            # Likely a percentage (40% SOC) misread as voltage — use default
            voltage = defaults["default_voltage"]
            assumptions.append(
                f"Voltage {voltage_raw}V out of range (C182 is 14V system). Using default {voltage}V."
            )
        else:
            voltage = voltage_raw or defaults["default_voltage"]

        # Battery charge: LLM gives SOC as 0.0-1.0 fraction
        battery_raw = llm_data.get("battery_charge")
        if battery_raw is not None and float(battery_raw) <= 1.0:
            # SOC fraction → convert to Ah (33Ah battery)
            battery = float(battery_raw) * defaults.get("battery_capacity_ah", 33.0)
        else:
            battery = battery_raw or defaults["default_battery_charge"]

        initial_state = InitialState(
            altitude_ft=float(altitude),
            airspeed_knots=float(airspeed),
            pitch_deg=float(pitch),
            roll_deg=float(roll),
            fuel_left_gal=float(fuel_left),
            fuel_right_gal=float(fuel_right),
            engine_rpm_left=float(rpm),
            engine_rpm_right=float(rpm) if defaults["engine_count"] == 2 else 0.0,
            electrical_voltage=float(voltage),
            battery_charge=float(battery),
        )

        return initial_state, assumptions

    # ─── PILOT STATE BUILDER ──────────────────────────────────────

    def _build_pilot_state(self, llm_data: dict) -> tuple[PilotState, list[str]]:
        """Build PilotState from LLM output with defaults."""
        assumptions = []

        # Skill level
        skill_raw = (llm_data.get("pilot_skill") or "").lower().strip()
        skill = SKILL_MAPPINGS.get(skill_raw, "commercial")
        if not skill_raw:
            assumptions.append("Pilot skill not specified. Defaulting to Commercial.")

        # Fatigue
        fatigue = llm_data.get("pilot_fatigue")
        if fatigue is None:
            fatigue_desc = (llm_data.get("fatigue_description") or "").lower().strip()
            if fatigue_desc in FATIGUE_MAPPINGS:
                fatigue = FATIGUE_MAPPINGS[fatigue_desc]
                assumptions.append(
                    f"Fatigue described as '{fatigue_desc}' → {fatigue}."
                )
            else:
                fatigue = 0.3
                assumptions.append("Pilot fatigue not specified. Defaulting to 0.3 (moderate).")

        return PilotState(
            skill_level=PilotSkillLevel(skill),
            fatigue=float(fatigue),
        ), assumptions

    # ─── FAILURE BUILDER ─────────────────────────────────────────

    def _build_failures(self, llm_data: dict) -> tuple[list[FailureEvent], list[str]]:
        """Build list of FailureEvents from LLM output."""
        warnings = []
        failures = []

        raw_failures = llm_data.get("failures") or []
        if not isinstance(raw_failures, list):
            warnings.append("LLM returned non-list failures field. Treating as empty.")
            return failures, warnings

        valid_components = {e.value for e in FailureComponent}
        valid_severities = {e.value for e in FailureSeverity}

        for i, raw in enumerate(raw_failures):
            if not isinstance(raw, dict):
                warnings.append(f"Failure[{i}] is not a dict, skipping.")
                continue

            component = (raw.get("component") or "").lower().strip()
            severity_raw = raw.get("severity")
            severity = (severity_raw or "catastrophic").lower().strip() if severity_raw is not None else "catastrophic"
            trigger = raw.get("trigger_time_seconds", 0)

            if component not in valid_components:
                warnings.append(
                    f"Failure[{i}] component '{component}' not recognized. Skipping."
                )
                continue

            if severity not in valid_severities:
                severity = "catastrophic"
                warnings.append(
                    f"Failure[{i}] severity not recognized. Defaulting to catastrophic."
                )

            try:
                trigger = float(trigger)
            except (TypeError, ValueError):
                trigger = 0.0
                warnings.append(f"Failure[{i}] trigger_time invalid. Defaulting to t=0s.")

            failures.append(FailureEvent(
                component=FailureComponent(component),
                trigger_time_seconds=trigger,
                severity=FailureSeverity(severity),
            ))

        return failures, warnings

    # ─── SUMMARY GENERATOR ───────────────────────────────────────

    def _generate_summary(
        self, scenario: ScenarioInput, assumptions: list[str]
    ) -> str:
        """
        Generate human-readable confirmation summary.
        Source: context1 [Layer 0: Step 2 Confirmation]
        Source: design [Screen 2: Confirmation]
        """
        s = scenario.initial_state
        p = scenario.pilot_state
        defaults = AIRCRAFT_DEFAULTS[scenario.aircraft_type]

        lines = [
            f"Aircraft:  {defaults['display_name']}",
            f"Altitude:  {s.altitude_ft:,.0f} ft",
            f"Airspeed:  {s.airspeed_knots:.0f} knots",
            f"Fuel:      {s.fuel_left_gal:.0f} gal left  |  {s.fuel_right_gal:.0f} gal right",
            f"Pilot:     {p.skill_level.value.capitalize()} pilot",
            f"Fatigue:   {p.fatigue:.1f} "
            f"({'fresh' if p.fatigue < 0.3 else 'moderate' if p.fatigue < 0.6 else 'tired' if p.fatigue < 0.8 else 'exhausted'})",
        ]

        if scenario.failures:
            lines.append("Failures:")
            for f in scenario.failures:
                t = f"at t={f.trigger_time_seconds:.0f}s"
                lines.append(
                    f"  ☑ {f.component.value.replace('_', ' ').title()} "
                    f"({f.severity.value}) {t}"
                )
        else:
            lines.append("Failures:  None specified")

        return "\n".join(lines)