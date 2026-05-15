"""
Layer 0: Aircraft Defaults, Phrase Mappings, Failure Component Mappings
Source: context1 [Layer 0: Intelligent Defaults]

All values sourced from:
  - Cessna 182 POH
  - Cessna 172 POH
  - Piper PA-28 POH
  - Beechcraft Bonanza V35 POH
  - Piper Seneca PA-34 POH
  - Cessna 310 POH
"""

# ─────────────────────────────────────────────────────────────
# AIRCRAFT DEFAULT CONFIGURATIONS
# ─────────────────────────────────────────────────────────────

AIRCRAFT_DEFAULTS = {
    "cessna_182": {
        "display_name": "Cessna 182 Skylane",
        "engine_count": 1,
        "default_altitude_ft": 5000,
        "default_airspeed_knots": 100,
        "default_pitch_deg": 5.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 46.0,
        "fuel_capacity_right_gal": 46.0,
        "default_fuel_left_gal": 40.0,
        "default_fuel_right_gal": 40.0,
        "default_rpm": 1800,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 18000,
        "vne_knots": 175,
        "vs0_knots": 44,
        "cruise_fuel_flow_gph": 8.5,
        "alternator_output_amps": 60,
        "battery_capacity_ah": 24,
        "min_fuel_pump_voltage": 20.0,
    },
    "cessna_172": {
        "display_name": "Cessna 172 Skyhawk",
        "engine_count": 1,
        "default_altitude_ft": 4000,
        "default_airspeed_knots": 95,
        "default_pitch_deg": 4.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 26.0,
        "fuel_capacity_right_gal": 26.0,
        "default_fuel_left_gal": 20.0,
        "default_fuel_right_gal": 20.0,
        "default_rpm": 1800,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 14000,
        "vne_knots": 163,
        "vs0_knots": 40,
        "cruise_fuel_flow_gph": 7.5,
        "alternator_output_amps": 60,
        "battery_capacity_ah": 24,
        "min_fuel_pump_voltage": 20.0,
    },
    "piper_cherokee": {
        "display_name": "Piper Cherokee PA-28",
        "engine_count": 1,
        "default_altitude_ft": 4500,
        "default_airspeed_knots": 105,
        "default_pitch_deg": 4.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 25.0,
        "fuel_capacity_right_gal": 25.0,
        "default_fuel_left_gal": 22.0,
        "default_fuel_right_gal": 22.0,
        "default_rpm": 2000,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 13000,
        "vne_knots": 160,
        "vs0_knots": 44,
        "cruise_fuel_flow_gph": 8.0,
        "alternator_output_amps": 60,
        "battery_capacity_ah": 24,
        "min_fuel_pump_voltage": 20.0,
    },
    "beechcraft_bonanza": {
        "display_name": "Beechcraft Bonanza V35",
        "engine_count": 1,
        "default_altitude_ft": 7000,
        "default_airspeed_knots": 155,
        "default_pitch_deg": 3.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 40.0,
        "fuel_capacity_right_gal": 40.0,
        "default_fuel_left_gal": 35.0,
        "default_fuel_right_gal": 35.0,
        "default_rpm": 2300,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 18500,
        "vne_knots": 225,
        "vs0_knots": 55,
        "cruise_fuel_flow_gph": 12.0,
        "alternator_output_amps": 70,
        "battery_capacity_ah": 35,
        "min_fuel_pump_voltage": 20.0,
    },
    "piper_seneca": {
        "display_name": "Piper Seneca PA-34",
        "engine_count": 2,
        "default_altitude_ft": 6000,
        "default_airspeed_knots": 160,
        "default_pitch_deg": 3.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 48.0,
        "fuel_capacity_right_gal": 48.0,
        "default_fuel_left_gal": 40.0,
        "default_fuel_right_gal": 40.0,
        "default_rpm": 2300,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 25000,
        "vne_knots": 195,
        "vs0_knots": 62,
        "cruise_fuel_flow_gph": 18.0,
        "alternator_output_amps": 70,
        "battery_capacity_ah": 35,
        "min_fuel_pump_voltage": 20.0,
    },
    "cessna_310": {
        "display_name": "Cessna 310",
        "engine_count": 2,
        "default_altitude_ft": 8000,
        "default_airspeed_knots": 185,
        "default_pitch_deg": 3.0,
        "default_roll_deg": 0.0,
        "fuel_capacity_left_gal": 50.0,
        "fuel_capacity_right_gal": 50.0,
        "default_fuel_left_gal": 45.0,
        "default_fuel_right_gal": 45.0,
        "default_rpm": 2300,
        "default_voltage": 14.0,
        "default_battery_charge": 100,
        "max_altitude_ft": 20700,
        "vne_knots": 223,
        "vs0_knots": 68,
        "cruise_fuel_flow_gph": 22.0,
        "alternator_output_amps": 70,
        "battery_capacity_ah": 35,
        "min_fuel_pump_voltage": 20.0,
    },
}

# ─────────────────────────────────────────────────────────────
# AIRCRAFT NATURAL LANGUAGE ALIASES
# Maps user input phrases → internal aircraft_type keys
# ─────────────────────────────────────────────────────────────

AIRCRAFT_ALIASES = {
    # Cessna 182
    "cessna 182": "cessna_182",
    "cessna182": "cessna_182",
    "c182": "cessna_182",
    "skylane": "cessna_182",
    "182": "cessna_182",

    # Cessna 172
    "cessna 172": "cessna_172",
    "cessna172": "cessna_172",
    "c172": "cessna_172",
    "skyhawk": "cessna_172",
    "172": "cessna_172",

    # Piper Cherokee
    "piper cherokee": "piper_cherokee",
    "cherokee": "piper_cherokee",
    "pa-28": "piper_cherokee",
    "pa28": "piper_cherokee",

    # Beechcraft Bonanza
    "beechcraft bonanza": "beechcraft_bonanza",
    "bonanza": "beechcraft_bonanza",
    "v35": "beechcraft_bonanza",
    "beechcraft": "beechcraft_bonanza",

    # Piper Seneca
    "piper seneca": "piper_seneca",
    "seneca": "piper_seneca",
    "pa-34": "piper_seneca",
    "pa34": "piper_seneca",

    # Cessna 310
    "cessna 310": "cessna_310",
    "c310": "cessna_310",
    "310": "cessna_310",
}

# ─────────────────────────────────────────────────────────────
# PHRASE MAPPINGS
# Source: context1 [Layer 0: Intelligent Defaults]
# ─────────────────────────────────────────────────────────────

FATIGUE_MAPPINGS = {
    # Direct descriptors
    "fresh": 0.1,
    "rested": 0.1,
    "alert": 0.1,
    "well rested": 0.1,
    "wide awake": 0.1,
    "slightly tired": 0.4,
    "moderate": 0.5,
    "moderately tired": 0.5,
    "tired": 0.6,
    "fatigued": 0.7,
    "very tired": 0.7,
    "exhausted": 0.9,
    "severely fatigued": 0.9,
    "dead tired": 0.95,

    # Time-based descriptors (flight hours)
    "1 hour": 0.2,
    "2 hours": 0.3,
    "3 hours": 0.5,
    "4 hours": 0.6,
    "5 hours": 0.7,
    "6 hours": 0.8,
    "7 hours": 0.9,
    "8 hours": 0.95,
}

FUEL_QUANTITY_MAPPINGS = {
    # Relative descriptors — multiplied by aircraft capacity
    "full": 1.0,
    "full tanks": 1.0,
    "topped off": 1.0,
    "three quarters": 0.75,
    "3/4": 0.75,
    "half": 0.5,
    "half fuel": 0.5,
    "half tanks": 0.5,
    "quarter": 0.25,
    "1/4": 0.25,
    "low": 0.15,
    "low on fuel": 0.15,
    "nearly empty": 0.05,
    "empty": 0.0,
}

ALTITUDE_PHRASE_MAPPINGS = {
    "taking off": 500,
    "taking_off": 500,   # LLM returns underscored version
    "takeoff": 500,
    "climb": 3000,
    "climbing": 3000,
    "cruise": 5000,
    "cruising": 5000,
    "high altitude": 10000,
    "high": 10000,
    "very high": 15000,
    "service ceiling": 16500,     # C182 published service ceiling (POH p4)
    "max altitude": 16500,
    "maximum altitude": 16500,
    "approach": 1500,
    "on approach": 1500,
    "final": 500,
    "pattern": 1000,
    "traffic pattern": 1000,
    "low altitude": 2000,
    "low": 2000,
}

SKILL_MAPPINGS = {
    "student": "student",
    "trainee": "student",
    "student pilot": "student",
    "solo": "student",
    "private": "private",
    "private pilot": "private",
    "ppl": "private",
    "recreational": "private",
    "commercial": "commercial",
    "cpl": "commercial",
    "experienced": "commercial",
    "instrument rated": "instrument",
    "ifr": "instrument",
    "instrument": "instrument",
    "airline": "airline",
    "atp": "airline",
    "ifr": "instrument",       # IFR rated = instrument rating
    "ifr rated": "instrument",
    "instrument rated": "instrument",
    "experienced": "commercial",
    "seasoned": "commercial",
    "veteran": "commercial",
    "military": "commercial",  # Military trained → commercial equivalent
    "military trained": "commercial",
    "airline pilot": "airline",
    "professional": "airline",
    "novice": "private",
    "beginner": "student",
    "expert": "airline",
}

# ─────────────────────────────────────────────────────────────
# FAILURE COMPONENT MAPPINGS
# Source: context1 [Layer 0: Failure mappings]
# ─────────────────────────────────────────────────────────────

FAILURE_COMPONENT_MAPPINGS = {
    # Engine
    "engine failure": {"component": "engine_left", "severity": "catastrophic"},
    "engine fails": {"component": "engine_left", "severity": "catastrophic"},
    "engine stops": {"component": "engine_left", "severity": "catastrophic"},
    "engine out": {"component": "engine_left", "severity": "catastrophic"},
    "engine failure left": {"component": "engine_left", "severity": "catastrophic"},
    "engine failure right": {"component": "engine_right", "severity": "catastrophic"},
    "left engine": {"component": "engine_left", "severity": "catastrophic"},
    "right engine": {"component": "engine_right", "severity": "catastrophic"},
    "dual engine failure": {"component": "engine_both", "severity": "catastrophic"},
    "both engines": {"component": "engine_both", "severity": "catastrophic"},
    "both engines fail": {"component": "engine_both", "severity": "catastrophic"},
    "total power loss": {"component": "engine_both", "severity": "catastrophic"},

    # Electrical
    "alternator failure": {"component": "alternator", "severity": "catastrophic"},
    "alternator fails": {"component": "alternator", "severity": "catastrophic"},
    "alternator": {"component": "alternator", "severity": "catastrophic"},
    "power loss": {"component": "alternator", "severity": "catastrophic"},
    "electrical failure": {"component": "alternator", "severity": "catastrophic"},
    "electrical system failure": {"component": "alternator", "severity": "catastrophic"},
    "battery failure": {"component": "battery", "severity": "catastrophic"},
    "battery dead": {"component": "battery", "severity": "catastrophic"},

    # Fuel
    "fuel pump failure": {"component": "fuel_pump", "severity": "catastrophic"},
    # Fuel selector failures — engineers write these several ways
    "fuel selector empty tank": {"component": "fuel_selector", "severity": "catastrophic"},
    "fuel selector off": {"component": "fuel_selector", "severity": "catastrophic"},
    "fuel selector wrong tank": {"component": "fuel_selector", "severity": "catastrophic"},
    "fuel selector on empty": {"component": "fuel_selector", "severity": "catastrophic"},
    "fuel starvation": {"component": "fuel_selector", "severity": "catastrophic"},
    # Carburetor — engineers write as icing or failure
    "carburetor ice": {"component": "carburetor", "severity": "degraded"},
    "carburetor icing": {"component": "carburetor", "severity": "degraded"},
    "carburetor failure": {"component": "carburetor", "severity": "catastrophic"},
    "carb ice": {"component": "carburetor", "severity": "degraded"},
    # Voltage regulator
    "voltage regulator fault": {"component": "voltage_regulator", "severity": "catastrophic"},
    "voltage regulator failure": {"component": "voltage_regulator", "severity": "catastrophic"},
    "regulator fault": {"component": "voltage_regulator", "severity": "catastrophic"},
    "fuel pump fails": {"component": "fuel_pump", "severity": "catastrophic"},
    "fuel pump problem": {"component": "fuel_pump", "severity": "degraded"},
    "fuel pump degraded": {"component": "fuel_pump", "severity": "degraded"},
    "fuel imbalance": {"component": "fuel_selector", "severity": "degraded"},
    "fuel leak": {"component": "fuel_system", "severity": "degraded"},
    "low fuel": {"component": "fuel_system", "severity": "degraded"},
    "fuel exhaustion": {"component": "fuel_system", "severity": "catastrophic"},

    # Instruments / Vacuum
    "vacuum failure": {"component": "vacuum_pump", "severity": "catastrophic"},
    "vacuum pump failure": {"component": "vacuum_pump", "severity": "catastrophic"},
    "vacuum pump fails": {"component": "vacuum_pump", "severity": "catastrophic"},
    "gyro failure": {"component": "gyroscope", "severity": "catastrophic"},
    "instrument failure": {"component": "gyroscope", "severity": "catastrophic"},
    "attitude indicator fails": {"component": "gyroscope", "severity": "catastrophic"},

    # Flight controls
    "hydraulic failure": {"component": "hydraulics", "severity": "catastrophic"},
    "hydraulics fail": {"component": "hydraulics", "severity": "catastrophic"},
    "landing gear failure": {"component": "landing_gear", "severity": "degraded"},
    "gear failure": {"component": "landing_gear", "severity": "degraded"},
    "flap failure": {"component": "flaps", "severity": "degraded"},
    "flaps fail": {"component": "flaps", "severity": "degraded"},
    "rudder failure": {"component": "rudder", "severity": "catastrophic"},
    "aileron failure": {"component": "ailerons", "severity": "catastrophic"},
    "elevator failure": {"component": "elevator", "severity": "catastrophic"},
}

# ─────────────────────────────────────────────────────────────
# PHYSICAL VALIDATION RANGES
# Values outside these ranges are physically impossible
# ─────────────────────────────────────────────────────────────

VALIDATION_RANGES = {
    "altitude_ft": (0, 45000),
    "airspeed_knots": (0, 350),
    "pitch_deg": (-90, 90),
    "roll_deg": (-180, 180),
    "fuel_left_gal": (0, 200),
    "fuel_right_gal": (0, 200),
    "engine_rpm": (0, 3500),
    "electrical_voltage": (0, 32),
    "battery_charge": (0, 100),
    "fatigue": (0.0, 1.0),
    "trigger_time_seconds": (0, 7200),
    "severity_values": ["catastrophic", "degraded", "intermittent"],
    "skill_values": ["student", "private", "commercial", "instrument", "airline"],
}

# ─────────────────────────────────────────────────────────────
# SUPPORTED AIRCRAFT LIST (for API responses)
# ─────────────────────────────────────────────────────────────

SUPPORTED_AIRCRAFT = list(AIRCRAFT_DEFAULTS.keys())