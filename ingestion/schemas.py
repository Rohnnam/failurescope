"""
Ingestion: Extraction Schemas
Defines EXACTLY which fields to extract from each document type.

ARCHITECTURAL PRINCIPLE: Vision model is a READER, not an INTERPRETER.
The schema constrains output to known fields only.
If a value cannot be clearly read → null. Never inferred.

Every field has:
  - field_id: unique key used in physics_cache.json
  - description: what to look for on the page
  - unit: expected unit (for validation)
  - section_hint: where in the POH this typically appears
  - layer_use: which FailureScope layer uses this value
"""

# ─────────────────────────────────────────────────────────────
# POH EXTRACTION SCHEMA
# Fields extracted from Pilot's Operating Handbook
# ─────────────────────────────────────────────────────────────

POH_SCHEMA = {
    "aircraft_identity": {
        "fields": [
            {
                "field_id": "aircraft_model",
                "description": "Aircraft model designation e.g. '182Q', '172N'",
                "unit": "string",
                "section_hint": "Cover page",
                "layer_use": "layer0_defaults",
            },
            {
                "field_id": "engine_model",
                "description": "Engine model number e.g. 'O-470-U', 'O-320-H2AD'",
                "unit": "string",
                "section_hint": "Cover page or Section 1",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "faa_type_certificate",
                "description": "FAA type certificate number e.g. '3A13'",
                "unit": "string",
                "section_hint": "Cover page",
                "layer_use": "citation",
            },
        ]
    },

    "electrical_system": {
        "fields": [
            {
                "field_id": "system_voltage",
                "description": "Electrical system voltage, typically 14V or 28V",
                "unit": "volts",
                "section_hint": "Systems description, electrical section",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "alternator_output_amps",
                "description": "Alternator rated output in amperes",
                "unit": "amps",
                "section_hint": "Electrical system description",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "battery_capacity_ah",
                "description": "Battery capacity in ampere-hours",
                "unit": "ampere-hours",
                "section_hint": "Electrical system description",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "min_fuel_pump_voltage",
                "description": "Minimum voltage required for electric fuel pump operation",
                "unit": "volts",
                "section_hint": "Fuel system or electrical system",
                "layer_use": "layer1_constraint_graph",
            },
        ]
    },

    "fuel_system": {
        "fields": [
            {
                "field_id": "fuel_capacity_total_gal",
                "description": "Total usable fuel capacity in US gallons",
                "unit": "US gallons",
                "section_hint": "Specifications or fuel system section",
                "layer_use": "layer0_defaults",
            },
            {
                "field_id": "fuel_capacity_left_gal",
                "description": "Left tank usable fuel capacity in US gallons",
                "unit": "US gallons",
                "section_hint": "Fuel system section",
                "layer_use": "layer0_defaults",
            },
            {
                "field_id": "fuel_capacity_right_gal",
                "description": "Right tank usable fuel capacity in US gallons",
                "unit": "US gallons",
                "section_hint": "Fuel system section",
                "layer_use": "layer0_defaults",
            },
            {
                "field_id": "cruise_fuel_flow_gph",
                "description": "Fuel flow at 75% power cruise in gallons per hour",
                "unit": "GPH",
                "section_hint": "Performance section or cruise performance table",
                "layer_use": "layer2_simulation",
            },
        ]
    },

    "engine_performance": {
        "fields": [
            {
                "field_id": "max_rpm",
                "description": "Maximum continuous engine RPM",
                "unit": "RPM",
                "section_hint": "Limitations section or engine specifications",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "max_power_bhp",
                "description": "Maximum engine power in brake horsepower",
                "unit": "BHP",
                "section_hint": "Specifications section",
                "layer_use": "layer2_simulation",
            },
            {
                "field_id": "engine_count",
                "description": "Number of engines: 1 for single, 2 for twin",
                "unit": "count",
                "section_hint": "Cover or general description",
                "layer_use": "layer0_defaults",
            },
        ]
    },

    "flight_envelope": {
        "fields": [
            {
                "field_id": "vne_knots",
                "description": "Never exceed speed (Vne) in knots",
                "unit": "knots",
                "section_hint": "Limitations section, airspeed indicator markings",
                "layer_use": "layer0_validator",
            },
            {
                "field_id": "vs0_knots",
                "description": "Stall speed flaps down (Vs0) in knots CAS",
                "unit": "knots CAS",
                "section_hint": "Limitations section",
                "layer_use": "layer0_validator",
            },
            {
                "field_id": "service_ceiling_ft",
                "description": "Service ceiling in feet MSL",
                "unit": "feet MSL",
                "section_hint": "Performance specifications",
                "layer_use": "layer0_validator",
            },
            {
                "field_id": "max_gross_weight_lbs",
                "description": "Maximum takeoff gross weight in pounds",
                "unit": "lbs",
                "section_hint": "Limitations or specifications",
                "layer_use": "layer2_simulation",
            },
        ]
    },

    "emergency_procedures": {
        "fields": [
            {
                "field_id": "alternator_failure_procedure",
                "description": "Full text of alternator failure emergency checklist",
                "unit": "text",
                "section_hint": "Emergency procedures section",
                "layer_use": "layer1_cascade_rules",
            },
            {
                "field_id": "engine_failure_procedure",
                "description": "Full text of engine failure emergency checklist",
                "unit": "text",
                "section_hint": "Emergency procedures section",
                "layer_use": "layer1_cascade_rules",
            },
            {
                "field_id": "fuel_system_failure_procedure",
                "description": "Full text of fuel system emergency checklist",
                "unit": "text",
                "section_hint": "Emergency procedures section",
                "layer_use": "layer1_cascade_rules",
            },
        ]
    },
}


# ─────────────────────────────────────────────────────────────
# PHYSICS DOCUMENT SCHEMA
# Fields extracted from FAA handbooks / NASA docs
# ─────────────────────────────────────────────────────────────

PHYSICS_SCHEMA = {
    "standard_atmosphere": {
        "source_doc": "FAA-H-8083-25C or ICAO Doc 7488",
        "fields": [
            {
                "field_id": "air_density_sl_slug_ft3",
                "description": "Air density at sea level in slug/ft³",
                "unit": "slug/ft³",
                "section_hint": "Chapter 11 Performance or atmosphere tables",
                "layer_use": "layer2_simulation",
            },
            {
                "field_id": "air_density_5000ft_slug_ft3",
                "description": "Air density at 5000 ft MSL in slug/ft³",
                "unit": "slug/ft³",
                "section_hint": "Standard atmosphere table",
                "layer_use": "layer2_simulation",
            },
            {
                "field_id": "air_density_10000ft_slug_ft3",
                "description": "Air density at 10000 ft MSL in slug/ft³",
                "unit": "slug/ft³",
                "section_hint": "Standard atmosphere table",
                "layer_use": "layer2_simulation",
            },
        ]
    },

    "aerodynamic_equations": {
        "source_doc": "FAA-H-8083-25C",
        "fields": [
            {
                "field_id": "lift_equation_form",
                "description": "Standard lift equation: L = CL * 0.5 * rho * V^2 * S",
                "unit": "equation",
                "section_hint": "Chapter 4 or 5 Principles of Flight",
                "layer_use": "layer2_simulation",
            },
            {
                "field_id": "drag_equation_form",
                "description": "Standard drag equation: D = CD * 0.5 * rho * V^2 * S",
                "unit": "equation",
                "section_hint": "Chapter 4 or 5",
                "layer_use": "layer2_simulation",
            },
        ]
    },

    "electrical_standards": {
        "source_doc": "FAA AC 43.13-1B",
        "fields": [
            {
                "field_id": "nominal_28v_bus_range",
                "description": "Acceptable voltage range for 28V electrical bus",
                "unit": "volts",
                "section_hint": "Chapter 11 Electrical Systems",
                "layer_use": "layer1_constraint_graph",
            },
            {
                "field_id": "battery_discharge_rate_formula",
                "description": "Battery discharge rate calculation method",
                "unit": "text/equation",
                "section_hint": "Electrical chapter",
                "layer_use": "layer2_simulation",
            },
        ]
    },
}


# ─────────────────────────────────────────────────────────────
# NTSB SCHEMA
# Fields extracted from NTSB accident database
# ─────────────────────────────────────────────────────────────

NTSB_SCHEMA = {
    "accident_record": {
        "fields": [
            "event_id",           # NTSB case number e.g. WPR13FA089
            "event_date",         # Date of accident
            "aircraft_make",      # e.g. CESSNA
            "aircraft_model",     # e.g. 182Q
            "far_description",    # Flight rules
            "phase_of_flight",    # e.g. CRUISE, APPROACH
            "probable_cause",     # Full probable cause text
            "narrative",          # Full accident narrative
            "total_fatal_injuries",
            "total_serious_injuries",
            "weather_condition",  # VMC/IMC
            "pilot_total_hours",  # Total flight experience
        ]
    }
}


# ─────────────────────────────────────────────────────────────
# DOCUMENT REGISTRY
# Maps document files to their schema and aircraft
# ─────────────────────────────────────────────────────────────

DOCUMENT_REGISTRY = {
    # Aircraft POHs — Source Group 1A
    "cessna_182q_poh_1977": {
        "aircraft": "cessna_182",
        "doc_type": "POH",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/cessna_182/cessna_182q_poh_1977.pdf",
        "citation": {
            "document": "Cessna Skylane 182Q Pilot's Operating Handbook",
            "manufacturer": "Cessna Aircraft Company",
            "year": 1977,
            "document_number": "D1087-13-RPC-300-2/92",
            "faa_approval": "CAR Part 3",
        },
    },
    "cessna_182p_poh_1976": {
        "aircraft": "cessna_182",
        "doc_type": "POH",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/cessna_182/cessna_182p_poh_1976.pdf",
        "citation": {
            "document": "Cessna Skylane 182P Pilot's Operating Handbook",
            "manufacturer": "Cessna Aircraft Company",
            "year": 1976,
            "faa_approval": "CAR Part 3",
        },
    },
    "cessna_172n_poh_1978": {
        "aircraft": "cessna_172",
        "doc_type": "POH",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/cessna_172/cessna_172n_poh_1978.pdf",
        "citation": {
            "document": "Cessna Skyhawk 172N Pilot's Operating Handbook",
            "manufacturer": "Cessna Aircraft Company",
            "year": 1978,
            "faa_approval": "CAR Part 3",
        },
    },
    "beechcraft_bonanza_v35_poh": {
        "aircraft": "beechcraft_bonanza",
        "doc_type": "POH",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/beechcraft_bonanza/bonanza_v35_poh.pdf",
        "citation": {
            "document": "Beechcraft Bonanza V35, V35A, V35B Pilot's Operating Handbook",
            "manufacturer": "Beech Aircraft Corporation",
            "year": 1982,
            "part_number": "35-590118-31B",
            "faa_approval": "CAR 3",
        },
    },
    "piper_seneca_ii_poh": {
        "aircraft": "piper_seneca",
        "doc_type": "POH",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/piper_seneca/seneca_ii_poh.pdf",
        "citation": {
            "document": "Piper Seneca II PA-34-200T Pilot's Operating Handbook",
            "manufacturer": "Piper Aircraft Corporation",
            "report": "VB-850",
            "faa_approval": "FAR 23",
        },
    },
    "piper_cherokee_warrior_mm": {
        "aircraft": "piper_cherokee",
        "doc_type": "MAINTENANCE_MANUAL",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/piper_cherokee/cherokee_warrior_mm_1995.pdf",
        "citation": {
            "document": "Piper Cherokee Warrior PA-28-151 Airplane Maintenance Manual",
            "manufacturer": "Piper Aircraft Corporation",
            "part_number": "761 539",
            "year": 1995,
        },
    },
    "cessna_310k_owners_manual": {
        "aircraft": "cessna_310",
        "doc_type": "OWNERS_MANUAL",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/cessna_310/cessna_310k_1966.pdf",
        "citation": {
            "document": "Cessna 310K Owner's Manual",
            "manufacturer": "Cessna Aircraft Company",
            "year": 1966,
        },
    },

    "cessna_182_service_manual": {
        "aircraft": "cessna_182",
        "doc_type": "SERVICE_MANUAL",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/cessna_182/cessna_182_service_manual_d2006.pdf",
        "citation": {
            "document": "Cessna 182 and Skylane Series Service Manual",
            "manufacturer": "Cessna Aircraft Company",
            "document_number": "D2006R4-13",
            "year_range": "1969-1976",
            "revision": "Revision 4, 1 March 2004",
            "sections_of_interest": [
                "Section 16 - Electrical Systems",
                "Section 12 - Fuel System",
                "Section 20 - Wiring Diagrams"
            ],
        },
    },

    "continental_o470_overhaul": {
        "aircraft": "cessna_182",
        "doc_type": "ENGINE_OVERHAUL",
        "schema": POH_SCHEMA,
        "expected_path": "data/aircraft_docs/engines/continental/continental_o470_overhaul_1976.pdf",
        "citation": {
            "document": "Maintenance and Overhaul Manual for O-470 and IO-470 Series Aircraft Engines",
            "manufacturer": "Teledyne Continental Motors",
            "form_number": "X30022A",
            "year": 1976,
            "obtained_from": "bomar.biz",
        },
    },

    # Physics Documents — Source Group 2
    "faa_phak_8083_25c": {
        "aircraft": None,
        "doc_type": "PHYSICS_REF",
        "schema": PHYSICS_SCHEMA,
        "expected_path": "data/physics_docs/faa_h8083_25c_phak.pdf",
        "citation": {
            "document": "Pilot's Handbook of Aeronautical Knowledge",
            "id": "FAA-H-8083-25C",
            "publisher": "FAA",
            "year": 2023,
            "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/phak",
        },
    },
    "faa_afh_8083_3c": {
        "aircraft": None,
        "doc_type": "PHYSICS_REF",
        "schema": PHYSICS_SCHEMA,
        "expected_path": "data/physics_docs/faa_h8083_3c_afh.pdf",
        "citation": {
            "document": "Airplane Flying Handbook",
            "id": "FAA-H-8083-3C",
            "publisher": "FAA",
            "year": 2023,
            "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/airplane_handbook",
        },
    },
    "faa_ac43_13_1b": {
        "aircraft": None,
        "doc_type": "PHYSICS_REF",
        "schema": PHYSICS_SCHEMA,
        "expected_path": "data/physics_docs/faa_ac43_13_1b.pdf",
        "citation": {
            "document": "Acceptable Methods, Techniques, and Practices - Aircraft Inspection and Repair",
            "id": "FAA AC 43.13-1B",
            "publisher": "FAA",
            "year": 1998,
            "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_Full.pdf",
        },
    },
    "nasa_ga_reliability_2001": {
        "aircraft": None,
        "doc_type": "RELIABILITY_REF",
        "schema": PHYSICS_SCHEMA,
        "expected_path": "data/physics_docs/nasa_cr2001_210647.pdf",
        "citation": {
            "document": "General Aviation Aircraft Reliability Study",
            "id": "NASA/CR-2001-210647",
            "publisher": "NASA",
            "year": 2001,
            "url": "https://www.cs.odu.edu/~mln/ltrs-pdfs/NASA-2001-cr210647.pdf",
        },
    },
}