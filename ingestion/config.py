"""
FailureScope — Centralized Path Configuration

All paths resolve to:
  /opt/failurescope/   ← project data root (root partition, 135GB)
  /opt/datasets/       ← user-placed manual documents (POHs, engine manuals)

DO NOT hardcode paths elsewhere. Always import from here.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────────────────────

# Project code root (wherever the repo lives)
PROJECT_ROOT = Path(__file__).parent.parent

# Data root — on root partition (135GB free)
# All heavy files, models, trajectories go here
DATA_ROOT = Path("/opt/failurescope/data")

# User-placed manual documents (POHs, engine manuals)
# User downloads these and places them here
DATASETS_ROOT = Path("/opt/datasets")

# ─────────────────────────────────────────────────────────────
# DATA SUBDIRECTORIES
# ─────────────────────────────────────────────────────────────

# Source Group 1 — Aircraft docs (POH, TCDS, engine manuals)
AIRCRAFT_DOCS_DIR   = DATA_ROOT / "aircraft_docs"

# Source Group 2 — Physics reference docs (FAA, NASA)
PHYSICS_DOCS_DIR    = DATA_ROOT / "physics_docs"

# Source Group 3 — NTSB accident data
NTSB_DATA_DIR       = DATA_ROOT / "ntsb_data"

# Page images extracted from PDFs (for vision model)
PAGE_IMAGES_DIR     = DATA_ROOT / "page_images"

# Human-validated spec outputs
VALIDATED_DIR       = DATA_ROOT / "validated"

# physics_cache.json — frozen after build-time validation
CACHE_DIR           = DATA_ROOT / "cache"
PHYSICS_CACHE_PATH  = CACHE_DIR / "physics_cache.json"
VALIDATED_SPECS_PATH= CACHE_DIR / "validated_specs.json"

# ChromaDB vector store (text passages for Layer 1)
CHROMADB_DIR        = DATA_ROOT / "chromadb"

# Constraint graphs output (Layer 1 deliverable)
CONSTRAINT_GRAPHS_DIR = DATA_ROOT / "constraint_graphs"

# Simulation trajectories (Layer 2 deliverable)
TRAJECTORIES_DIR    = DATA_ROOT / "simulation_trajectories"

# Trained models (Layer 3 deliverable)
MODELS_DIR          = Path("/opt/failurescope/models")


# ─────────────────────────────────────────────────────────────
# MANUAL DOCUMENT EXPECTED LOCATIONS
# Maps doc_id → expected filename in /opt/datasets/
#
# User places files here. Pipeline scans and links.
# Filename matching is case-insensitive, partial match allowed.
# ─────────────────────────────────────────────────────────────

MANUAL_DOC_HINTS = {
    # POHs — Source Group 1A
    "cessna_182q_poh_1977": {
        "search_terms": ["182q", "cessna_182q", "cessna 182q"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "cessna_182",
        "dest_name": "cessna_182q_poh_1977.pdf",
    },
    "cessna_182p_poh_1976": {
        "search_terms": ["182p", "skylane_182p", "skylane 182p"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "cessna_182",
        "dest_name": "cessna_182p_poh_1976.pdf",
    },
    "cessna_172n_poh_1978": {
        "search_terms": ["172n", "172", "skyhawk", "1978"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "cessna_172",
        "dest_name": "cessna_172n_poh_1978.pdf",
    },
    "beechcraft_bonanza_v35_poh": {
        "search_terms": ["bonanza", "v35", "beechcraft", "beech"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "beechcraft_bonanza",
        "dest_name": "bonanza_v35_poh.pdf",
    },
    "piper_seneca_ii_poh": {
        "search_terms": ["seneca", "pa-34", "pa34", "seneca ii"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "piper_seneca",
        "dest_name": "seneca_ii_poh.pdf",
    },
    "piper_cherokee_warrior_mm": {
        "search_terms": ["cherokee", "warrior", "pa-28", "pa28"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "piper_cherokee",
        "dest_name": "cherokee_warrior_mm_1995.pdf",
    },
    "cessna_310k_owners_manual": {
        "search_terms": ["310k", "310", "cessna 310"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "cessna_310",
        "dest_name": "cessna_310k_1966.pdf",
    },

    # Service/Maintenance manuals — Source Group 1B
    "cessna_182_service_manual": {
        "search_terms": ["d2006", "skylane", "service manual", "1969"],
        "dest_dir": __import__("pathlib").Path("/opt/failurescope/data/aircraft_docs/cessna_182"),
        "dest_name": "cessna_182_service_manual_d2006.pdf",
    },

    # Engine manuals — Source Group 1C
    "continental_o470u_spec": {
        "search_terms": ["o470u", "o-470-u", "o470u.pdf"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "engines" / "continental",
        "dest_name": "continental_o470u_spec.pdf",
    },
    "continental_o470_overhaul": {
        "search_terms": ["o-470", "io-470", "x30022", "overhaul"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "engines" / "continental",
        "dest_name": "continental_o470_overhaul_1976.pdf",
    },
    "lycoming_o320_operators": {
        "search_terms": ["o-320", "o320", "60297-30"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "engines" / "lycoming",
        "dest_name": "lycoming_o320_operators.pdf",
    },
    "lycoming_o360_operators": {
        "search_terms": ["o-360", "o360", "60297-12", "ho-360", "io-360"],
        "dest_dir": AIRCRAFT_DOCS_DIR / "engines" / "lycoming",
        "dest_name": "lycoming_o360_operators.pdf",
    },
}


# ─────────────────────────────────────────────────────────────
# DIRECTORY CREATION
# ─────────────────────────────────────────────────────────────

ALL_DIRS = [
    AIRCRAFT_DOCS_DIR / "cessna_182",
    AIRCRAFT_DOCS_DIR / "cessna_172",
    AIRCRAFT_DOCS_DIR / "beechcraft_bonanza",
    AIRCRAFT_DOCS_DIR / "piper_seneca",
    AIRCRAFT_DOCS_DIR / "piper_cherokee",
    AIRCRAFT_DOCS_DIR / "cessna_310",
    PHYSICS_DOCS_DIR,
    NTSB_DATA_DIR,
    PAGE_IMAGES_DIR,
    VALIDATED_DIR,
    CACHE_DIR,
    CHROMADB_DIR,
    AIRCRAFT_DOCS_DIR / "engines" / "continental",
    AIRCRAFT_DOCS_DIR / "engines" / "lycoming",
    CONSTRAINT_GRAPHS_DIR,
    TRAJECTORIES_DIR,
    MODELS_DIR,
]


def ensure_dirs():
    """Create all required directories. Safe to call multiple times."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)