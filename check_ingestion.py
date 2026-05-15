"""
FailureScope — Ingestion Status Check
Run: python check_ingestion.py
"""
from pathlib import Path

DATA_ROOT = Path("/opt/failurescope/data")

files = {
    # Manual docs — Group 1A POHs
    "cessna_182q_poh":        DATA_ROOT / "aircraft_docs/cessna_182/cessna_182q_poh_1977.pdf",
    "cessna_182p_poh":        DATA_ROOT / "aircraft_docs/cessna_182/cessna_182p_poh_1976.pdf",
    "cessna_172n_poh":        DATA_ROOT / "aircraft_docs/cessna_172/cessna_172n_poh_1978.pdf",
    "bonanza_v35_poh":        DATA_ROOT / "aircraft_docs/beechcraft_bonanza/bonanza_v35_poh.pdf",
    "seneca_ii_poh":          DATA_ROOT / "aircraft_docs/piper_seneca/seneca_ii_poh.pdf",
    "cherokee_warrior_mm":    DATA_ROOT / "aircraft_docs/piper_cherokee/cherokee_warrior_mm_1995.pdf",
    "cessna_310k_manual":     DATA_ROOT / "aircraft_docs/cessna_310/cessna_310k_1966.pdf",
    # Group 1C engine manuals
    "continental_o470u_spec": DATA_ROOT / "aircraft_docs/engines/continental/continental_o470u_spec.pdf",
    "continental_o470_oh":    DATA_ROOT / "aircraft_docs/engines/continental/continental_o470_overhaul_1976.pdf",
    "lycoming_o320":          DATA_ROOT / "aircraft_docs/engines/lycoming/lycoming_o320_operators.pdf",
    "lycoming_o360":          DATA_ROOT / "aircraft_docs/engines/lycoming/lycoming_o360_operators.pdf",
    # Auto-downloaded — Group 2 physics
    "faa_ac43_13_1b":         DATA_ROOT / "physics_docs/faa_ac43_13_1b.pdf",
    "nasa_reliability":       DATA_ROOT / "physics_docs/nasa_cr2001_210647.pdf",
    "faa_phak":               DATA_ROOT / "physics_docs/faa_h8083_25c_phak.pdf",
    "faa_afh":                DATA_ROOT / "physics_docs/faa_h8083_3c_afh.pdf",
    # Group 3 NTSB
    "ntsb_avall":             DATA_ROOT / "ntsb_data/avall.zip",
}

MIN_SIZES = {
    "faa_phak":   10_000_000,   # 10MB minimum
    "faa_afh":    10_000_000,
    "ntsb_avall": 10_000_000,
    "continental_o470_oh": 1_000_000,
    "lycoming_o360": 1_000_000,
}

print("=" * 65)
print("FAILURESCOPE — DATA INGESTION STATUS")
print("=" * 65)

ok = []
missing = []
empty = []

for doc_id, path in files.items():
    if not path.exists():
        missing.append(doc_id)
        print(f"  ✗ MISSING   {doc_id}")
        print(f"              {path}")
    else:
        size = path.stat().st_size
        size_mb = size / (1024*1024)
        min_size = MIN_SIZES.get(doc_id, 1000)
        if size < min_size:
            empty.append(doc_id)
            print(f"  ⚠ EMPTY     {doc_id}  ({size} bytes)")
        else:
            ok.append(doc_id)
            print(f"  ✓ OK        {doc_id}  ({size_mb:.1f} MB)")

print("=" * 65)
print(f"  OK: {len(ok)}/{len(files)}  |  Missing: {len(missing)}  |  Empty/corrupt: {len(empty)}")
print("=" * 65)

if not missing and not empty:
    print("  ALL DOCUMENTS PRESENT AND VALID")
    print("  Ready for: python -m ingestion.pipeline --stage process --doc cessna_182q_poh_1977")
else:
    if missing:
        print(f"\n  MISSING: {missing}")
    if empty:
        print(f"\n  EMPTY/CORRUPT: {empty}")