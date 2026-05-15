"""
FailureScope — Full System State Check
Run: python full_system_check.py
"""
import json
from pathlib import Path

DATA_ROOT = Path("/opt/failurescope/data")
PROJECT_ROOT = Path.home() / "Documents/VSCODE/failurescope"

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

issues = []
warnings = []

def check(label, condition, error_msg="", warn=False):
    if condition:
        print(f"  {PASS} {label}")
    elif warn:
        print(f"  {WARN} {label} — {error_msg}")
        warnings.append(f"{label}: {error_msg}")
    else:
        print(f"  {FAIL} {label} — {error_msg}")
        issues.append(f"{label}: {error_msg}")

print("\n" + "="*65)
print("FAILURESCOPE — FULL SYSTEM STATE CHECK")
print("="*65)

# ── 1. Data files ──────────────────────────────────────────────
print("\n── INGESTED DOCUMENTS ─────────────────────────────────────────")

docs = {
    "cessna_182q_poh_1977":       DATA_ROOT / "aircraft_docs/cessna_182/cessna_182q_poh_1977.pdf",
    "cessna_182p_poh_1976":       DATA_ROOT / "aircraft_docs/cessna_182/cessna_182p_poh_1976.pdf",
    "cessna_182_service_manual":  DATA_ROOT / "aircraft_docs/cessna_182/cessna_182_service_manual_d2006.pdf",
    "cessna_172n_poh_1978":       DATA_ROOT / "aircraft_docs/cessna_172/cessna_172n_poh_1978.pdf",
    "bonanza_v35_poh":            DATA_ROOT / "aircraft_docs/beechcraft_bonanza/bonanza_v35_poh.pdf",
    "seneca_ii_poh":              DATA_ROOT / "aircraft_docs/piper_seneca/seneca_ii_poh.pdf",
    "cherokee_warrior_mm":        DATA_ROOT / "aircraft_docs/piper_cherokee/cherokee_warrior_mm_1995.pdf",
    "cessna_310k_manual":         DATA_ROOT / "aircraft_docs/cessna_310/cessna_310k_1966.pdf",
    "continental_o470u_spec":     DATA_ROOT / "aircraft_docs/engines/continental/continental_o470u_spec.pdf",
    "continental_o470_overhaul":  DATA_ROOT / "aircraft_docs/engines/continental/continental_o470_overhaul_1976.pdf",
    "lycoming_o320":              DATA_ROOT / "aircraft_docs/engines/lycoming/lycoming_o320_operators.pdf",
    "lycoming_o360":              DATA_ROOT / "aircraft_docs/engines/lycoming/lycoming_o360_operators.pdf",
    "faa_ac43_13_1b":             DATA_ROOT / "physics_docs/faa_ac43_13_1b.pdf",
    "nasa_reliability":           DATA_ROOT / "physics_docs/nasa_cr2001_210647.pdf",
    "faa_phak":                   DATA_ROOT / "physics_docs/faa_h8083_25c_phak.pdf",
    "faa_afh":                    DATA_ROOT / "physics_docs/faa_h8083_3c_afh.pdf",
    "ntsb_avall":                 DATA_ROOT / "ntsb_data/avall.zip",
}

for doc_id, path in docs.items():
    if path.exists():
        mb = path.stat().st_size / (1024*1024)
        print(f"  {PASS} {doc_id} ({mb:.1f} MB)")
    else:
        print(f"  {FAIL} {doc_id} — MISSING")
        issues.append(f"Missing doc: {doc_id}")

# ── 2. Page images (processed) ────────────────────────────────
print("\n── PAGE PROCESSING STATUS ──────────────────────────────────────")
page_image_docs = ["cessna_182q_poh_1977", "cessna_182_service_manual"]
for doc_id in page_image_docs:
    pages_json = DATA_ROOT / "page_images" / doc_id / f"{doc_id}_pages.json"
    if pages_json.exists():
        pages = json.loads(pages_json.read_text())
        has_text = sum(1 for p in pages if len(p.get('text','')) > 50)
        print(f"  {PASS} {doc_id}: {len(pages)} pages, {has_text} with OCR text")
    else:
        print(f"  {WARN} {doc_id}: not processed yet")
        warnings.append(f"{doc_id} not processed")

# ── 3. Extracted specs ────────────────────────────────────────
print("\n── EXTRACTION STATUS ───────────────────────────────────────────")
extracted_docs = ["cessna_182q_poh_1977", "cessna_182_service_manual"]
for doc_id in extracted_docs:
    path = DATA_ROOT / "validated" / f"{doc_id}_extracted.json"
    if path.exists():
        data = json.loads(path.read_text())
        found = sum(1 for v in data.values() if v.get('raw_value') is not None)
        total = len(data)
        print(f"  {PASS} {doc_id}: {found}/{total} fields extracted")
    else:
        print(f"  {WARN} {doc_id}: not extracted yet")

# ── 4. Validated specs ────────────────────────────────────────
print("\n── VALIDATION STATUS ───────────────────────────────────────────")
validated_docs = ["cessna_182q_poh_1977", "cessna_182_service_manual"]
for doc_id in validated_docs:
    path = DATA_ROOT / "validated" / f"{doc_id}_validated.json"
    if path.exists():
        data = json.loads(path.read_text())
        accepted = sum(1 for v in data.values() if v.get('status') in ('accepted', 'corrected'))
        skipped = sum(1 for v in data.values() if v.get('status') == 'skipped')
        total = len(data)
        print(f"  {PASS} {doc_id}: {accepted} accepted/corrected, {skipped} skipped, {total} total")
        # Check critical fields
        for field, expected in [
            ("system_voltage", "14"),
            ("vne_knots", "172"),
            ("max_rpm", "2400"),
            ("fuel_capacity_left_gal", "30.5"),
        ]:
            if field in data:
                val = data[field].get('raw_value')
                status = data[field].get('status')
                if val and status not in ('skipped', 'rejected'):
                    ok = val == expected
                    mark = PASS if ok else WARN
                    print(f"    {mark} {field} = {val} (expected {expected})")
                    if not ok:
                        warnings.append(f"{doc_id}.{field} = {val}, expected {expected}")
    else:
        print(f"  {WARN} {doc_id}: not validated yet")
        warnings.append(f"{doc_id} not validated")

# ── 5. Cache status ───────────────────────────────────────────
print("\n── CACHE STATUS ────────────────────────────────────────────────")
cache_path = DATA_ROOT / "cache/physics_cache.json"
chromadb_path = DATA_ROOT / "chromadb"
check("physics_cache.json built", cache_path.exists(), "run --stage build-cache", warn=True)
check("ChromaDB directory exists", chromadb_path.exists(), "run --stage build-cache", warn=True)

# ── 6. Project source files ───────────────────────────────────
print("\n── PROJECT SOURCE FILES ────────────────────────────────────────")
source_files = {
    "backend/app.py":                    "FastAPI app",
    "backend/layer0/__init__.py":        "Layer 0 init",
    "backend/layer0/defaults.py":        "Aircraft defaults",
    "backend/layer0/nlp_parser.py":      "NLP parser",
    "backend/layer0/schemas.py":         "Pydantic schemas",
    "backend/layer0/validator.py":       "Scenario validator",
    "ingestion/config.py":              "Path config",
    "ingestion/dataset_linker.py":      "Dataset linker",
    "ingestion/downloader.py":          "Auto downloader",
    "ingestion/pdf_processor.py":       "PDF processor",
    "ingestion/ocr_extractor.py":       "OCR extractor",
    "ingestion/spec_extractor.py":      "Vision extractor",
    "ingestion/validator_cli.py":       "Validator CLI",
    "ingestion/cache_builder.py":       "Cache builder",
    "ingestion/pipeline.py":            "Pipeline orchestrator",
    "ingestion/schemas.py":             "Document registry",
    "tests/test_layer0.py":             "Layer 0 tests",
    "requirements.txt":                 "Dependencies",
}
for rel_path, desc in source_files.items():
    full = PROJECT_ROOT / rel_path
    if full.exists():
        kb = full.stat().st_size / 1024
        print(f"  {PASS} {rel_path} ({kb:.0f} KB) — {desc}")
    else:
        print(f"  {FAIL} {rel_path} MISSING — {desc}")
        issues.append(f"Missing source: {rel_path}")

# ── 7. Ollama models ──────────────────────────────────────────
print("\n── OLLAMA MODELS ───────────────────────────────────────────────")
import subprocess
result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    models = [l.split()[0] for l in lines[1:] if l.strip()]
    for m in ['mistral:7b', 'llava:7b']:
        if any(m in model for model in models):
            print(f"  {PASS} {m}")
        else:
            print(f"  {FAIL} {m} — not pulled")
            issues.append(f"Missing model: {m}")
else:
    print(f"  {FAIL} Cannot connect to Ollama")
    issues.append("Ollama not running")

# ── 8. Disk space ─────────────────────────────────────────────
print("\n── DISK SPACE ──────────────────────────────────────────────────")
import shutil
root_usage = shutil.disk_usage("/opt")
home_usage = shutil.disk_usage("/home")
root_free_gb = root_usage.free / (1024**3)
home_free_gb = home_usage.free / (1024**3)
check(f"/opt  — {root_free_gb:.0f} GB free", root_free_gb > 50, f"Only {root_free_gb:.0f}GB — need 50GB+ for trajectories")
check(f"/home — {home_free_gb:.0f} GB free", home_free_gb > 5, f"Only {home_free_gb:.0f}GB free")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
if not issues and not warnings:
    print(f"  {PASS} ALL CHECKS PASSED — ready to proceed")
else:
    if issues:
        print(f"\n  BLOCKING ISSUES ({len(issues)}):")
        for i in issues:
            print(f"    {FAIL} {i}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    {WARN} {w}")
print("="*65 + "\n")