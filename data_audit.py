"""
FailureScope — Complete Data Audit (Fixed for actual JSON structure)
Validated JSON uses "final_value" key.
Physics cache uses cache["values"][field_id]["value"] structure.
"""
import json
from pathlib import Path

DATA_ROOT = Path("/opt/failurescope/data")
CACHE_PATH = DATA_ROOT / "cache/physics_cache.json"
VALIDATED_DIR = DATA_ROOT / "validated"

PASS = "✓"; FAIL = "✗"; WARN = "⚠"; FLAG = "🚩"
errors = []; warnings = []; flags = []

def err(msg):  print(f"  {FAIL} ERROR: {msg}"); errors.append(msg)
def warn(msg): print(f"  {WARN} WARN:  {msg}"); warnings.append(msg)
def flag(msg): print(f"  {FLAG} FLAG:  {msg}"); flags.append(msg)
def ok(msg):   print(f"  {PASS} {msg}")

# Ground truth: field_id → (expected_value, source_note)
GROUND_TRUTH = {
    "aircraft_model":          ("182q",    "POH cover"),
    "engine_model":            ("o-470-u", "POH p4+p17 — letter O not zero"),
    "faa_type_certificate":    ("3a13",    "FAA TCDS 3A13"),
    "alternator_output_amps":  ("60",      "POH equipment list p101"),
    "fuel_capacity_total_gal": ("56",      "POH p4 standard usable"),
    "fuel_capacity_left_gal":  ("30.5",    "POH p9 standard tank each"),
    "fuel_capacity_right_gal": ("30.5",    "POH p9 standard tank each"),
    "cruise_fuel_flow_gph":    ("11.0",    "POH p64 75% power"),
    "max_rpm":                 ("2400",    "POH p17 + Continental E-273"),
    "max_power_bhp":           ("230",     "POH p4"),
    "engine_count":            ("1",       "Single engine aircraft"),
    "vne_knots":               ("172",     "POH p16 KCAS"),
    "vs0_knots":               ("45",      "POH p17 white arc lower limit"),
    "service_ceiling_ft":      ("16,500",  "POH p4"),
    "max_gross_weight_lbs":    ("2950",    "POH p4+p18"),
    "system_voltage":          ("14",      "Service manual p248 12V battery→14V system"),
    "battery_capacity_ah":     ("33",      "Service manual D2006R4-13 p248"),
}

KNOWN_MISSING = {
    "min_fuel_pump_voltage":         "CRITICAL — not in POH or service manual",
    "alternator_failure_procedure":  "OCR split on p38 — not extracted",
    "engine_failure_procedure":      "Incomplete extraction p26",
    "fuel_system_failure_procedure": "Not cleanly extracted p33",
}

print("\n" + "="*65)
print("FAILURESCOPE — COMPLETE DATA AUDIT (CORRECTED)")
print("="*65)

# ── 1. VALIDATED JSON FILES ───────────────────────────────────
print("\n── VALIDATED JSON FILES ────────────────────────────────────────")
validated_files = sorted(VALIDATED_DIR.glob("*_validated.json"))

all_final_values = {}  # field_id → {doc_id, final_value, status}

for vf in validated_files:
    print(f"\n  [{vf.stem}]")
    data = json.loads(vf.read_text())
    for field_id, record in data.items():
        final_val = record.get("final_value")
        status    = record.get("status", "unknown")
        note      = record.get("correction_note", "")
        page      = record.get("page_number")
        doc_id    = record.get("doc_id", "")

        if status == "skipped":
            if field_id in KNOWN_MISSING:
                warn(f"{field_id} = SKIPPED — {KNOWN_MISSING[field_id]}")
            else:
                warn(f"{field_id} = SKIPPED — check if expected")
            continue

        if status == "rejected":
            warn(f"{field_id} = REJECTED")
            continue

        if status not in ("accepted", "corrected"):
            err(f"{field_id} unknown status: {status!r}")
            continue

        # Store for cross-checks
        all_final_values[field_id] = {
            "value": final_val, "doc_id": doc_id, "status": status
        }

        if field_id not in GROUND_TRUTH:
            ok(f"{field_id} = {final_val!r} [{status}] p{page}")
            continue

        expected, src = GROUND_TRUTH[field_id]
        actual_n   = str(final_val or "").strip().lower().replace(",","").replace(" ","")
        expected_n = str(expected).lower().replace(",","").replace(" ","")

        if actual_n == expected_n:
            ok(f"{field_id} = {final_val!r} ← CORRECT ({src})")
        elif field_id == "engine_model" and "0-470" in actual_n:
            flag(f"{field_id} = {final_val!r} — OCR ARTIFACT: zero '0' instead of letter 'O'")
            flag(f"    Correct value should be 'O-470-U' — fix before Layer 1")
        else:
            flag(f"{field_id} = {final_val!r} BUT expected {expected!r} ({src})")

        if note:
            print(f"      note: {note}")

# ── 2. PHYSICS CACHE ──────────────────────────────────────────
print("\n\n── PHYSICS CACHE (cache/physics_cache.json) ────────────────────")
cache_raw = json.loads(CACHE_PATH.read_text())
meta      = cache_raw.get("_metadata", {})
values    = cache_raw.get("values", {})

print(f"  Built at:  {meta.get('built_at','?')}")
print(f"  Status:    {meta.get('status','?')}")
print(f"  Total entries: {len(values)}\n")

NUMERIC_RANGES = {
    "system_voltage":         (10, 30),
    "alternator_output_amps": (30, 100),
    "battery_capacity_ah":    (10, 60),
    "fuel_capacity_total_gal":(20, 200),
    "fuel_capacity_left_gal": (10, 100),
    "fuel_capacity_right_gal":(10, 100),
    "cruise_fuel_flow_gph":   (5, 25),
    "max_rpm":                (1500, 3500),
    "max_power_bhp":          (100, 400),
    "engine_count":           (1, 4),
    "vne_knots":              (100, 300),
    "vs0_knots":              (30, 80),
    "service_ceiling_ft":     (10000, 35000),
    "max_gross_weight_lbs":   (1000, 10000),
}

by_layer = {}
for fid, rec in values.items():
    layer = rec.get("layer_use", "unknown")
    by_layer.setdefault(layer, []).append((fid, rec))

for layer in sorted(by_layer):
    print(f"  [{layer}]")
    for fid, rec in by_layer[layer]:
        val  = rec.get("value")
        unit = rec.get("unit", "")
        cite = rec.get("citation", {})
        src  = cite.get("document", "?")[:45]
        stat = rec.get("status", "?")

        if val is None:
            err(f"{fid} = NULL in cache")
            continue

        # OCR zero/O check
        if fid == "engine_model" and "0-470" in str(val):
            flag(f"{fid} = {val!r} — OCR ARTIFACT zero not letter O in cache")

        if fid in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[fid]
            try:
                num = float(str(val).replace(",",""))
                if lo <= num <= hi:
                    ok(f"{fid} = {val} {unit} [range OK {lo}–{hi}] | {src}")
                else:
                    flag(f"{fid} = {val} {unit} — OUT OF RANGE [{lo},{hi}]")
            except:
                err(f"{fid} = {val!r} — expected number")
        else:
            ok(f"{fid} = {val!r} | {src}")

# ── 3. CROSS-DOCUMENT CONSISTENCY ─────────────────────────────
print("\n\n── CROSS-DOCUMENT CONSISTENCY ──────────────────────────────────")

def get_num(field):
    v = values.get(field, {}).get("value")
    try: return float(str(v).replace(",",""))
    except: return None

# Fuel math
fl = get_num("fuel_capacity_left_gal")
fr = get_num("fuel_capacity_right_gal")
ft = get_num("fuel_capacity_total_gal")
if fl and fr and ft:
    if abs(fl - fr) < 0.1:
        ok(f"Fuel tanks symmetric: left={fl}, right={fr}")
    else:
        flag(f"Fuel tanks ASYMMETRIC: left={fl} ≠ right={fr}")
    computed = fl * 2
    if abs(computed - ft) < 2.0:
        ok(f"Fuel math consistent: {fl}×2={computed:.0f} ≈ total {ft:.0f} gal")
    else:
        flag(f"Fuel math: {fl}×2={computed:.0f} ≠ total={ft:.0f}")

# VNE > VS0
vne = get_num("vne_knots")
vs0 = get_num("vs0_knots")
if vne and vs0:
    if vne > vs0 * 2:
        ok(f"Speed envelope consistent: Vne={vne} >> VS0={vs0}")
    else:
        flag(f"Speed envelope suspicious: Vne={vne}, VS0={vs0}")

# System voltage reasonable
sv = get_num("system_voltage")
if sv:
    if sv in (14, 28):
        ok(f"System voltage {sv}V is a standard aircraft voltage")
    else:
        flag(f"System voltage {sv}V is non-standard (expected 14 or 28)")

# Battery runtime estimate
bat_ah = get_num("battery_capacity_ah")
alt_a  = get_num("alternator_output_amps")
if bat_ah and sv:
    load_amps = 15  # typical C182 electrical load
    runtime_min = (bat_ah / load_amps) * 60
    ok(f"Estimated battery runtime (alternator failed): {runtime_min:.0f} min at {load_amps}A load")

# ── 4. CHROMADB ────────────────────────────────────────────────
print("\n\n── CHROMADB SEARCH INDEX ───────────────────────────────────────")
try:
    import chromadb
    client = chromadb.PersistentClient(path=str(DATA_ROOT / "chromadb"))
    col = client.get_collection("failurescope_docs")
    ok(f"Collection 'failurescope_docs': {col.count()} chunks indexed")

    queries = [
        ("alternator failure", "electrical cascade procedure"),
        ("fuel pump voltage",  "fuel pump electrical specs"),
        ("service ceiling",    "performance limits"),
        ("battery",            "electrical system"),
    ]
    for q, purpose in queries:
        r = col.query(query_texts=[q], n_results=1)
        if r and r.get("documents") and r["documents"][0]:
            snippet = r["documents"][0][0][:100].replace("\n"," ")
            ok(f"'{q}' ({purpose}) → '{snippet}'")
        else:
            warn(f"'{q}' returned no results")
except Exception as e:
    warn(f"ChromaDB: {e}")

# ── 5. LAYER 1 READINESS ──────────────────────────────────────
print("\n\n── LAYER 1 READINESS CHECK ─────────────────────────────────────")
MUST_HAVE = {
    "system_voltage":         "14V system — alternator cascade threshold",
    "battery_capacity_ah":    "33Ah — battery runtime",
    "alternator_output_amps": "60A — normal charge rate",
    "max_rpm":                "2400 — engine limit",
    "vne_knots":              "172 — flight envelope",
    "vs0_knots":              "45 — stall speed",
    "fuel_capacity_left_gal": "30.5 — simulation initial condition",
    "fuel_capacity_right_gal":"30.5 — simulation initial condition",
    "max_gross_weight_lbs":   "2950 — weight limits",
    "cruise_fuel_flow_gph":   "11.0 — fuel depletion rate",
    "max_power_bhp":          "230 — thrust calculation",
    "engine_count":           "1 — single engine rules",
}
CRITICAL_MISSING = {
    "min_fuel_pump_voltage":  "CRITICAL — alternator→fuel_pump cascade threshold"
}

print("  REQUIRED VALUES:")
all_ok = True
for fid, note in MUST_HAVE.items():
    val = values.get(fid, {}).get("value")
    if val is not None:
        ok(f"  {fid} = {val} — {note}")
    else:
        err(f"  {fid} MISSING — {note}")
        all_ok = False

print("\n  CRITICAL MISSING:")
for fid, note in CRITICAL_MISSING.items():
    val = values.get(fid, {}).get("value")
    if val:
        ok(f"  {fid} = {val}")
    else:
        flag(f"  {fid} — {note}")
        flag(f"    Source: Continental O-470 overhaul manual, electrical section")
        flag(f"    Impact: Cannot simulate alternator→fuel_pump failure cascade")
        flag(f"    Workaround: Use 10.5V (standard 12V battery minimum) as conservative estimate")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*65)
print("AUDIT SUMMARY")
print("="*65)
if errors:
    print(f"\n  ERRORS ({len(errors)}) — must fix:")
    for e in errors: print(f"    {FAIL} {e}")
if flags:
    print(f"\n  FLAGS ({len(flags)}) — review:")
    for f in flags: print(f"    {FLAG} {f}")
if warnings:
    print(f"\n  WARNINGS ({len(warnings)}) — acceptable:")
    for w in warnings: print(f"    {WARN} {w}")
if not errors and not flags:
    print(f"\n  {PASS} CLEAN — data ready for Layer 1")
elif not errors:
    print(f"\n  {WARN} No blocking errors. {len(flags)} flags need review.")
else:
    print(f"\n  {FAIL} {len(errors)} errors block Layer 1")
print("="*65 + "\n")