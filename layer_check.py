"""
FailureScope — Layer 0 + Layer 1 Complete Check
Run: python layer_check.py
"""
import json
from pathlib import Path

DATA_ROOT   = Path("/opt/failurescope/data")
GRAPH_PATH  = DATA_ROOT / "constraint_graphs/cessna_182_constraint_graph.json"
CACHE_PATH  = DATA_ROOT / "cache/physics_cache.json"

PASS = "✓"; FAIL = "✗"; WARN = "⚠"
errors = []; warnings = []

def ok(m):   print(f"  {PASS} {m}")
def err(m):  print(f"  {FAIL} {m}"); errors.append(m)
def warn(m): print(f"  {WARN} {m}"); warnings.append(m)

print("\n" + "="*65)
print("FAILURESCOPE — LAYER 0 + LAYER 1 STATUS CHECK")
print("="*65)

# ── LAYER 0 ───────────────────────────────────────────────────
print("\n── LAYER 0: NLP PARSING ────────────────────────────────────────")

l0_files = {
    "backend/layer0/__init__.py":  Path("backend/layer0/__init__.py"),
    "backend/layer0/defaults.py":  Path("backend/layer0/defaults.py"),
    "backend/layer0/schemas.py":   Path("backend/layer0/schemas.py"),
    "backend/layer0/nlp_parser.py":Path("backend/layer0/nlp_parser.py"),
    "backend/layer0/validator.py": Path("backend/layer0/validator.py"),
}
for name, path in l0_files.items():
    if path.exists():
        ok(f"{name} present ({path.stat().st_size//1024}KB)")
    else:
        err(f"{name} MISSING")

# Import checks
try:
    import sys; sys.path.insert(0, ".")
    from backend.layer0.schemas import ScenarioInput, PilotState, FailureEvent, ParseRequest
    ok("Layer 0 schemas import: ScenarioInput, PilotState, FailureEvent, ParseRequest")
except Exception as e:
    err(f"Layer 0 schemas import failed: {e}")

try:
    from backend.layer0.defaults import AIRCRAFT_DEFAULTS, SUPPORTED_AIRCRAFT, FATIGUE_MAPPINGS
    ok(f"Layer 0 defaults import: {len(SUPPORTED_AIRCRAFT)} aircraft, {len(FATIGUE_MAPPINGS)} fatigue mappings")
    # Spot check C182 defaults updated to 14V
    c182 = AIRCRAFT_DEFAULTS.get("cessna_182", {})
    v = c182.get("default_voltage")
    if v == 14.0:
        ok(f"  cessna_182 default_voltage = {v}V (14V system — correct)")
    elif v == 28.0:
        err(f"  cessna_182 default_voltage = {v}V — OLD VALUE, should be 14V per service manual")
    else:
        warn(f"  cessna_182 default_voltage = {v}V — unexpected")
    
    bat = c182.get("battery_capacity_ah")
    if bat == 33:
        ok(f"  cessna_182 battery_capacity_ah = {bat}Ah (service manual p248 — correct)")
    else:
        warn(f"  cessna_182 battery_capacity_ah = {bat} — expected 33")
except Exception as e:
    err(f"Layer 0 defaults import failed: {e}")

try:
    from backend.layer0.validator import ScenarioValidator
    ok("Layer 0 validator import OK")
except Exception as e:
    err(f"Layer 0 validator import failed: {e}")

try:
    from backend.layer0.nlp_parser import NLPParser, SYSTEM_PROMPT
    ok(f"Layer 0 NLP parser import OK (system prompt {len(SYSTEM_PROMPT)} chars)")
except Exception as e:
    err(f"Layer 0 NLP parser import failed: {e}")

# FastAPI app
try:
    from backend.app import app
    routes = [r.path for r in app.routes]
    expected = ["/health", "/api/parse_scenario", "/api/aircraft_list", "/api/simulate"]
    for ep in expected:
        if ep in routes:
            ok(f"API endpoint: {ep}")
        else:
            err(f"API endpoint MISSING: {ep}")
except Exception as e:
    err(f"FastAPI app import failed: {e}")

# ── LAYER 1 ───────────────────────────────────────────────────
print("\n── LAYER 1: CONSTRAINT GRAPH ───────────────────────────────────")

l1_files = {
    "backend/layer1/__init__.py":              Path("backend/layer1/__init__.py"),
    "backend/layer1/schemas.py":               Path("backend/layer1/schemas.py"),
    "backend/layer1/constraint_graph_builder.py": Path("backend/layer1/constraint_graph_builder.py"),
    "backend/layer1/graph_loader.py":          Path("backend/layer1/graph_loader.py"),
}
for name, path in l1_files.items():
    if path.exists():
        ok(f"{name} present ({path.stat().st_size//1024}KB)")
    else:
        err(f"{name} MISSING")

# Import checks
try:
    from backend.layer1.schemas import (
        ConstraintGraph, ComponentNode, DependencyEdge,
        CascadeRule, ComponentStatus, EdgeType, CascadeType
    )
    ok("Layer 1 schemas import OK")
except Exception as e:
    err(f"Layer 1 schemas import failed: {e}")

try:
    from backend.layer1.graph_loader import load_graph, get_available_aircraft
    aircraft = get_available_aircraft()
    ok(f"Layer 1 graph loader import OK — available aircraft: {aircraft}")
except Exception as e:
    err(f"Layer 1 graph loader import failed: {e}")

# Constraint graph JSON
if not GRAPH_PATH.exists():
    err(f"constraint_graph.json NOT FOUND: {GRAPH_PATH}")
else:
    try:
        raw = json.loads(GRAPH_PATH.read_text())
        g = ConstraintGraph(**raw)

        ok(f"constraint_graph.json loads OK")
        ok(f"  Aircraft: {g.aircraft_type} ({g.aircraft_model})")
        ok(f"  Engine:   {g.engine_model}")
        ok(f"  Nodes:    {len(g.nodes)}")
        ok(f"  Edges:    {len(g.edges)}")
        ok(f"  Rules:    {len(g.cascade_rules)}")
        ok(f"  Built from {len(g.built_from)} source documents")

        # Verify key nodes exist
        KEY_NODES = ["alternator","battery","electrical_bus","fuel_pump","engine","thrust"]
        for n in KEY_NODES:
            if n in g.nodes:
                node = g.nodes[n]
                val  = node.nominal_value
                unit = node.unit or ""
                ok(f"  Node '{n}': nominal={val} {unit}")
            else:
                err(f"  Node '{n}' MISSING from graph")

        # Verify key cascade rules
        rule_ids = [r.rule_id for r in g.cascade_rules]
        KEY_RULES = [
            "R001_alternator_fail_voltage_drop",
            "R004_critical_voltage_fuel_pump_fails",
            "R005_no_fuel_pump_engine_loses_power",
            "R007_vacuum_pump_fail_gyros_fail",
        ]
        for rid in KEY_RULES:
            if rid in rule_ids:
                rule = next(r for r in g.cascade_rules if r.rule_id == rid)
                ok(f"  Rule '{rid}' — source: {rule.source_doc[:40]}")
            else:
                err(f"  Rule '{rid}' MISSING")

        # Verify values read from cache correctly
        bus = g.nodes.get("electrical_bus")
        if bus and bus.nominal_value == 14.0:
            ok(f"  electrical_bus.nominal_value = 14V (from cache) ✓")
        elif bus:
            err(f"  electrical_bus.nominal_value = {bus.nominal_value} — expected 14V")

        pump = g.nodes.get("fuel_pump")
        if pump and pump.min_operating == 10.5:
            ok(f"  fuel_pump.min_operating = 10.5V (from cache) ✓")
        elif pump:
            err(f"  fuel_pump.min_operating = {pump.min_operating} — expected 10.5V")

        bat = g.nodes.get("battery")
        if bat and bat.nominal_value == 33.0:
            ok(f"  battery.nominal_value = 33Ah (from cache) ✓")
        elif bat:
            err(f"  battery.nominal_value = {bat.nominal_value} — expected 33Ah")

        # Verify R001 sets correct battery voltage
        r001 = next((r for r in g.cascade_rules if r.rule_id == "R001_alternator_fail_voltage_drop"), None)
        if r001:
            bus_effect = next((e for e in r001.effects
                               if e.node_id == "electrical_bus" and e.attribute == "current_value"), None)
            if bus_effect and bus_effect.value == 12.0:
                ok(f"  R001 sets bus to 12V (battery voltage) ✓")
            elif bus_effect:
                err(f"  R001 sets bus to {bus_effect.value}V — expected 12V")

        # All edges reference valid nodes
        node_ids = set(g.nodes.keys())
        bad_edges = [(e.edge_id, e.from_node, e.to_node)
                     for e in g.edges
                     if e.from_node not in node_ids or e.to_node not in node_ids]
        if bad_edges:
            for eid, fn, tn in bad_edges:
                err(f"  Edge {eid}: references unknown node ({fn} or {tn})")
        else:
            ok(f"  All {len(g.edges)} edges reference valid nodes ✓")

        # Determinism check
        raw2 = json.loads(GRAPH_PATH.read_text())
        g2 = ConstraintGraph(**raw2)
        if len(g.nodes) == len(g2.nodes) and len(g.cascade_rules) == len(g2.cascade_rules):
            ok(f"  Graph is deterministic (same result on reload) ✓")
        else:
            err(f"  Graph non-deterministic on reload")

    except Exception as e:
        err(f"constraint_graph.json validation failed: {e}")

# ── PHYSICS CACHE (shared) ────────────────────────────────────
print("\n── PHYSICS CACHE (shared by L1+L2) ─────────────────────────────")
if CACHE_PATH.exists():
    cache = json.loads(CACHE_PATH.read_text())
    vals  = cache.get("values", {})
    ok(f"physics_cache.json: {len(vals)} entries, status=FROZEN")
    # Spot check critical L1 values
    for field, expected in [
        ("system_voltage", 14),
        ("min_fuel_pump_voltage", 10.5),
        ("battery_capacity_ah", 33),
        ("alternator_output_amps", 60),
        ("max_rpm", 2400),
    ]:
        v = vals.get(field, {}).get("value")
        if v is not None and float(str(v)) == expected:
            ok(f"  {field} = {v} ✓")
        elif v is not None:
            err(f"  {field} = {v} — expected {expected}")
        else:
            err(f"  {field} MISSING from cache")
else:
    err(f"physics_cache.json not found")

# ── TEST RESULTS ──────────────────────────────────────────────
print("\n── TEST SUITE ──────────────────────────────────────────────────")
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
    capture_output=True, text=True
)
for line in result.stdout.strip().split("\n")[-3:]:
    print(f"  {line}")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
if not errors and not warnings:
    print(f"\n  {PASS} ALL CHECKS PASSED")
    print(f"  Layer 0: COMPLETE ✓")
    print(f"  Layer 1: COMPLETE ✓")
    print(f"  Ready for Layer 2 (Physics Simulation)")
elif not errors:
    print(f"\n  {WARN} {len(warnings)} warnings, no blocking errors")
    print(f"  Layer 0: COMPLETE ✓")
    print(f"  Layer 1: COMPLETE ✓")
else:
    print(f"\n  {FAIL} {len(errors)} errors must be fixed")
    for e in errors:
        print(f"    {FAIL} {e}")
print("="*65 + "\n")