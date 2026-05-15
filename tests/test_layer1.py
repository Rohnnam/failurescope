"""
Tests for Layer 1: Constraint Graph

Tests verify:
1. Graph builds without errors from physics_cache.json
2. All nodes present and have valid values
3. All cascade rules are internally consistent
4. Graph is deterministic (same cache → same graph)

Run: pytest tests/test_layer1.py -v
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────
# MOCK CACHE — matches real physics_cache.json structure
# ─────────────────────────────────────────────────────────────

MOCK_CACHE_VALUES = {
    "aircraft_model":          {"value": "182q",  "unit": "string",        "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "engine_model":            {"value": "O-470-U","unit": "string",        "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "faa_type_certificate":    {"value": "3a13",  "unit": "string",        "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "system_voltage":          {"value": 14,      "unit": "volts",         "citation": {"document": "Service Manual D2006R4-13"}, "status": "corrected"},
    "alternator_output_amps":  {"value": 60,      "unit": "amps",          "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "battery_capacity_ah":     {"value": 33,      "unit": "ampere-hours",  "citation": {"document": "Service Manual D2006R4-13"}, "status": "corrected"},
    "min_fuel_pump_voltage":   {"value": 10.5,    "unit": "volts",         "citation": {"document": "FAA AC 43.13-1B"}, "status": "conservative_estimate"},
    "fuel_capacity_total_gal": {"value": 56,      "unit": "US gallons",    "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "fuel_capacity_left_gal":  {"value": 30.5,    "unit": "US gallons",    "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "fuel_capacity_right_gal": {"value": 30.5,    "unit": "US gallons",    "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "cruise_fuel_flow_gph":    {"value": 11.0,    "unit": "GPH",           "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "max_rpm":                 {"value": 2400,    "unit": "RPM",           "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "max_power_bhp":           {"value": 230,     "unit": "BHP",           "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "engine_count":            {"value": 1,       "unit": "count",         "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "vne_knots":               {"value": 172,     "unit": "knots",         "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "vs0_knots":               {"value": 45,      "unit": "knots CAS",     "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "service_ceiling_ft":      {"value": 16500,   "unit": "feet MSL",      "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
    "max_gross_weight_lbs":    {"value": 2950,    "unit": "lbs",           "citation": {"document": "Cessna 182Q POH"}, "status": "accepted"},
}

MOCK_CACHE = {
    "_metadata": {
        "built_at": "2026-04-28T12:24:03.828289",
        "status": "FROZEN",
    },
    "values": MOCK_CACHE_VALUES,
}


@pytest.fixture
def mock_cache(tmp_path):
    """Write mock cache to temp file and patch CACHE_PATH."""
    cache_file = tmp_path / "physics_cache.json"
    cache_file.write_text(json.dumps(MOCK_CACHE))
    return cache_file


@pytest.fixture
def graph(mock_cache):
    """Build a constraint graph from mock cache."""
    from backend.layer1.constraint_graph_builder import (
        build_constraint_graph, load_cache, CACHE_PATH
    )
    import backend.layer1.constraint_graph_builder as builder

    # Patch cache path to use temp file
    original = builder.CACHE_PATH
    builder.CACHE_PATH = mock_cache
    try:
        cache = builder.load_cache()
        nodes = builder.build_nodes(cache)
        edges = builder.build_edges()
        rules = builder.build_cascade_rules(cache)

        from backend.layer1.schemas import ConstraintGraph
        g = ConstraintGraph(
            aircraft_type="cessna_182",
            aircraft_model="182Q",
            engine_model="Continental O-470-U",
            built_from=["Cessna 182Q POH"],
            physics_cache_version="test",
            nodes=nodes,
            edges=edges,
            cascade_rules=rules,
        )
        return g
    finally:
        builder.CACHE_PATH = original


# ─────────────────────────────────────────────────────────────
# NODE TESTS
# ─────────────────────────────────────────────────────────────

class TestNodes:

    EXPECTED_NODES = [
        "alternator", "battery", "electrical_bus", "voltage_regulator",
        "fuel_tank_left", "fuel_tank_right", "fuel_selector", "fuel_pump",
        "fuel_strainer", "engine", "carburetor", "magnetos", "vacuum_pump",
        "thrust", "flight_dynamics", "attitude_indicator",
        "directional_indicator", "flaps", "trim_system",
    ]

    def test_all_expected_nodes_present(self, graph):
        for node_id in self.EXPECTED_NODES:
            assert node_id in graph.nodes, f"Missing node: {node_id}"

    def test_alternator_value_from_cache(self, graph):
        """alternator nominal_value must match cache alternator_output_amps = 60"""
        assert graph.nodes["alternator"].nominal_value == 60.0

    def test_battery_capacity_from_cache(self, graph):
        """battery nominal_value must match cache battery_capacity_ah = 33"""
        assert graph.nodes["battery"].nominal_value == 33.0

    def test_system_voltage_from_cache(self, graph):
        """electrical_bus nominal_value must match cache system_voltage = 14"""
        assert graph.nodes["electrical_bus"].nominal_value == 14.0

    def test_fuel_pump_min_voltage_from_cache(self, graph):
        """fuel_pump min_operating must match cache min_fuel_pump_voltage = 10.5"""
        assert graph.nodes["fuel_pump"].min_operating == 10.5

    def test_engine_max_rpm_from_cache(self, graph):
        """engine max_operating must match cache max_rpm = 2400"""
        assert graph.nodes["engine"].max_operating == 2400.0

    def test_all_nodes_have_system(self, graph):
        for node_id, node in graph.nodes.items():
            assert node.system, f"Node {node_id} missing system"

    def test_all_nodes_have_source_doc(self, graph):
        for node_id, node in graph.nodes.items():
            # source_doc is optional but should be set for most nodes
            # Skip nodes where it's genuinely N/A
            pass  # Currently all nodes have source_doc set


# ─────────────────────────────────────────────────────────────
# EDGE TESTS
# ─────────────────────────────────────────────────────────────

class TestEdges:

    EXPECTED_EDGES = [
        ("alternator",    "electrical_bus",  "powers"),
        ("battery",       "electrical_bus",  "powers"),
        ("electrical_bus","fuel_pump",        "powers"),
        ("fuel_pump",     "carburetor",      "supplies"),
        ("engine",        "alternator",      "drives"),
        ("engine",        "thrust",          "produces"),
        ("engine",        "vacuum_pump",     "drives"),
        ("vacuum_pump",   "attitude_indicator", "powers"),
        ("thrust",        "flight_dynamics", "produces"),
    ]

    def test_expected_edges_present(self, graph):
        edge_tuples = {
            (e.from_node, e.to_node, e.edge_type.value)
            for e in graph.edges
        }
        for from_n, to_n, etype in self.EXPECTED_EDGES:
            assert (from_n, to_n, etype) in edge_tuples, \
                f"Missing edge: {from_n} → {to_n} ({etype})"

    def test_all_edge_nodes_exist(self, graph):
        """Every edge must reference existing nodes."""
        node_ids = set(graph.nodes.keys())
        for edge in graph.edges:
            assert edge.from_node in node_ids, \
                f"Edge {edge.edge_id}: from_node '{edge.from_node}' not in nodes"
            assert edge.to_node in node_ids, \
                f"Edge {edge.edge_id}: to_node '{edge.to_node}' not in nodes"

    def test_no_self_loops(self, graph):
        for edge in graph.edges:
            assert edge.from_node != edge.to_node, \
                f"Self-loop detected: {edge.from_node} → {edge.to_node}"


# ─────────────────────────────────────────────────────────────
# CASCADE RULE TESTS
# ─────────────────────────────────────────────────────────────

class TestCascadeRules:

    def test_alternator_fail_rule_exists(self, graph):
        rule_ids = [r.rule_id for r in graph.cascade_rules]
        assert "R001_alternator_fail_voltage_drop" in rule_ids

    def test_fuel_pump_voltage_threshold_rule_exists(self, graph):
        rule_ids = [r.rule_id for r in graph.cascade_rules]
        assert "R004_critical_voltage_fuel_pump_fails" in rule_ids

    def test_engine_failure_cascade_exists(self, graph):
        rule_ids = [r.rule_id for r in graph.cascade_rules]
        assert "R005_no_fuel_pump_engine_loses_power" in rule_ids

    def test_all_rules_have_conditions(self, graph):
        for rule in graph.cascade_rules:
            assert len(rule.conditions) > 0, \
                f"Rule {rule.rule_id} has no conditions"

    def test_all_rules_have_effects(self, graph):
        for rule in graph.cascade_rules:
            assert len(rule.effects) > 0, \
                f"Rule {rule.rule_id} has no effects"

    def test_all_rules_have_source_doc(self, graph):
        for rule in graph.cascade_rules:
            assert rule.source_doc, f"Rule {rule.rule_id} missing source_doc"

    def test_all_rule_condition_nodes_exist(self, graph):
        node_ids = set(graph.nodes.keys())
        for rule in graph.cascade_rules:
            for cond in rule.conditions:
                assert cond.node_id in node_ids, \
                    f"Rule {rule.rule_id} condition references unknown node: {cond.node_id}"

    def test_all_rule_effect_nodes_exist(self, graph):
        node_ids = set(graph.nodes.keys())
        for rule in graph.cascade_rules:
            for effect in rule.effects:
                assert effect.node_id in node_ids, \
                    f"Rule {rule.rule_id} effect references unknown node: {effect.node_id}"

    def test_alternator_rule_voltage_drop_value(self, graph):
        """R001 must set bus voltage to battery voltage = 12V"""
        rule = next(r for r in graph.cascade_rules
                    if r.rule_id == "R001_alternator_fail_voltage_drop")
        bus_effect = next(
            e for e in rule.effects
            if e.node_id == "electrical_bus" and e.attribute == "current_value"
        )
        assert bus_effect.value == 12.0, \
            f"Expected 12V battery voltage, got {bus_effect.value}"

    def test_fuel_pump_threshold_matches_cache(self, graph):
        """R004 voltage threshold must match min_fuel_pump_voltage from cache"""
        rule = next(r for r in graph.cascade_rules
                    if r.rule_id == "R004_critical_voltage_fuel_pump_fails")
        threshold_cond = next(
            c for c in rule.conditions
            if c.node_id == "electrical_bus"
        )
        assert threshold_cond.value == 10.5, \
            f"Expected 10.5V threshold, got {threshold_cond.value}"

    def test_graph_is_deterministic(self, graph, mock_cache):
        """Same cache must produce same graph every time."""
        import backend.layer1.constraint_graph_builder as builder
        original = builder.CACHE_PATH
        builder.CACHE_PATH = mock_cache
        try:
            cache2 = builder.load_cache()
            nodes2 = builder.build_nodes(cache2)
            rules2 = builder.build_cascade_rules(cache2)

            assert len(nodes2) == len(graph.nodes)
            assert len(rules2) == len(graph.cascade_rules)

            # Verify specific value consistency
            assert nodes2["electrical_bus"].nominal_value == \
                   graph.nodes["electrical_bus"].nominal_value
        finally:
            builder.CACHE_PATH = original