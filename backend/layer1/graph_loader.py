"""
Layer 1: Constraint Graph Loader
Loads constraint_graph.json for use by Layer 2 physics simulator.

This is the runtime interface — the simulator calls this,
not the builder.

Source: context1 [Layer 1: Why This Layer is Critical]
  "Deterministic: No randomness. Same input = same output always."
  "Constraint-respecting: All cascade rules enforced."
"""

import json
from pathlib import Path
from functools import lru_cache
from loguru import logger

from backend.layer1.schemas import ConstraintGraph, ComponentStatus

GRAPH_DIR = Path("/opt/failurescope_data/constraint_graphs")


@lru_cache(maxsize=8)
def load_graph(aircraft_type: str) -> ConstraintGraph:
    """
    Load constraint graph for an aircraft type.
    Cached after first load — same graph object reused across all simulations.

    Args:
        aircraft_type: e.g. "cessna_182"

    Returns:
        ConstraintGraph with all nodes, edges, cascade rules
    """
    path = GRAPH_DIR / f"{aircraft_type}_constraint_graph.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Constraint graph not found for '{aircraft_type}' at {path}. "
            f"Run: python -m backend.layer1.constraint_graph_builder"
        )

    raw = json.loads(path.read_text())
    graph = ConstraintGraph(**raw)

    logger.info(
        f"Loaded constraint graph: {aircraft_type} "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.cascade_rules)} rules)"
    )
    return graph


def get_cascade_rules_for_component(
    graph: ConstraintGraph,
    component_id: str
) -> list:
    """
    Get all cascade rules that are triggered by a specific component's state.
    Used by Layer 2 simulator at each time step.
    """
    return [
        rule for rule in graph.cascade_rules
        if any(c.node_id == component_id for c in rule.conditions)
    ]


def get_downstream_components(
    graph: ConstraintGraph,
    component_id: str
) -> list[str]:
    """
    Get all components that depend on this component.
    Traverses edges to find what gets affected when component_id fails.
    """
    return [
        edge.to_node for edge in graph.edges
        if edge.from_node == component_id
    ]


def get_available_aircraft() -> list[str]:
    """List all aircraft types with built constraint graphs."""
    if not GRAPH_DIR.exists():
        return []
    return [
        p.stem.replace("_constraint_graph", "")
        for p in GRAPH_DIR.glob("*_constraint_graph.json")
    ]