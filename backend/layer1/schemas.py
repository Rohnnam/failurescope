"""
Layer 1: Constraint Graph Schemas
Defines all data structures used in the constraint graph.

Source: context1 [Layer 1: Deterministic Simulation]
Source: context2 [Stage 3: Constraint Graph Lookup]

Hallucination Risk: NONE
  - Pure data structures, no logic, no learning
  - All values come from physics_cache.json
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class ComponentStatus(str, Enum):
    """Operational status of any aircraft component."""
    NORMAL       = "normal"
    DEGRADED     = "degraded"       # Functioning but reduced efficiency
    FAILED       = "failed"         # Complete failure
    UNRELIABLE   = "unreliable"     # Intermittent, unpredictable


class EdgeType(str, Enum):
    """Type of dependency relationship between components."""
    POWERS       = "powers"         # A provides electrical power to B
    SUPPLIES     = "supplies"       # A provides fuel/fluid to B
    DRIVES       = "drives"         # A mechanically drives B
    CONTROLS     = "controls"       # A provides control signal to B
    PRODUCES     = "produces"       # A produces output consumed by B
    MONITORS     = "monitors"       # A monitors state of B


class FailureSeverity(str, Enum):
    CATASTROPHIC = "catastrophic"   # Component fails completely, immediately
    DEGRADED     = "degraded"       # Component loses partial function
    INTERMITTENT = "intermittent"   # Component fails unpredictably


class CascadeType(str, Enum):
    """Type of cascade effect."""
    IMMEDIATE    = "immediate"      # Effect happens at same time step
    GRADUAL      = "gradual"        # Effect degrades over multiple steps
    THRESHOLD    = "threshold"      # Effect triggers when value crosses threshold


# ─────────────────────────────────────────────────────────────
# NODE SCHEMA
# ─────────────────────────────────────────────────────────────

class ComponentNode(BaseModel):
    """
    A single aircraft system component.
    Source: context2 [Stage 3: Nodes]
    """
    node_id:         str   = Field(..., description="Unique component identifier")
    display_name:    str   = Field(..., description="Human-readable name")
    system:          str   = Field(..., description="Parent system: electrical/fuel/engine/flight_control/aerodynamics")
    status:          ComponentStatus = Field(default=ComponentStatus.NORMAL)
    
    # Operating parameters (populated from physics_cache.json)
    nominal_value:   Optional[float] = Field(default=None, description="Normal operating value")
    current_value:   Optional[float] = Field(default=None, description="Current value in simulation")
    unit:            Optional[str]   = Field(default=None)
    min_operating:   Optional[float] = Field(default=None, description="Minimum for normal operation")
    max_operating:   Optional[float] = Field(default=None, description="Maximum for normal operation")
    
    # Citation — every value traces to a document
    source_doc:      Optional[str]   = Field(default=None)
    source_page:     Optional[int]   = Field(default=None)


# ─────────────────────────────────────────────────────────────
# EDGE SCHEMA
# ─────────────────────────────────────────────────────────────

class DependencyEdge(BaseModel):
    """
    A directed dependency between two components.
    Source: context2 [Stage 3: Edges]
    """
    edge_id:         str      = Field(..., description="Unique edge identifier")
    from_node:       str      = Field(..., description="Source component node_id")
    to_node:         str      = Field(..., description="Target component node_id")
    edge_type:       EdgeType = Field(..., description="Nature of dependency")
    description:     str      = Field(..., description="Human-readable description")
    
    # Transfer function: how does from_node state affect to_node?
    transfer_rate:   Optional[float] = Field(
        default=1.0,
        description="Fraction of from_node output that reaches to_node (0.0-1.0)"
    )
    source_doc:      Optional[str] = Field(default=None)


# ─────────────────────────────────────────────────────────────
# CASCADE RULE SCHEMA
# ─────────────────────────────────────────────────────────────

class CascadeCondition(BaseModel):
    """A single IF condition in a cascade rule."""
    node_id:     str           = Field(..., description="Component being checked")
    attribute:   str           = Field(..., description="Attribute to check: status/value")
    operator:    str           = Field(..., description="Comparison: eq/lt/gt/lte/gte")
    value:       object        = Field(..., description="Threshold value")


class CascadeEffect(BaseModel):
    """A single THEN effect in a cascade rule."""
    node_id:     str           = Field(..., description="Component being affected")
    attribute:   str           = Field(..., description="Attribute to modify")
    operation:   str           = Field(..., description="set/multiply/add/subtract")
    value:       object        = Field(..., description="New value or delta")
    cascade_type: CascadeType  = Field(default=CascadeType.IMMEDIATE)
    rate_per_second: Optional[float] = Field(
        default=None,
        description="For GRADUAL cascades: rate of change per second"
    )


class CascadeRule(BaseModel):
    """
    IF [conditions] THEN [effects]
    Source: context2 [Stage 3: Cascade rules hard-coded from POH]

    Every rule cites its source document.
    Hallucination risk: NONE — deterministic rule application only.
    """
    rule_id:     str                   = Field(..., description="Unique rule identifier")
    description: str                   = Field(..., description="Human-readable rule description")
    conditions:  list[CascadeCondition] = Field(..., description="All conditions must be true (AND logic)")
    effects:     list[CascadeEffect]   = Field(..., description="Effects applied when conditions met")
    
    # Citation
    source_doc:  str                   = Field(..., description="Source document for this rule")
    source_page: Optional[int]         = Field(default=None)
    source_section: Optional[str]      = Field(default=None)
    poh_quote:   Optional[str]         = Field(default=None, description="Relevant POH text")


# ─────────────────────────────────────────────────────────────
# COMPLETE CONSTRAINT GRAPH SCHEMA
# ─────────────────────────────────────────────────────────────

class ConstraintGraph(BaseModel):
    """
    Complete constraint graph for one aircraft type.
    This is the Layer 1 deliverable.
    Source: context1 [Week 1 deliverable: constraint_graph.json]
    """
    aircraft_type:   str                    = Field(..., description="e.g. cessna_182")
    aircraft_model:  str                    = Field(..., description="e.g. 182Q")
    engine_model:    str                    = Field(..., description="e.g. Continental O-470-U")
    built_from:      list[str]              = Field(..., description="Source documents used")
    physics_cache_version: str              = Field(..., description="physics_cache.json build timestamp")

    nodes:           dict[str, ComponentNode]  = Field(default_factory=dict)
    edges:           list[DependencyEdge]      = Field(default_factory=list)
    cascade_rules:   list[CascadeRule]         = Field(default_factory=list)

    # Validated spec values used (references physics_cache.json)
    spec_values:     dict[str, object]         = Field(
        default_factory=dict,
        description="Key validated values used in this graph, with citations"
    )