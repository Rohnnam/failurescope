"""
FailureScope Backend — FastAPI Application
==========================================
All 5 layers active.

Endpoints:
  GET  /health                  — system status (all layers)
  GET  /api/aircraft_list       — supported aircraft
  POST /api/parse_scenario      — Layer 0: NL text → ScenarioInput JSON
  POST /api/simulate            — Layer 1+2: ScenarioInput → trajectory + outcome
  POST /api/run_full            — L0+L1+L2+L3+L4: NL text → full analysis (one shot)
  POST /api/analyse             — L3+L4: trajectory + scenario → analysis only
  POST /api/validate            — L4 findings lookup by component name

Flow for production UI:
  POST /api/parse_scenario   → show confirmation screen
  POST /api/simulate         → show trajectory
  POST /api/analyse          → show L3+L4 analysis alongside trajectory

One-shot flow (testing / simple frontend):
  POST /api/run_full         → everything in one call
"""

import os
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.layer0.nlp_parser import NLPParser
from backend.layer0.schemas import ParseRequest, ParsedScenarioResponse, ScenarioInput
from backend.layer0.defaults import SUPPORTED_AIRCRAFT, AIRCRAFT_DEFAULTS
from backend.layer1.graph_loader import load_graph
from backend.layer2.simulator import Simulator
from backend.layer5 import inference as l3_inference
from backend.layer5 import findings_matcher as l4_findings


OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
DEBUG_MODE   = os.getenv("DEBUG", "false").lower() == "true"

# Module-level instances
nlp_parser: NLPParser = None
simulator:  Simulator  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp_parser, simulator

    # ── L0: NLP Parser ────────────────────────────────────────────────────────
    logger.info(f"Starting FailureScope — Ollama: {OLLAMA_HOST}, model: {OLLAMA_MODEL}")
    nlp_parser = NLPParser(model=OLLAMA_MODEL, host=OLLAMA_HOST)
    logger.info("  L0 NLP Parser initialised")

    # ── L1+L2: Simulator ─────────────────────────────────────────────────────
    simulator = Simulator(aircraft_type="cessna_182")
    logger.info("  L1+L2 Simulator initialised")

    # ── L3: ML Models ────────────────────────────────────────────────────────
    l3_ok = l3_inference.load_models()
    logger.info(f"  L3 models: {'loaded' if l3_ok else 'unavailable (L0-L2 unaffected)'}")

    # ── L4: Validated Findings ────────────────────────────────────────────────
    l4_ok = l4_findings.load_findings()
    logger.info(f"  L4 findings: {'loaded' if l4_ok else 'unavailable (L0-L2 unaffected)'}")

    logger.info("FailureScope ready.")
    yield
    logger.info("FailureScope shutting down.")


app = FastAPI(
    title="FailureScope API",
    description="Discovery Engine for Aircraft Failure Mode Analysis.",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    l3_ok = l3_inference._models_loaded
    l4_ok = l4_findings._findings_loaded
    return {
        "status":  "ok",
        "version": "0.5.0",
        "layers": {
            "layer0_nlp":        nlp_parser is not None,
            "layer1_graph":      simulator is not None,
            "layer2_simulator":  simulator is not None,
            "layer3_ml_models":  l3_ok,
            "layer4_findings":   l4_ok,
        },
        "full_pipeline_available": all([
            nlp_parser is not None,
            simulator is not None,
            l3_ok,
            l4_ok,
        ]),
    }


# ── Aircraft list (unchanged) ─────────────────────────────────────────────────

@app.get("/api/aircraft_list")
async def get_aircraft_list():
    return {"aircraft": [
        {"aircraft_type": k, "display_name": AIRCRAFT_DEFAULTS[k]["display_name"]}
        for k in SUPPORTED_AIRCRAFT
    ]}


# ── Layer 0: Parse scenario (unchanged) ──────────────────────────────────────

@app.post("/api/parse_scenario", response_model=ParsedScenarioResponse)
async def parse_scenario(request: ParseRequest):
    if nlp_parser is None:
        raise HTTPException(503, "NLP Parser not initialised.")
    try:
        return nlp_parser.parse(request.text)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Parse failed: {e}")


# ── Layer 1+2: Simulate (unchanged) ──────────────────────────────────────────

@app.post("/api/simulate")
async def simulate(scenario: ScenarioInput):
    if simulator is None:
        raise HTTPException(503, "Simulator not initialised.")

    sim_input = {
        "scenario_id":          f"api_{scenario.aircraft_type}",
        "initial_altitude_ft":  scenario.initial_state.altitude_ft,
        "initial_airspeed_kts": scenario.initial_state.airspeed_knots,
        "initial_fuel_gal":     (scenario.initial_state.fuel_left_gal +
                                  scenario.initial_state.fuel_right_gal),
        "pilot_skill":          scenario.pilot_state.skill_level,
        "pilot_fatigue":        scenario.pilot_state.fatigue,
        "weather":              getattr(scenario, "weather", "vmc"),
        "failures": [
            {"component": f.component,
             "trigger_time_s": f.trigger_time_seconds,
             "severity": f.severity}
            for f in scenario.failures
        ],
    }

    try:
        traj = simulator.run(sim_input)
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")

    return _format_simulation(traj, sim_input)


def _format_simulation(traj, sim_input: dict) -> dict:
    """Format TrajectoryRecord into API response dict (shared by simulate + run_full)."""
    return {
        "outcome":               traj.outcome.value,
        "time_to_criticality_s": traj.time_to_criticality_s,
        "recovery_possible":     traj.outcome.value == "recoverable",
        "failure_chain":         traj.failure_chain,
        "unrecoverable_reason":  traj.unrecoverable_reason,
        "duration_s":            traj.simulation_duration_s,
        "initial": {
            "altitude_ft":  sim_input["initial_altitude_ft"],
            "airspeed_kts": sim_input["initial_airspeed_kts"],
            "fatigue":      sim_input["pilot_fatigue"],
            "skill":        sim_input["pilot_skill"],
        },
        "state_snapshots": {
            "start":   traj.states[0]                    if traj.states else {},
            "quarter": traj.states[len(traj.states)//4]  if len(traj.states) > 3 else {},
            "half":    traj.states[len(traj.states)//2]  if len(traj.states) > 1 else {},
            "end":     traj.states[-1]                   if traj.states else {},
        },
    }


# ── Layer 3+4: Analyse (standalone endpoint for step-by-step UI flow) ────────

@app.post("/api/analyse")
async def analyse(scenario: ScenarioInput):
    """
    Run L3 ML analysis + L4 findings lookup on a scenario.
    Simulates internally, then runs L3+L4.
    Use this in the step-by-step flow after the user confirms the scenario.
    """
    if simulator is None:
        raise HTTPException(503, "Simulator not initialised.")

    sim_input = {
        "scenario_id":          f"analyse_{scenario.aircraft_type}",
        "initial_altitude_ft":  scenario.initial_state.altitude_ft,
        "initial_airspeed_kts": scenario.initial_state.airspeed_knots,
        "initial_fuel_gal":     (scenario.initial_state.fuel_left_gal +
                                  scenario.initial_state.fuel_right_gal),
        "pilot_skill":          scenario.pilot_state.skill_level,
        "pilot_fatigue":        scenario.pilot_state.fatigue,
        "weather":              getattr(scenario, "weather", "vmc"),
        "failures": [
            {"component": f.component,
             "trigger_time_s": f.trigger_time_seconds,
             "severity": f.severity}
            for f in scenario.failures
        ],
    }

    try:
        traj = simulator.run(sim_input)
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")

    l3_result = l3_inference.analyse(traj, scenario)
    component = l3_result.get("component", "unknown")
    l4_result = l4_findings.match(component)

    return {
        "simulation":  _format_simulation(traj, sim_input),
        "l3_analysis": l3_result,
        "l4_findings": l4_result,
    }


# ── L4 validate: findings lookup by component ─────────────────────────────────

@app.post("/api/validate")
async def validate(body: dict):
    """
    Look up validated findings for a failure component.
    Body: {"component": "alternator"}
    Returns the full validated finding from Layer 4.
    """
    component = body.get("component", "")
    if not component:
        raise HTTPException(400, "Request body must include 'component' field.")
    return l4_findings.match(component)


# ── run_full: NL text → full pipeline in one shot ────────────────────────────

@app.post("/api/run_full", summary="Natural language → full analysis (L0 → L4)")
async def run_full(request: ParseRequest):
    """
    One-shot endpoint: NL description → parse → simulate → L3 ML → L4 findings.

    Production flow (with confirmation screen):
      POST /api/parse_scenario → user confirms → POST /api/simulate → POST /api/analyse

    This endpoint skips the confirmation step — for testing and simple frontend use.
    """
    # ── L0: Parse ─────────────────────────────────────────────────────────────
    parsed = await parse_scenario(request)
    scenario = parsed.scenario

    # ── L1+L2: Simulate ───────────────────────────────────────────────────────
    sim_input = {
        "scenario_id":          "run_full",
        "initial_altitude_ft":  scenario.initial_state.altitude_ft,
        "initial_airspeed_kts": scenario.initial_state.airspeed_knots,
        "initial_fuel_gal":     (scenario.initial_state.fuel_left_gal +
                                  scenario.initial_state.fuel_right_gal),
        "pilot_skill":          scenario.pilot_state.skill_level,
        "pilot_fatigue":        scenario.pilot_state.fatigue,
        "weather":              getattr(scenario, "weather", "vmc"),
        "failures": [
            {"component": f.component,
             "trigger_time_s": f.trigger_time_seconds,
             "severity": f.severity}
            for f in scenario.failures
        ],
    }

    try:
        traj = simulator.run(sim_input)
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")

    sim_result = _format_simulation(traj, sim_input)

    # ── L3: ML Analysis ───────────────────────────────────────────────────────
    l3_result = l3_inference.analyse(traj, scenario)
    component = l3_result.get("component", "unknown")

    # ── L4: Validated Findings ────────────────────────────────────────────────
    l4_result = l4_findings.match(component)

    return {
        "input":       request.text,
        "understood":  parsed.summary,
        "assumptions": parsed.assumptions,
        "result": {
            **sim_result,
            "l3_analysis": l3_result,
            "l4_findings": l4_result,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000,
                reload=DEBUG_MODE, log_level="info")