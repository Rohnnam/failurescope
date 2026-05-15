# FailureScope

<div align="center">

**Discovery Engine for Aircraft Failure Cascade Analysis**

*ML-driven cascade discovery · POH/FAA sourced physics · NTSB validated*

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Hub-blue?style=flat-square&logo=docker)](https://hub.docker.com/r/rnrk/failurescope)

[**Docker Hub**](https://hub.docker.com/r/rnrk/failurescope) · [**Quick Start**](#quick-start) · [**Architecture**](#architecture)

</div>

---

## What It Does

FailureScope discovers how aircraft failures cascade — not just individually, but in combinations that aren't explicitly documented. You describe a scenario in plain English; the system runs a physics simulation, applies trained ML models, and surfaces the exact state combinations that make recovery impossible, backed by citations to POH sections and matching NTSB accident records.

**Example:** *"Cessna 182 at 5000ft, tired pilot, alternator fails"* →
The system identifies that the danger isn't the alternator failure itself — it's that the pilot's recovery maneuver increases electrical load at exactly the moment the fuel pump crosses its minimum voltage threshold, making the cascade unrecoverable. That interaction is not written anywhere. It emerges from the simulation data.

---

## Screenshots

### Results Dashboard
![Results Dashboard](docs/screenshots/results_dashboard.png)

### Cascade Timeline  
![Cascade Timeline](docs/screenshots/cascade_timeline.png)

### Validation & Evidence
![Validation Evidence](docs/screenshots/validation_evidence.png)

---

## Quick Start

```bash
# 1. Install Ollama and pull model
# https://ollama.com
ollama pull mistral:7b

# 2. Pull and run FailureScope (no other setup needed)
docker pull rnrk/failurescope:latest
docker run -p 8080:80 rnrk/failurescope:latest

# Linux only: add --add-host=host.docker.internal:host-gateway

# 3. Open http://localhost:8080
```

---

## System Requirements

| Requirement | Detail |
|---|---|
| **Ollama** | Running on host at `localhost:11434` with `mistral:7b` pulled |
| **Docker** | Docker Engine or Docker Desktop |
| **RAM** | 8GB minimum |
| **Disk** | ~3.5GB for Docker image |

---

## Architecture

```
Natural language input
        │
        ▼
┌───────────────────┐
│  Layer 0 · NLP    │  Ollama mistral:7b → structured scenario JSON
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Layer 1 · Graph  │  Constraint graph (14 components, POH/FAA rules)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Layer 2 · Sim    │  Deterministic physics at 0.1s resolution
└────────┬──────────┘  Same input → same output always.
         │
         ▼
┌───────────────────┐
│  Layer 3 · ML     │  XGBoost snapshot + Transformer cascade models
└────────┬──────────┘  Trained on 500,000 simulated trajectories
         │
         ▼
┌───────────────────┐
│  Layer 4 · Val    │  Physics soundness + NTSB accident matching
└────────┬──────────┘  21,761 Cessna records searched
         │
         ▼
    React Dashboard
```

**Why synthetic training data?** Real accident records are sparse and survival-biased. The physics simulator generates 500,000 complete, ground-truth-labeled trajectories covering the full failure space — including combinations that have never occurred. ML models trained on this data discover interaction patterns that aren't explicitly programmed into any rule.

---

## ML Architecture

### Dual-Model Design

**XGBoost Snapshot Model** — *"given aircraft state at failure injection, how likely is a crash?"*
- Input: 8 pre-existing condition features at failure moment
- Interpretability: SHAP values show which features drove the prediction

**Transformer Cascade Model** — *"given how the cascade is evolving, is recovery possible?"*
- Input: 400-step state sequence (40s at 0.1s resolution) from injection point
- Architecture: 4 layers, 8 attention heads
- Key design: failure one-hot encoding moved to context token — forces model to learn from state dynamics, not failure identity

---

## Validated Failure Modes (Cessna 182)

| Component | Class | Score | Status |
|---|---|---|---|
| Carburetor | INDUCED | 93.0 | ✅ VALIDATED |
| Fuel Selector | INDUCED | 90.0 | ✅ VALIDATED |
| Fuel Pump | INDUCED | 90.0 | ✅ VALIDATED |
| Attitude Indicator | NOVEL | 86.3 | ✅ VALIDATED |
| Alternator | INDUCED | 86.1 | ✅ VALIDATED |
| Voltage Regulator | INDUCED | 86.1 | ✅ VALIDATED |
| Battery | INDUCED | 83.0 | ✅ VALIDATED |
| Left Fuel Tank | NOVEL | 82.8 | ✅ VALIDATED |
| Right Fuel Tank | NOVEL | 82.8 | ✅ VALIDATED |
| Magnetos | NOVEL | 80.0 | 🟡 LIKELY REAL |
| Engine | INDUCED | 79.0 | 🟡 LIKELY REAL |
| Flaps | NOVEL | 73.0 | 🟡 LIKELY REAL |
| Trim System | NOVEL | 73.0 | 🟡 LIKELY REAL |
| Vacuum Pump | CONFIRMED | 53.3 | 🔵 POSSIBLE |

**INDUCED** — emerges from state variable interactions not explicitly programmed  
**NOVEL** — not covered by any L1 constraint rule  
**CONFIRMED** — matches known L1 rule (validation, not discovery)

---

## Data Sources

| Source | Purpose |
|---|---|
| Cessna 182Q POH §4.2, §5.1, §7.1, §7.3 | Constraint graph, cascade rules, physics constants |
| Continental O-470-U Manual | Fuel pump voltage specifications |
| FAA AC 43.13-1B §11 | Electrical system cascade thresholds |
| FAA AC 23.1309-1A | System safety analysis |
| MIL-SPEC-21030 | Battery internal resistance model |
| NTSB Aviation Accident Database | 21,761 Cessna records for validation |

---

## Limitations

- **Aircraft scope:** Cessna 182Q only. Architecture supports multi-aircraft expansion.
- **Physics fidelity:** Single-axis dynamics. No crosswind, no icing accretion model.
- **L3 inference:** Cascade Transformer accuracy reduced for slow cascades (>300s) due to training/inference resolution mismatch. L2 physics is always ground truth.
- **Scores are validation scores**, not real-world failure likelihood estimates.

---

## Test Results

```
23 edge case scenarios · 19 PASS · 4 FAIL · 0 ERRORS

4 documented limitations (not bugs):
  [04] Dual failure: L3 snapshot sees clean state before cascade fires
  [05] Battery injection: model reads low voltage, physics gives pilot recovery time  
  [16] Attitude indicator alone ≠ R014 (requires vacuum_pump fail + IMC)
  [21] Dual engine+alternator at 5000ft: commercial pilot has sufficient glide range
```

---

## Reproducing Training

```bash
# Models are pre-trained and included in the image.
# To reproduce from scratch on your own hardware:

conda activate aviate

python preprocess_cascade.py   # Generate 500K trajectories (~4 hrs, RTX 4060)
python train_layer3.py --all   # Train ML models (~3 hrs)
python train_layer4.py --all   # NTSB validation (~5 min)
```

---

## Project Structure

```
failurescope/
├── backend/
│   ├── app.py              FastAPI — all layer wiring
│   ├── layer0/             NLP parsing (Ollama mistral:7b)
│   ├── layer1/             Constraint graph (POH/FAA rules)
│   ├── layer2/             Physics simulator
│   ├── layer3/             XGBoost + Transformer models
│   ├── layer4/             NTSB validation
│   └── layer5/             Response assembly
├── frontend/src/App.jsx    React SPA (single file)
├── models/                 Pre-trained models
├── data/                   Constraint graph JSON
├── edge_case_tests.py      Test suite
├── Dockerfile
└── requirements.txt
```

---

## Related Work

- **IGLA** — Physics-inspired LLM safety classifier (submitted COLM workshops)
- **THMI** — Transformer hierarchical modeling (SSRN + Zenodo, workshop review)

---

## License

MIT

---

<div align="center">
<i>Built by Rohan Nambiar · BTech Year 2 · 2026</i>
</div>
