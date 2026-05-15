# FailureScope

**Discovery Engine for Aircraft Failure Cascade Analysis**  
*ML-driven cascade discovery · POH/FAA sourced physics · NTSB validated*

---

## What It Does

FailureScope discovers how aircraft failures cascade — not just individually, but in combinations that aren't explicitly documented. You describe a scenario in plain English; the system runs a physics simulation, applies trained ML models, and surfaces the exact state combinations that make recovery impossible, backed by citations to POH sections and matching NTSB accident records.

**Example:** *"Cessna 182 at 5000ft, tired pilot, alternator fails"* →  
The system identifies that the danger isn't the alternator failure itself — it's that the pilot's recovery maneuver increases electrical load at exactly the moment the fuel pump crosses its minimum voltage threshold, making the cascade unrecoverable. That interaction is not written anywhere. It emerges from the simulation data.

---

## System Requirements

| Requirement | Detail |
|---|---|
| **Ollama** | Must be running on host at `localhost:11434` with `mistral:7b` pulled |
| **Docker** | Docker Engine + Docker Compose |
| **RAM** | 8GB minimum (models load ~2GB) |
| **Disk** | ~500MB for container + ~25MB runtime data |

**Ollama setup (required before starting container):**
```bash
# Install Ollama: https://ollama.com
ollama pull mistral:7b
ollama serve   # must be running when container starts
```

---

## Quick Start

```bash
git clone https://github.com/Rohnnam/failurescope
cd failurescope

# Start (Ollama must be running on host)
docker compose up --build

# Open browser
http://localhost:8080
```

The container mounts `/opt/failurescope_data` read-only — pre-trained models are loaded at startup, no training happens at runtime.

---

## Architecture

```
Natural language input
        │
        ▼
┌───────────────┐
│ Layer 0 · NLP │  Ollama mistral:7b parses scenario → structured JSON
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Layer 1 · L1  │  Constraint graph lookup (POH/FAA rules, 14 components)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Layer 2 · Sim │  Deterministic physics simulation (POH-sourced equations)
└───────┬───────┘  Same input → same output always. No randomness.
        │
        ▼
┌───────────────┐
│ Layer 3 · ML  │  XGBoost (pre-existing conditions) + Transformer (cascade)
└───────┬───────┘  Trained on 500K simulated trajectories
        │
        ▼
┌───────────────┐
│ Layer 4 · Val │  Physics validation + NTSB accident matching
└───────┬───────┘  21,761 Cessna accident records searched
        │
        ▼
  Results dashboard
```

**Why this architecture:** The physics simulator generates ground-truth training data — 500,000 complete trajectories with known outcomes. The ML models learn from this complete, deterministic dataset rather than sparse real-world accident data. Discovered patterns are validated against both physics soundness and historical accident correspondence before being surfaced to the user.

---

## Validated Failure Modes (Cessna 182)

14 components analyzed. Combined physics + NTSB validation scores:

| Component | Class | Score | Status |
|---|---|---|---|
| Carburetor | INDUCED | 93.0 | VALIDATED |
| Fuel Selector | INDUCED | 90.0 | VALIDATED |
| Fuel Pump | INDUCED | 90.0 | VALIDATED |
| Attitude Indicator | NOVEL | 86.3 | VALIDATED |
| Alternator | INDUCED | 86.1 | VALIDATED |
| Voltage Regulator | INDUCED | 86.1 | VALIDATED |
| Battery | INDUCED | 83.0 | VALIDATED |
| Left Fuel Tank | NOVEL | 82.8 | VALIDATED |
| Right Fuel Tank | NOVEL | 82.8 | VALIDATED |
| Magnetos | NOVEL | 80.0 | LIKELY REAL |
| Engine | INDUCED | 79.0 | LIKELY REAL |
| Flaps | NOVEL | 73.0 | LIKELY REAL |
| Trim System | NOVEL | 73.0 | LIKELY REAL |
| Vacuum Pump | CONFIRMED | 53.3 | POSSIBLE |

**INDUCED** = relationship emerges from state variable interactions not explicitly programmed  
**NOVEL** = not covered by any L1 constraint rule  
**CONFIRMED** = matches known L1 rule (validation, not discovery)

---

## Data Sources

| Source | Purpose | Citation |
|---|---|---|
| Cessna 182Q POH | Constraint graph, cascade rules, physics constants | §4.2, §5.1, §7.1, §7.3 |
| Continental O-470-U Manual | Engine/fuel pump specifications | Fuel pump voltage requirements |
| FAA AC 43.13-1B | Electrical system cascade thresholds | §11 |
| FAA AC 23.1309-1A | System safety analysis (flaps/trim) | — |
| FAA-H-8083-25C / 3C | Aeronautical knowledge baseline | — |
| MIL-SPEC-21030 | Battery internal resistance model | Ohm's law voltage sag |
| Cessna Service Manual D2006R4-13 | Battery capacity (33Ah) | — |
| NTSB Aviation Accident Database | Historical validation | 21,761 Cessna records |

---

## Limitations

- **Aircraft scope:** Cessna 182Q only. Architecture supports multi-aircraft expansion.
- **Physics fidelity:** Single-axis dynamics model. No crosswind, no icing accretion model, no structural failure.
- **L3 inference:** Cascade Transformer has reduced accuracy for slow cascades (>300s) due to inference resolution mismatch between training (5s steps) and inference (0.1s steps). L2 physics is always ground truth.
- **L3/L2 disagreement:** Two known cases where XGBoost and physics disagree — battery injection (model sees low voltage, physics gives pilot time to recover on alternator) and dual simultaneous failure (snapshot model has no context for which failure was injected vs naturally cascaded). Both are documented limitations.
- **Probability claims:** The system makes no probability claims about real-world failure rates. All scores are validation scores (physics soundness + historical fit), not likelihood estimates.

---

## Tech Stack

**Backend:** Python 3.11 · FastAPI · PyTorch · XGBoost · SHAP · Ollama  
**Frontend:** React 18 · Vite · IBM Plex Mono · Barlow Condensed  
**Infrastructure:** Docker · Nginx · Uvicorn  
**ML:** Custom Transformer (4 layers, 8 heads) · XGBoost with SHAP interpretability  
**Data:** 500K simulated trajectories · 3.38GB training dataset

---

## Project Structure

```
failurescope/
├── backend/
│   ├── app.py              FastAPI application, all layer wiring
│   ├── layer0/             NLP parsing (Ollama mistral:7b)
│   ├── layer1/             Constraint graph (POH rules)
│   ├── layer2/             Physics simulator
│   ├── layer3/             ML models (XGBoost + Transformer)
│   ├── layer4/             NTSB validation
│   └── layer5/             Response assembly
├── frontend/
│   ├── src/App.jsx         Single-file React SPA
│   └── docker/             Nginx config + entrypoint
├── train_layer3.py         Training orchestrator
├── train_layer4.py         Validation orchestrator
├── edge_case_tests.py      23-scenario test suite (19/23 pass)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Reproducing the Training

```bash
# Not needed for deployment — models are pre-trained.
# For reference only:

conda activate aviate
cd failurescope

# Generate trajectories (~4 hrs, RTX 4060)
python preprocess_cascade.py

# Train L3 models (~3 hrs)
python train_layer3.py --all

# Run L4 validation (~5 min)
python train_layer4.py --all
```

---

## Related Work

- IGLA: Physics-inspired LLM safety classifier (submitted COLM workshops)
- THMI: Transformer hierarchical modeling (SSRN + Zenodo, workshop review)

---

*Built by Rohan Nambiar · BTech Year 2 · 2026*