"""
FailureScope CLI — Layer 0→2 integration test tool.
Usage: python simulate.py
"""

import sys, json
sys.path.insert(0, '.')

from backend.layer0.nlp_parser import NLPParser
from backend.layer2.simulator import Simulator

BANNER = """
╔══════════════════════════════════════════════════════╗
║          FailureScope — Scenario Simulator           ║
║   Type a scenario in plain English. Type 'quit'.    ║
╚══════════════════════════════════════════════════════╝
"""

EXAMPLES = [
    "Cessna 182 at 5000 feet, new tired pilot, engine fails",
    "Cessna at cruise altitude, exhausted student, alternator failure",
    "C182 on approach at 1500 feet, fresh commercial pilot, fuel pump fails",
    "Cessna 182 in IMC at 8000 feet, tired pilot, vacuum pump fails",
]

def format_chain(chain: list) -> str:
    if not chain:
        return "  (no cascade events)"
    lines = []
    for i, event in enumerate(chain):
        t = event.get('time_s', '?')
        comp = str(event.get('component', '?'))
        sev  = str(event.get('severity', ''))
        # Strip enum prefixes: FailureComponent.ENGINE_LEFT → engine_left
        if '.' in comp: comp = comp.split('.')[-1].lower()
        if '.' in sev:  sev  = sev.split('.')[-1].lower()
        lines.append(f"  {'└─' if i > 0 else '►'} t={t}s → {comp} ({sev})")
    return '\n'.join(lines)

def format_state(label: str, s: dict) -> str:
    if not s: return ""
    return (f"  {label}: t={s.get('t','?')}s  "
            f"alt={s.get('alt','?'):.0f}ft  "
            f"v={s.get('v','?'):.1f}V  "
            f"rpm={s.get('rpm','?'):.0f}  "
            f"pump={s.get('pump_eff','?'):.2f}")

def print_result(text: str, parsed, traj):
    outcome = traj.outcome.value.upper()
    symbol = "💥 CATASTROPHIC" if outcome == "CATASTROPHIC" else "✓  RECOVERABLE"

    print(f"\n{'─'*54}")
    print(f"INPUT:     {text}")
    print(f"PARSED AS: {parsed.summary}")
    if parsed.assumptions:
        print(f"ASSUMED:   {'; '.join(parsed.assumptions[:3])}")
    print(f"{'─'*54}")
    print(f"OUTCOME:   {symbol}")
    if traj.time_to_criticality_s:
        mins = traj.time_to_criticality_s / 60
        print(f"CRASHED:   t={traj.time_to_criticality_s:.0f}s ({mins:.1f} min into flight)")
        print(f"REASON:    {traj.unrecoverable_reason}")
    else:
        dur = traj.simulation_duration_s
        dur_str = f"{dur:.0f}s" if dur < 120 else f"{dur/60:.1f} min"
        print(f"SURVIVED:  Full {dur_str} simulation — no crash")

    print(f"\nFAILURE CHAIN:")
    print(format_chain(traj.failure_chain))

    if traj.states:
        print(f"\nKEY STATES:")
        states = traj.states
        print(format_state("START  ", states[0]))
        if len(states) > 4:
            print(format_state("MID    ", states[len(states)//2]))
        print(format_state("END    ", states[-1]))
    print(f"{'─'*54}\n")


def main():
    print(BANNER)
    print("Initialising... (loading Ollama + constraint graph)\n")

    parser   = NLPParser(model="mistral:7b", host="http://localhost:11434")
    sim      = Simulator(aircraft_type="cessna_182")

    print("Ready. Example inputs:")
    for ex in EXAMPLES:
        print(f"  • {ex}")
    print()

    while True:
        try:
            text = input("Scenario> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue
        if text.lower() in ('quit', 'exit', 'q'):
            print("Exiting.")
            break

        print("Parsing...", end=" ", flush=True)
        try:
            parsed = parser.parse(text)
        except Exception as e:
            print(f"\nParse error: {e}")
            continue

        # Show any parse warnings (unmapped components, etc.)
        if parsed.parse_warnings:
            print(f"\n⚠  Warnings:")
            for w in parsed.parse_warnings:
                print(f"   {w}")

        print("Simulating...", end=" ", flush=True)
        scenario = {
            "scenario_id":          "cli_test",
            "initial_altitude_ft":  parsed.scenario.initial_state.altitude_ft,
            "initial_airspeed_kts": parsed.scenario.initial_state.airspeed_knots,
            "initial_fuel_gal":     (parsed.scenario.initial_state.fuel_left_gal +
                                     parsed.scenario.initial_state.fuel_right_gal),
            "pilot_skill":          parsed.scenario.pilot_state.skill_level,
            "pilot_fatigue":        parsed.scenario.pilot_state.fatigue,
            "weather":              parsed.scenario.weather,
            "failures": [
                {"component": f.component,
                 "trigger_time_s": f.trigger_time_seconds,
                 "severity": f.severity}
                for f in parsed.scenario.failures
            ],
        }

        try:
            traj = sim.run(scenario)
        except Exception as e:
            print(f"\nSimulation error: {e}")
            continue

        print("Done.")
        print_result(text, parsed, traj)


if __name__ == "__main__":
    main()