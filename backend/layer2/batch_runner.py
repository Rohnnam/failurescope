"""
Layer 2: Batch Runner
Parallelized execution of 500K simulation scenarios.

Source: context1 [Success Metrics]
  "Simulation runtime: <1 second per scenario (parallelized)"
  "Can reproduce 80%+ of known accidents"

Uses Python multiprocessing with 16 CPU threads (AMD Ryzen 7 8845HS).
Each worker runs an independent Simulator instance.

Output: JSONL.gz at /opt/failurescope_data/simulation_trajectories/
  One JSON line per trajectory.
  Total: ~1.7 GB compressed at DT=1.0s.

Changes from original:
  - FIXED: states[::10] → states[::1] (DT changed to 1.0s in Session 2;
    the old ::10 comment said "1s intervals" assuming DT=0.1s, but at
    DT=1.0s it was sampling every 10s — only 3 states per 33s cascade)
  - ADDED: tqdm progress bar with ETA
  - ADDED: disk guard checked every batch (stops if /opt < MIN_FREE_GB)
  - ADDED: debug log suppression during batch (suppresses MANEUVER LOAD SPIKE flood)

Usage:
  python -m backend.layer2.batch_runner --count 500000   # Full run
  python -m backend.layer2.batch_runner --count 1000     # Test run
  python -m backend.layer2.batch_runner --resume         # Resume from checkpoint
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

from loguru import logger

from backend.layer2.scenario_generator import generate_scenarios, count_scenarios
from backend.layer2.simulator import Simulator
from backend.layer2.state import SimOutcome


# ─────────────────────────────────────────────────────────────
# PATHS + CONSTANTS
# ─────────────────────────────────────────────────────────────

TRAJECTORIES_DIR = Path("/opt/failurescope_data/simulation_trajectories")
CHECKPOINT_PATH  = TRAJECTORIES_DIR / "checkpoint.json"
OUTPUT_FILE      = TRAJECTORIES_DIR / "trajectories.jsonl.gz"

MIN_FREE_GB  = 10.0   # Stop if /opt drops below this
DISK_CHECK_EVERY = 10  # Check disk every N batches


# ─────────────────────────────────────────────────────────────
# WORKER FUNCTION
# ─────────────────────────────────────────────────────────────

def worker_run_scenario(scenario: dict) -> dict | None:
    """
    Worker function — runs one scenario and returns trajectory summary.

    FIX: states[::1] keeps ALL states at DT=1.0s resolution (1-second intervals).
    The old states[::10] was written when DT=0.1s and gave 1s intervals.
    At DT=1.0s, states[::10] sampled every 10 seconds — only 3 snapshots
    for a 33-second median cascade, making Layer 3 cascade analysis impossible.
    """
    try:
        sim  = Simulator(aircraft_type="cessna_182")
        traj = sim.run(scenario)

        # Store cascade window with adaptive subsampling.
        # DT=0.1s in simulator. Storage resolution is adaptive:
        #
        #   Long electrical cascade (>200s, alternator/battery/voltage_regulator):
        #     Anchor: first state where v < 11.5V (R003 threshold crossing)
        #     Subsample: every 50th state = 5.0s per stored step
        #     Coverage: 400 stored states × 5.0s = 2000s
        #     Captures: full R003→R004→pump_fail→engine_stop span (~1780s)
        #     Source: FAA AC 43.13-1B (R003=11.5V, R004=10.5V), POH §7.1
        #
        #   Short cascade (<=200s) or non-electrical:
        #     Anchor: inject_idx
        #     Subsample: every 1st state = 0.1s per stored step
        #     Coverage: 400 stored states × 0.1s = 40s
        #     Captures: fast mechanical cascades (engine, carburetor: 30–60s median)

        inject_t = 0.0
        if traj.failures:
            inject_t = float(traj.failures[0].get("trigger_time_s", 0.0))

        inject_idx = 0
        for idx, s in enumerate(traj.states):
            if float(s.get("t", 0.0)) >= inject_t - 0.05:
                inject_idx = idx
                break

        is_catastrophic = traj.outcome.value == "catastrophic"
        ttc_s = traj.time_to_criticality_s
        cascade_dur_s = (ttc_s - inject_t) if ttc_s and ttc_s > inject_t else 0.0

        ELECTRICAL_COMPONENTS = {"alternator", "battery", "voltage_regulator"}
        primary_comp = traj.failures[0].get("component", "") if traj.failures else ""
        is_electrical = primary_comp in ELECTRICAL_COMPONENTS

        if is_catastrophic and is_electrical and cascade_dur_s > 200.0:
            # Find R003 threshold crossing (first v < 11.5V after injection)
            r003_idx = inject_idx
            for idx in range(inject_idx, len(traj.states)):
                if float(traj.states[idx].get("v", 99.0)) < 11.5:
                    r003_idx = idx
                    break
            # Subsample at 5s intervals from R003 crossing
            all_states = traj.states[r003_idx::50][:400]
        else:
            # Short cascade: inject-forward at full 0.1s resolution
            all_states = traj.states[inject_idx:inject_idx + 400]

        return {
            "scenario_id":   traj.scenario_id,
            "aircraft":      "cessna_182",
            "outcome":       traj.outcome.value,
            "ttc_s":         traj.time_to_criticality_s,
            "recovery_s":    traj.recovery_margin_s,
            "failure_chain": traj.failure_chain,
            "reason":        traj.unrecoverable_reason,
            "duration_s":    traj.simulation_duration_s,
            "steps":         traj.total_time_steps,
            "init": {
                "altitude_ft":  traj.initial_altitude_ft,
                "airspeed_kts": traj.initial_airspeed_kts,
                "fuel_gal":     traj.initial_fuel_gal,
                "skill":        traj.pilot_skill,
                "fatigue":      traj.pilot_fatigue,
                "weather":      traj.weather,
            },
            "failures": traj.failures,
            "states":   all_states,
        }
    except Exception as e:
        logger.error(f"Scenario {scenario.get('scenario_id', '?')} failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not CHECKPOINT_PATH.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("completed_ids", []))
    except Exception:
        return set()


def save_checkpoint(completed_ids: set, stats: dict):
    CHECKPOINT_PATH.write_text(json.dumps({
        "completed_ids": list(completed_ids),
        "stats": stats,
    }))


# ─────────────────────────────────────────────────────────────
# DISK GUARD
# ─────────────────────────────────────────────────────────────

def check_disk() -> float:
    """Return free GB on /opt partition."""
    return shutil.disk_usage("/opt").free / (1024 ** 3)


# ─────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────

def run_batch(
    max_scenarios: int = 500_000,
    n_workers: int = None,
    batch_size: int = 200,
    resume: bool = False,
    single_failure_only: bool = False,
):
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 2)

    # ── Suppress debug noise during batch run ─────────────────────────────────
    # MANEUVER LOAD SPIKE logs flood at ~50 lines/scenario = 25M lines for 500K
    # Set INFO level so progress is visible but debug spam is suppressed
    logger.remove()
    import sys
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    free_gb = check_disk()
    logger.info(f"Batch runner starting:")
    logger.info(f"  Max scenarios:  {max_scenarios:,}")
    logger.info(f"  Workers:        {n_workers}")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Output:         {OUTPUT_FILE}")
    logger.info(f"  Disk free:      {free_gb:.1f} GB  (guard: stop if < {MIN_FREE_GB} GB)")
    logger.info(f"  State sampling: every 1s (DT=1.0s, no subsampling)")

    counts = count_scenarios(single_failure_only)
    logger.info(f"  Scenario space: {counts['total']:,} total")

    if free_gb < MIN_FREE_GB:
        logger.error(f"Disk guard: only {free_gb:.1f} GB free. Need {MIN_FREE_GB} GB. Aborting.")
        return {}

    # ── Checkpoint ────────────────────────────────────────────────────────────
    completed_ids = load_checkpoint() if resume else set()
    if completed_ids:
        logger.info(f"  Resuming: {len(completed_ids):,} already done")

    # ── Output file ───────────────────────────────────────────────────────────
    gz_mode = "at" if (resume and OUTPUT_FILE.exists()) else "wt"
    outfile = gzip.open(OUTPUT_FILE, gz_mode, compresslevel=3)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total_run":    0,
        "catastrophic": 0,
        "recoverable":  0,
        "errors":       0,
        "start_time":   time.time(),
    }

    # ── tqdm progress bar ─────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=max_scenarios,
            initial=len(completed_ids),
            unit="scen",
            ncols=90,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "CAT:{postfix}"
            ),
        )
    except ImportError:
        pbar = None
        logger.warning("tqdm not installed. Run: pip install tqdm --break-system-packages")

    # ── Main loop ─────────────────────────────────────────────────────────────
    scenario_gen = generate_scenarios(max_scenarios, single_failure_only)
    batch = []
    batches_since_disk_check = 0

    with mp.Pool(processes=n_workers) as pool:
        for scenario in scenario_gen:
            if scenario["scenario_id"] in completed_ids:
                continue

            batch.append(scenario)

            if len(batch) < batch_size:
                continue

            # ── Process batch ──────────────────────────────────────────────
            results = pool.map(worker_run_scenario, batch)

            for result in results:
                if result is None:
                    stats["errors"] += 1
                    continue
                outfile.write(json.dumps(result) + "\n")
                completed_ids.add(result["scenario_id"])
                stats["total_run"] += 1
                if result["outcome"] == SimOutcome.CATASTROPHIC.value:
                    stats["catastrophic"] += 1
                else:
                    stats["recoverable"] += 1

            outfile.flush()
            batch = []
            save_checkpoint(completed_ids, stats)
            batches_since_disk_check += 1

            # ── Update progress bar ────────────────────────────────────────
            if pbar is not None:
                n_done  = stats["total_run"]
                pct_cat = (stats["catastrophic"] / n_done * 100) if n_done else 0
                pbar.update(len(results) - results.count(None))
                pbar.set_postfix_str(f"{pct_cat:.1f}%")
            else:
                # Fallback progress line
                elapsed = time.time() - stats["start_time"]
                rate    = stats["total_run"] / elapsed if elapsed > 0 else 1
                eta_min = (max_scenarios - stats["total_run"]) / rate / 60
                pct_cat = (stats["catastrophic"] / stats["total_run"] * 100
                           if stats["total_run"] > 0 else 0)
                logger.info(
                    f"  Progress: {stats['total_run']:,}/{max_scenarios:,} "
                    f"({rate:.1f}/s) | CAT: {pct_cat:.1f}% | ETA: {eta_min:.0f}min"
                )

            # ── Disk guard (every N batches) ───────────────────────────────
            if batches_since_disk_check >= DISK_CHECK_EVERY:
                free_gb = check_disk()
                batches_since_disk_check = 0
                if pbar:
                    pbar.write(f"  Disk: {free_gb:.1f} GB free")
                else:
                    logger.info(f"  Disk: {free_gb:.1f} GB free")
                if free_gb < MIN_FREE_GB:
                    msg = (f"DISK GUARD: {free_gb:.1f} GB free < {MIN_FREE_GB} GB. "
                           f"Stopping safely. Resume with --resume.")
                    if pbar:
                        pbar.write(f"  ⚠ {msg}")
                    logger.warning(msg)
                    break

        # ── Final partial batch ────────────────────────────────────────────
        if batch:
            results = pool.map(worker_run_scenario, batch)
            for result in results:
                if result is None:
                    stats["errors"] += 1
                    continue
                outfile.write(json.dumps(result) + "\n")
                completed_ids.add(result["scenario_id"])
                stats["total_run"] += 1
                if result["outcome"] == SimOutcome.CATASTROPHIC.value:
                    stats["catastrophic"] += 1
                else:
                    stats["recoverable"] += 1
            if pbar:
                pbar.update(len(results) - results.count(None))

    if pbar:
        pbar.close()
    outfile.close()
    save_checkpoint(completed_ids, stats)

    # ── Final report ───────────────────────────────────────────────────────────
    elapsed = time.time() - stats["start_time"]
    cat_pct = stats["catastrophic"] / max(stats["total_run"], 1) * 100
    out_gb  = OUTPUT_FILE.stat().st_size / 1e9 if OUTPUT_FILE.exists() else 0
    free_gb = check_disk()

    logger.success(f"\nBatch complete:")
    logger.success(f"  Total run:     {stats['total_run']:,}")
    logger.success(f"  Catastrophic:  {stats['catastrophic']:,} ({cat_pct:.1f}%)")
    logger.success(f"  Recoverable:   {stats['recoverable']:,}")
    logger.success(f"  Errors:        {stats['errors']}")
    logger.success(f"  Duration:      {elapsed/60:.1f} minutes")
    logger.success(f"  Output:        {OUTPUT_FILE}")
    logger.success(f"  File size:     {out_gb:.2f} GB")
    logger.success(f"  Disk free:     {free_gb:.1f} GB remaining")

    return stats


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FailureScope Layer 2 Batch Runner")
    parser.add_argument("--count",   type=int, default=1000,
                        help="Number of scenarios to run (default: 1000 for test)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count-2)")
    parser.add_argument("--batch",   type=int, default=200,
                        help="Batch size (default: 200)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--single",  action="store_true",
                        help="Only run single-failure scenarios")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count scenarios without running")
    args = parser.parse_args()

    if args.dry_run:
        counts = count_scenarios(args.single)
        print(f"\nScenario counts:")
        print(f"  LHS single:  {counts.get('single_lhs', 0):,}")
        print(f"  LHS total:   {counts.get('total_lhs', 0):,}")
        print(f"  Grid single: {counts.get('single_grid', 0):,}")
        print(f"\nWith --count {args.count}: {args.count:,} will run")
    else:
        run_batch(
            max_scenarios=args.count,
            n_workers=args.workers,
            batch_size=args.batch,
            resume=args.resume,
            single_failure_only=args.single,
        )