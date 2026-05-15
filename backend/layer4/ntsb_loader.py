"""
FailureScope Layer 4 — NTSB Data Loader
=========================================
Reads the NTSB avall.mdb Access database via mdbtools (mdb-export).

Source file: /opt/failurescope_data/avall.mdb  (523 MB, already extracted)
Download:    avall.zip (~90 MB) from data.ntsb.gov/avdata/FileDirectory/

Pipeline:
  1. ensure_mdb()        — check MDB present, extract from ZIP if needed,
                           download ZIP if needed. Disk guard on /opt.
  2. _export_cessna()    — join events + aircraft + narratives via mdb-export
                           subprocess, filter to Cessna 182, write a small
                           Cessna-only CSV (~few MB). Runs once, reused.
  3. _parse_csv()        — keyword-filter the Cessna CSV per component.
  4. Cache               — per-component JSON cache avoids re-parsing.

Disk budget on /opt:
  avall.zip:               90 MB  (kept — source ZIP)
  avall.mdb:              523 MB  (extracted — source DB)
  ntsb_aviation_data.csv: ~10 MB  (Cessna-filtered output, reused across runs)
  ntsb_cache_<comp>.json:  ~10 KB each  (per-component keyword cache)

Table join:
  aircraft  →  ev_id, Aircraft_Key, acft_make, acft_model, phase_flt_spec
  narratives → ev_id, Aircraft_Key, narr_accp (description), narr_cause
  events    →  ev_id, ev_date, ev_highest_injury, inj_tot_f, wx_cond_basic

mdbtools confirmed installed: mdb-export v1.0.1 (Ubuntu 26.04 resolute)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.layer3.constants import (
    NTSB_BULK_CSV_URL,
    NTSB_CACHE_PATH,
    NTSB_CSV_PATH,
    NTSB_MDB_PATH,
    NTSB_MAX_RESULTS,
)

log = logging.getLogger(__name__)

NTSB_ZIP_PATH  = str(Path(NTSB_MDB_PATH).parent / "ntsb_avall.zip")
MIN_FREE_BYTES = 10 * 1024 ** 3   # 10 GB disk guard on /opt


# ── Disk guard ────────────────────────────────────────────────────────────────

def _check_disk(path: str = "/opt", required: int = MIN_FREE_BYTES) -> bool:
    try:
        stat    = shutil.disk_usage(path)
        free_gb = stat.free  / 1024 ** 3
        tot_gb  = stat.total / 1024 ** 3
        log.info(f"  Disk ({path}): {free_gb:.1f} GB free / {tot_gb:.1f} GB total")
        if stat.free < required:
            log.error(
                f"  DISK GUARD: {free_gb:.1f} GB free, need >{required/1024**3:.0f} GB. "
                "Aborting."
            )
            return False
        return True
    except Exception as e:
        log.warning(f"  Disk check failed: {e} — proceeding")
        return True


# ── mdb-export subprocess helper ─────────────────────────────────────────────

def _mdb_export(mdb_path: str, table: str) -> Iterator[dict]:
    """
    Stream rows from an MDB table via mdb-export.
    Yields dicts keyed by column name.
    mdb-export outputs comma-delimited CSV with a header row.
    """
    try:
        proc = subprocess.Popen(
            ["mdb-export", mdb_path, table],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        reader = csv.DictReader(io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"))
        for row in reader:
            yield row
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode(errors="replace")[:200]
            log.warning(f"  mdb-export {table} exited {proc.returncode}: {err}")
    except FileNotFoundError:
        log.error(
            "  mdb-export not found. Install with:\n"
            "    sudo apt install mdbtools"
        )
    except Exception as e:
        log.error(f"  mdb-export {table} error: {e}")


# ── Core join: events + aircraft + narratives → Cessna CSV ───────────────────

def _export_cessna(mdb_path: str, csv_out: str) -> bool:
    """
    Join the three relevant MDB tables, filter to Cessna aircraft,
    write a small normalised CSV to csv_out.

    Returns True on success.

    Column mapping to normalised schema used by ntsb_matcher.py:
      ev_id          -> accident_id
      ev_date        -> event_date
      acft_make      -> aircraft_make
      acft_model     -> aircraft_model
      narr_accp      -> narrative
      narr_cause     -> probable_cause
      ev_highest_injury -> injury_severity
      inj_tot_f      -> total_fatalities
      wx_cond_basic  -> weather_condition
      phase_flt_spec -> phase_of_flight
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    log.info("  Building Cessna-filtered CSV from MDB (3-table join)")
    log.info("  Step 1/3: Loading aircraft table (filter Cessna)")

    # ── Step 1: aircraft table → {ev_id: {Aircraft_Key, make, model, phase}} ─
    aircraft: dict[str, dict] = {}
    ac_total = 0
    for row in _mdb_export(mdb_path, "aircraft"):
        ac_total += 1
        make = (row.get("acft_make") or "").upper()
        if "CESSNA" not in make:
            continue
        ev_id = row.get("ev_id", "")
        if not ev_id:
            continue
        # Keep all Cessna models (182 and others — keyword filter handles specifics)
        aircraft[ev_id] = {
            "Aircraft_Key":   row.get("Aircraft_Key", ""),
            "aircraft_make":  row.get("acft_make", ""),
            "aircraft_model": row.get("acft_model", ""),
            "phase_of_flight":row.get("phase_flt_spec", ""),
            "aircraft_damage":row.get("damage", ""),
        }

    log.info(f"  Aircraft table: {ac_total} total rows, {len(aircraft)} Cessna records")

    if not aircraft:
        log.error("  No Cessna records found in aircraft table — MDB may be corrupt")
        return False

    # ── Step 2: narratives table → {ev_id: {narrative, probable_cause}} ──────
    log.info("  Step 2/3: Loading narratives table")
    narratives: dict[str, dict] = {}
    narr_total = 0

    narr_iter = _mdb_export(mdb_path, "narratives")
    if tqdm:
        narr_iter = tqdm(
            narr_iter,
            desc="  Narratives",
            unit=" rows",
            bar_format="{l_bar}{bar}| {n:,} rows [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        )

    for row in narr_iter:
        narr_total += 1
        ev_id = row.get("ev_id", "")
        if ev_id not in aircraft:
            continue
        # narr_accp = accident description, narr_cause = probable cause
        existing = narratives.get(ev_id, {})
        narrative  = (existing.get("narrative", "")  + " " + (row.get("narr_accp") or "")).strip()
        prob_cause = (existing.get("probable_cause","") + " " + (row.get("narr_cause") or "")).strip()
        narratives[ev_id] = {
            "narrative":      narrative,
            "probable_cause": prob_cause,
        }

    log.info(f"  Narratives table: {narr_total} total rows, {len(narratives)} matched Cessna")

    # ── Step 3: events table → join everything, write CSV ────────────────────
    log.info("  Step 3/3: Loading events table + writing Cessna CSV")
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "accident_id", "event_date", "aircraft_make", "aircraft_model",
        "narrative", "probable_cause", "injury_severity", "total_fatalities",
        "weather_condition", "phase_of_flight", "aircraft_damage",
    ]

    written = 0
    ev_iter = _mdb_export(mdb_path, "events")
    if tqdm:
        ev_iter = tqdm(
            ev_iter,
            desc="  Events",
            unit=" rows",
            bar_format="{l_bar}{bar}| {n:,} rows [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in ev_iter:
            ev_id = row.get("ev_id", "")
            if ev_id not in aircraft:
                continue

            ac  = aircraft[ev_id]
            nar = narratives.get(ev_id, {"narrative": "", "probable_cause": ""})

            writer.writerow({
                "accident_id":     ev_id,
                "event_date":      row.get("ev_date", ""),
                "aircraft_make":   ac["aircraft_make"],
                "aircraft_model":  ac["aircraft_model"],
                "narrative":       nar["narrative"],
                "probable_cause":  nar["probable_cause"],
                "injury_severity": row.get("ev_highest_injury", ""),
                "total_fatalities":row.get("inj_tot_f", ""),
                "weather_condition":row.get("wx_cond_basic", ""),
                "phase_of_flight": ac["phase_of_flight"],
                "aircraft_damage": ac["aircraft_damage"],
            })
            written += 1

    size_kb = os.path.getsize(csv_out) / 1024
    log.info(f"  Cessna CSV written: {written} records → {csv_out} ({size_kb:.0f} KB)")
    return written > 0


# ── CSV keyword parser (same as before, now on small Cessna CSV) ─────────────

def _parse_csv(csv_path: str, keywords: list[str]) -> list[dict]:
    """
    Keyword-filter the Cessna-only CSV. Fast — file is small (~21 MB, ~7K rows).

    Uses row-count tqdm (not byte-count) because csv.DictReader buffers
    internally — f.tell() raises OSError after the first next() call.
    """
    kws     = [k.lower() for k in keywords if len(k) > 2]
    results = []

    # Raise csv field size limit — some NTSB narratives exceed the 128 KB default
    import sys as _sys
    try:
        import csv as _csv
        _csv.field_size_limit(_sys.maxsize)
    except OverflowError:
        _csv.field_size_limit(2 ** 31 - 1)  # Windows fallback

    # Count total rows for progress bar (fast — CSV is small)
    # Note: uses line count, not row count — multi-line narrative fields will
    # make total_rows > actual row count. Progress bar is approximate; results are correct.
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            total_rows = sum(1 for _ in f)
    except Exception:
        total_rows = 0

    try:
        from tqdm import tqdm
        bar_kws = (keywords[:2] if keywords else ["all"])

        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            with tqdm(
                total=total_rows,
                unit=" rows",
                desc=f"  Filtering ({'+'.join(bar_kws[:2])})",
                bar_format="{l_bar}{bar}| {n:,}/{total:,} rows [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True,
            ) as bar:
                for row in reader:
                    bar.update(1)
                    blob = (
                        (row.get("narrative") or "") + " " +
                        (row.get("probable_cause") or "")
                    ).lower()
                    if kws and not any(k in blob for k in kws):
                        continue
                    results.append(dict(row))
                    if len(results) >= NTSB_MAX_RESULTS * 3:
                        break

    except ImportError:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                blob = ((row.get("narrative") or "") + " " + (row.get("probable_cause") or "")).lower()
                if kws and not any(k in blob for k in kws):
                    continue
                results.append(dict(row))
                if len(results) >= NTSB_MAX_RESULTS * 3:
                    break
    except Exception as e:
        log.error(f"  CSV parse error: {e}")

    log.info(f"  Filtered: {len(results)} matching records")
    return results


# ── ZIP download ──────────────────────────────────────────────────────────────

def _download_zip(dest: str) -> bool:
    try:
        import requests
        from tqdm import tqdm
    except ImportError as e:
        log.error(f"Missing: {e}. Run: pip install requests tqdm --break-system-packages")
        return False

    if not _check_disk("/opt"):
        return False

    log.info(f"  Downloading avall.zip → {dest}")
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(NTSB_BULK_CSV_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"  ntsb_avall.zip ({total/1024**2:.0f} MB)",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk); bar.update(len(chunk))

        log.info(f"  Downloaded: {os.path.getsize(dest)/1024**2:.1f} MB")
        return True
    except Exception as e:
        log.error(f"  Download failed: {e}")
        if Path(dest).exists(): Path(dest).unlink()
        return False


def _extract_mdb(zip_path: str, dest_dir: str) -> str | None:
    """Extract avall.mdb from ZIP with tqdm progress."""
    try:
        from tqdm import tqdm
        with zipfile.ZipFile(zip_path, "r") as zf:
            mdb_members = [m for m in zf.infolist() if m.filename.lower().endswith(".mdb")]
            if not mdb_members:
                log.error("No .mdb file found in ZIP")
                return None
            target   = mdb_members[0]
            out_path = str(Path(dest_dir) / Path(target.filename).name)

            if not _check_disk("/opt", required=target.file_size + MIN_FREE_BYTES):
                return None

            log.info(f"  Extracting {target.filename} ({target.file_size/1024**2:.0f} MB)")
            with zf.open(target) as src, open(out_path, "wb") as dst, tqdm(
                total=target.file_size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"  Extracting avall.mdb",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                dynamic_ncols=True,
            ) as bar:
                while chunk := src.read(1 << 20):
                    dst.write(chunk); bar.update(len(chunk))

            log.info(f"  Extracted: {out_path}")
            return out_path
    except Exception as e:
        log.error(f"  MDB extraction failed: {e}")
        return None


# ── ensure_mdb: guarantee avall.mdb is present ───────────────────────────────

def ensure_mdb(mdb_path: str | None = None) -> str | None:
    """
    Ensure avall.mdb is present. Download ZIP + extract if needed.
    Returns mdb_path on success, None if unavailable.
    """
    mdb_path = mdb_path or NTSB_MDB_PATH
    dest_dir = str(Path(mdb_path).parent)

    if os.path.exists(mdb_path) and os.path.getsize(mdb_path) > 1024 ** 2:
        log.info(f"  MDB present: {mdb_path} ({os.path.getsize(mdb_path)/1024**2:.0f} MB)")
        return mdb_path

    if os.path.exists(NTSB_ZIP_PATH) and os.path.getsize(NTSB_ZIP_PATH) > 1024:
        log.info(f"  ZIP present at {NTSB_ZIP_PATH} — extracting MDB")
        return _extract_mdb(NTSB_ZIP_PATH, dest_dir)

    log.info("  avall.zip not found — downloading")
    if not _download_zip(NTSB_ZIP_PATH):
        return None
    return _extract_mdb(NTSB_ZIP_PATH, dest_dir)


# ── ensure_csv: guarantee Cessna CSV is present ──────────────────────────────

def ensure_csv(
    csv_path: str | None = None,
    mdb_path: str | None = None,
) -> str | None:
    """
    Ensure the Cessna-filtered CSV is present.
    Builds it from the MDB if needed.
    """
    csv_path = csv_path or NTSB_CSV_PATH
    mdb_path = mdb_path or NTSB_MDB_PATH

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1024:
        log.info(
            f"  Cessna CSV present: {csv_path} "
            f"({os.path.getsize(csv_path)/1024:.0f} KB)"
        )
        return csv_path

    if os.path.exists(csv_path):
        log.warning(f"  Stale/empty CSV at {csv_path} — rebuilding")
        os.unlink(csv_path)

    resolved_mdb = ensure_mdb(mdb_path)
    if not resolved_mdb:
        return None

    success = _export_cessna(resolved_mdb, csv_path)
    return csv_path if success else None


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            c = json.load(f)
        records = c.get("records", [])
        if not records:
            log.info(f"  Cache empty at {path} — treating as miss")
            return None
        log.info(f"  Cache hit: {len(records)} records ({path})")
        return records
    except Exception as e:
        log.warning(f"  Cache read error: {e}")
        return None


def _save_cache(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_records": len(records),
                "records":   records,
            }, f, indent=2)
        log.info(f"  Cache saved -> {path} ({len(records)} records)")
    except Exception as e:
        log.warning(f"  Cache write failed: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def load_ntsb_accidents(
    query_keywords: list[str],
    cache_path: str | None = None,
    csv_path: str | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Load NTSB Cessna accident records matching query_keywords.
    Builds Cessna CSV from MDB on first call; uses cache on subsequent calls.
    """
    cache_path = cache_path or NTSB_CACHE_PATH
    csv_path   = csv_path   or NTSB_CSV_PATH

    if not force_refresh:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    resolved = ensure_csv(csv_path)
    if not resolved:
        log.warning(
            "\n  ┌─── NTSB DATA UNAVAILABLE ──────────────────────────────────────┐\n"
            "  │ avall.mdb could not be obtained. NTSB scores = 0.             │\n"
            "  │ Ensure mdbtools is installed: sudo apt install mdbtools        │\n"
            "  └────────────────────────────────────────────────────────────────┘\n"
        )
        return []

    records = _parse_csv(resolved, query_keywords)
    _save_cache(records, cache_path)
    return records


def load_ntsb_per_component(
    components: list[str],
    force_refresh: bool = False,
    csv_path: str | None = None,
) -> dict[str, list[dict]]:
    """
    Load NTSB records for each failure component using targeted keywords.
    Builds the Cessna CSV once before the component loop.
    """
    from backend.layer4.ntsb_matcher import COMPONENT_KEYWORDS

    csv_path = csv_path or NTSB_CSV_PATH

    # Build CSV once (no-op if already present)
    resolved = ensure_csv(csv_path)
    if not resolved:
        log.warning("NTSB data unavailable — all components will have 0 records")
        return {comp: [] for comp in components}

    results: dict[str, list[dict]] = {}
    for comp in components:
        keywords   = COMPONENT_KEYWORDS.get(comp, [comp.replace("_", " ")])
        cache_path = NTSB_CACHE_PATH.replace(".json", f"_{comp}.json")
        log.info(f"\n  Component: {comp}  keywords={keywords[:3]}")
        results[comp] = load_ntsb_accidents(
            query_keywords=keywords,
            cache_path=cache_path,
            csv_path=resolved,
            force_refresh=force_refresh,
        )

    total = sum(len(v) for v in results.values())
    log.info(f"\nPer-component complete: {total} records across {len(components)} components")
    return results