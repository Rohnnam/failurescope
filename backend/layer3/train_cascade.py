"""
FailureScope Layer 3 — Cascade Transformer Training
=====================================================
Trains the CascadeTransformer on cascade windows.
Also runs attention extraction post-training.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .cascade_dataset import (
    CASCADE_SHARD_DIR, CascadeShardDataset,
    cascade_collate_fn, shards_exist,
)
from .cascade_model import CascadeTransformer
from .constants import (
    CASCADE_CONTEXT_DIM, CASCADE_MAX_STEPS, CASCADE_COMPONENTS,
    FAILURE_COMPONENTS, L1_RULES, MIN_CASCADE_DURATION_S,
    N_STATE_KEYS, STATE_KEYS, TRAJECTORIES_PATH_STR,
)

MODELS_DIR   = Path("/opt/failurescope_data/models")
MODEL_PATH   = MODELS_DIR / "cascade_model.pt"
ATTN_PATH    = MODELS_DIR / "cascade_attention.json"
LOG_PATH     = MODELS_DIR / "cascade_training_log.jsonl"

# Class weights: 33.8% cat → upweight catastrophic
CLASS_WEIGHTS = [0.77, 1.43]


def train_cascade(
    batch_size: int   = 128,
    epochs:     int   = 10,
    lr:         float = 3e-4,
) -> float:
    """Train CascadeTransformer. Returns best val accuracy."""

    if not shards_exist():
        raise RuntimeError("Cascade shards not found. Run: python preprocess_cascade.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Preflight ──────────────────────────────────────────────────────────────
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Cascade] GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    print(f"[Cascade] Batch={batch_size}  Epochs={epochs}  LR={lr}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = CascadeShardDataset(CASCADE_SHARD_DIR, split="train", shuffle=True)
    val_ds   = CascadeShardDataset(CASCADE_SHARD_DIR, split="val",   shuffle=False)

    n_workers = min(4, 8)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=n_workers,
        collate_fn=cascade_collate_fn, pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if n_workers > 0 else None,
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, num_workers=n_workers,
        collate_fn=cascade_collate_fn, pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if n_workers > 0 else None,
        persistent_workers=(n_workers > 0),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = CascadeTransformer(
        feature_dim=N_STATE_KEYS,
        context_dim=CASCADE_CONTEXT_DIM,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_steps=CASCADE_MAX_STEPS,
        use_gradient_checkpointing=True,
    ).to(device)
    print(f"[Cascade] Parameters: {model.count_parameters():,}")

    # ── Loss + optimiser ───────────────────────────────────────────────────────
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        ep_loss = ep_correct = ep_total = 0
        t0 = time.time()

        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                         unit="batch", ncols=90, leave=False)
            loader_iter = pbar
        except ImportError:
            pbar = None
            loader_iter = train_loader

        for bi, (seq, ctx, mask, lbl) in enumerate(loader_iter):
            try:
                seq  = seq.to(device, non_blocking=True)
                ctx  = ctx.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                lbl  = lbl.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(seq, ctx, mask)
                    loss   = criterion(logits, lbl)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    preds      = logits.argmax(-1)
                    ep_correct += (preds == lbl).sum().item()
                    ep_total   += lbl.size(0)
                    ep_loss    += loss.item()

                if pbar is not None:
                    acc = ep_correct / max(ep_total, 1)
                    vram = torch.cuda.memory_allocated(device)/1e9 if device.type=="cuda" else 0
                    pbar.set_postfix(loss=f"{ep_loss/(bi+1):.3f}",
                                     acc=f"{acc:.3f}",
                                     vram=f"{vram:.2f}GB")
                elif (bi + 1) % 200 == 0:
                    vram = f"  VRAM: {torch.cuda.memory_allocated(device)/1e9:.2f}GB" \
                           if device.type == "cuda" else ""
                    print(f"Epoch {epoch+1}/{epochs} | Batch {bi+1} | "
                          f"Loss {ep_loss/(bi+1):.4f} | "
                          f"TrainAcc {ep_correct/max(ep_total,1):.4f}{vram}")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if pbar: pbar.write(f"[OOM] Batch {bi+1} skipped.")
                else: print(f"[OOM] Batch {bi+1} skipped. Reduce --batch_size if this repeats.")
                continue

        if pbar is not None:
            pbar.close()

        # Validate
        val_loss, val_acc, recalls = _validate(model, val_loader, criterion, device)
        train_acc = ep_correct / max(ep_total, 1)
        scheduler.step()

        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{epochs}  ({time.time()-t0:.0f}s)")
        print(f"  Train acc:   {train_acc:.4f}")
        print(f"  Val acc:     {val_acc:.4f}")
        print(f"  Val loss:    {val_loss:.4f}")
        print(f"  Cat recall:  {recalls['cat']:.4f}")
        print(f"  Rec recall:  {recalls['rec']:.4f}")
        print(f"  LR:          {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*55}\n")

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "epoch": epoch+1, "train_acc": train_acc,
                "val_acc": val_acc, "val_loss": val_loss, **recalls,
            }) + "\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "model_config": {
                    "feature_dim": N_STATE_KEYS,
                    "context_dim": CASCADE_CONTEXT_DIM,
                    "d_model": 128, "nhead": 8, "num_layers": 4,
                    "dim_feedforward": 512, "dropout": 0.1,
                    "max_steps": CASCADE_MAX_STEPS,
                    "use_gradient_checkpointing": False,
                },
            }, MODEL_PATH)
            print(f"[Cascade] ✓ Best model saved  val_acc={val_acc:.4f}")

    print(f"\n[Cascade] Training complete. Best val acc: {best_val_acc:.4f}")
    return best_val_acc


def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = tp = fn_cat = fn_rec = tn = 0
    n_batches = 0
    with torch.no_grad():
        for seq, ctx, mask, lbl in loader:
            try:
                seq  = seq.to(device, non_blocking=True)
                ctx  = ctx.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                lbl  = lbl.to(device, non_blocking=True)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(seq, ctx, mask)
                    loss   = criterion(logits, lbl)
                preds = logits.argmax(-1)
                tp    += ((preds == 1) & (lbl == 1)).sum().item()
                tn    += ((preds == 0) & (lbl == 0)).sum().item()
                fn_cat += ((preds == 0) & (lbl == 1)).sum().item()
                fn_rec += ((preds == 1) & (lbl == 0)).sum().item()
                total_loss += loss.item()
                n_batches  += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
    n_batches = max(n_batches, 1)
    correct   = tp + tn
    total     = tp + tn + fn_cat + fn_rec
    return (
        total_loss / n_batches,
        correct / max(total, 1),
        {
            "cat": tp / max(tp + fn_cat, 1),
            "rec": tn / max(tn + fn_rec, 1),
        },
    )


def extract_cascade_attention(
    model_path: Path = MODEL_PATH,
    trajectories_path: Path = Path(TRAJECTORIES_PATH_STR),
    output_path: Path = ATTN_PATH,
    n_per_cascade: int = 200,
    seed: int = 42,
) -> dict:
    """
    Extract attention patterns from trained CascadeTransformer.
    Only uses trajectories with cascade_duration >= MIN_CASCADE_DURATION_S.
    Only looks at state changes AFTER inject_t (actual cascade physics).
    Kill zones will show actual degradation ranges, not nominal values.
    """
    import gzip
    import pickle
    from collections import Counter

    if not model_path.exists():
        raise FileNotFoundError(f"Cascade model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    cfg    = dict(ckpt["model_config"])
    cfg["use_gradient_checkpointing"] = False

    model = CascadeTransformer(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[CascadeExtractor] Loaded model  val_acc={ckpt['val_acc']:.4f}")
    print(f"[CascadeExtractor] Cascade filter: duration >= {MIN_CASCADE_DURATION_S:.0f}s")

    # Load val line indices
    val_idx_path = CASCADE_SHARD_DIR / "val_line_indices.pkl"
    val_line_set = None
    if val_idx_path.exists():
        with open(val_idx_path, "rb") as f:
            val_line_set = pickle.load(f)
        print(f"[CascadeExtractor] Val lines: {len(val_line_set):,}")

    # Build index of catastrophic val trajectories with slow cascades
    from .data_inspector import inspect as _inspect
    cat_index: Dict[str, List] = defaultdict(list)
    rec_lines = []

    with gzip.open(trajectories_path, "rt") as f:
        for line_idx, line in enumerate(f):
            if val_line_set is not None and line_idx not in val_line_set:
                continue
            if val_line_set is None and line_idx < int(0.8 * 500_000):
                continue
            try:
                rec     = json.loads(line)
                outcome = rec.get("outcome", "")
                fails   = rec.get("failures", [])
                if not fails:
                    continue
                inject_t = float(fails[0].get("trigger_time_s", 0))
                ttc_s    = float(rec.get("ttc_s") or 0)
                cascade  = ttc_s - inject_t

                if outcome == "catastrophic" and cascade >= MIN_CASCADE_DURATION_S:
                    comp = fails[0].get("component", "unknown")
                    cat_index[comp].append(line_idx)
                elif outcome == "recoverable" and len(rec_lines) < 400:
                    rec_lines.append(line_idx)
            except Exception:
                continue

    print(f"[CascadeExtractor] Components found: {len(cat_index)}")
    for comp, lines in sorted(cat_index.items()):
        print(f"  {comp:25s}  {len(lines):5d}")

    # Sample per component
    rng = np.random.default_rng(seed)
    sampled: Dict[str, List[int]] = {}
    for comp, lines in cat_index.items():
        n = min(n_per_cascade, len(lines))
        sampled[comp] = sorted(rng.choice(lines, size=n, replace=False).tolist())

    # Load records
    needed = {ln: comp for comp, lines in sampled.items() for ln in lines}
    needed.update({ln: "__rec__" for ln in rec_lines})
    records: Dict[str, List] = defaultdict(list)

    with gzip.open(trajectories_path, "rt") as f:
        for line_idx, line in enumerate(f):
            if line_idx not in needed:
                continue
            try:
                rec  = json.loads(line)
                comp = needed[line_idx]
                records[comp].append(rec)
            except Exception:
                continue

    rec_records = records.pop("__rec__", [])
    print(f"[CascadeExtractor] Loaded {sum(len(v) for v in records.values())} catastrophic "
          f"+ {len(rec_records)} recoverable")

    # Analyse each component
    per_component: Dict[str, dict] = {}
    from .cascade_dataset import extract_cascade_sequence, extract_context_vector
    # All timesteps in cascade window are post-injection state changes

    for comp, recs in sorted(records.items()):
        analyses = []
        for rec in recs:
            analysis = _analyse_cascade_record(model, rec, device, cfg["max_steps"])
            if analysis:
                analyses.append(analysis)

        if analyses:
            agg = _aggregate_cascade_analyses(analyses, comp, rec_records)
            per_component[comp] = agg
            tf = agg.get("timing_fractions", {})
            print(f"  {comp:25s}  quality={agg.get('quality_score',0):.3f}  "
                  f"EARLY={tf.get('EARLY_CASCADE',0):.2f}  "
                  f"MID={tf.get('MID_CASCADE',0):.2f}  "
                  f"crash_prob={agg.get('crash_probability_in_zone')}")

    # Global variable importance
    var_scores: Dict[str, list] = defaultdict(list)
    for res in per_component.values():
        for entry in res.get("variable_attention_ranking", []):
            var_scores[entry["variable"]].append(entry["mean_attention"])

    global_var_importance = sorted([
        {"variable": v, "mean_attention": round(float(np.mean(s)), 3), "n_components": len(s)}
        for v, s in var_scores.items()
    ], key=lambda x: -x["mean_attention"])

    print(f"\n[CascadeExtractor] Global variable importance:")
    for e in global_var_importance[:6]:
        print(f"  {e['variable']:15s}  {e['mean_attention']:.3f}  (across {e['n_components']} components)")

    # Emergent rules
    emergent_rules = [
        res["induced_rule"]
        for res in per_component.values()
        if res.get("induced_rule", {}).get("quality_score", 0) > 0.1
    ]
    emergent_rules.sort(key=lambda r: -(r.get("quality_score") or 0))

    output = {
        "meta": {
            "model_val_acc":      ckpt["val_acc"],
            "extractor_version":  "cascade_v1",
            "min_cascade_s":      MIN_CASCADE_DURATION_S,
            "n_components":       len(per_component),
            "n_emergent_rules":   len(emergent_rules),
        },
        "variable_importance_global": global_var_importance,
        "per_component":              per_component,
        "emergent_rules":             emergent_rules,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[CascadeExtractor] ✓ Saved → {output_path}")
    return output


def _analyse_cascade_record(model, record: dict, device, max_steps: int) -> Optional[dict]:
    from .cascade_dataset import extract_cascade_sequence, extract_context_vector
    import torch

    fails   = record.get("failures", [])
    states  = record.get("states", [])
    if not fails or not states:
        return None

    inject_t = float(fails[0].get("trigger_time_s", 0))
    ttc_s    = float(record.get("ttc_s") or len(states))
    cascade  = ttc_s - inject_t
    if cascade < MIN_CASCADE_DURATION_S:
        return None

    seq, actual_len = extract_cascade_sequence(states, inject_t, ttc_s, max_steps)

    # Find inject_state
    inject_state = None
    for s in states:
        if float(s.get("t", 0)) >= inject_t - 0.5:
            inject_state = s
            break
    ctx = extract_context_vector(record, inject_state)

    seq_t = seq.unsqueeze(0).float().to(device)
    ctx_t = ctx.unsqueeze(0).to(device)
    mask  = torch.ones(1, actual_len, dtype=torch.bool).to(device)

    with torch.no_grad():
        logits, attn_list = model(seq_t, ctx_t, mask, return_attention=True)

    prob_cat = logits.softmax(-1)[0, 1].item()

    # Get cascade window states (after inject_t)
    cascade_states = []
    for s in states:
        if float(s.get("t", 0)) >= inject_t - 0.5:
            cascade_states.append(s)
        if len(cascade_states) >= actual_len:
            break

    # Average attention (context token = position 0, sequence starts at 1)
    all_head_attns = []
    for lw in attn_list:
        cls_row = lw[0, :, 0, 1:actual_len + 1].numpy()  # (nhead, actual_len)
        all_head_attns.append(cls_row)
    avg_attn = np.array(all_head_attns).mean(axis=(0, 1))  # (actual_len,)

    # Key: all timesteps are DURING CASCADE (after inject_t)
    # So ALL attention is meaningful — no "nominal flight" contamination
    # Classify relative to when cascade becomes terminal (last 20%)
    terminal_start = inject_t + cascade * 0.8

    critical_states = []
    for idx in np.argsort(-avg_attn)[:15]:
        if idx >= len(cascade_states):
            continue
        s  = cascade_states[idx]
        t  = float(s.get("t", 0))
        tc = "TERMINAL" if t >= terminal_start else "MID_CASCADE"
        critical_states.append({
            "t":        t,
            "attn_w":   float(avg_attn[idx]),
            "timing_cls": tc,
            "v":        float(s.get("v", 14.0)),
            "ff":       float(s.get("ff", 11.0)),
            "pump_eff": float(s.get("pump_eff", 1.0)),
            "rpm":      float(s.get("rpm", 2400.0)),
            "pitch":    float(s.get("pitch", 5.0)),
            "alt":      float(s.get("alt", 5000.0)),
            "vs":       float(s.get("vs", 0.0)),
        })

    # Variable attention ranking (states during cascade — WILL have IQR > 0)
    var_ranking = []
    for var in ("v", "ff", "pump_eff", "pitch", "alt", "rpm", "vs"):
        if len(cascade_states) > 0:
            # Correlate attention with this variable's value
            vals = np.array([float(cascade_states[i].get(var, 0)) for i in range(len(avg_attn))])
            if vals.std() > 1e-6:
                corr = float(np.corrcoef(avg_attn, vals)[0, 1])
                var_ranking.append({
                    "variable": var,
                    "mean_attention": round(float(avg_attn.mean()), 4),
                    "correlation_with_attn": round(corr, 3),
                })

    # Non-terminal states
    mid_states = [s for s in critical_states if s["timing_cls"] == "MID_CASCADE"]
    terminal_states = [s for s in critical_states if s["timing_cls"] == "TERMINAL"]
    n_total = max(len(critical_states), 1)

    return {
        "primary_failure":   fails[0].get("component", "unknown"),
        "cascade_duration":  cascade,
        "prob_cat":          prob_cat,
        "inject_t":          inject_t,
        "ttc_s":             ttc_s,
        "critical_states":   critical_states,
        "timing_fractions": {
            "MID_CASCADE": len(mid_states) / n_total,
            "TERMINAL":    len(terminal_states) / n_total,
            "EARLY_CASCADE": 0.0,  # N/A — all states are DURING cascade
        },
        "var_ranking": var_ranking,
    }


def _aggregate_cascade_analyses(analyses: List[dict], component: str, rec_records: List[dict]) -> dict:
    from .snapshot_model import _build_snapshot_induced_rule
    import numpy as np
    from collections import Counter

    n = len(analyses)
    timing_agg = {"MID_CASCADE": [], "TERMINAL": [], "EARLY_CASCADE": []}
    for a in analyses:
        for cls, frac in a["timing_fractions"].items():
            timing_agg[cls].append(frac)

    timing_summary = {cls: round(float(np.mean(v)), 3) for cls, v in timing_agg.items()}

    # Kill zone from MID_CASCADE states (before terminal — most informative)
    mid_states = [
        s for a in analyses
        for s in a["critical_states"]
        if s["timing_cls"] == "MID_CASCADE"
    ]

    kill_zone = {}
    if mid_states:
        for var in ("v", "ff", "pump_eff", "pitch", "alt", "rpm", "vs"):
            vals = [s[var] for s in mid_states]
            kill_zone[var] = {
                "p10": round(float(np.percentile(vals, 10)), 3),
                "p25": round(float(np.percentile(vals, 25)), 3),
                "p75": round(float(np.percentile(vals, 75)), 3),
                "p90": round(float(np.percentile(vals, 90)), 3),
                "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 3),
                "mean": round(float(np.mean(vals)), 3),
            }
        times = [s["t"] for s in mid_states]
        kill_zone["time_since_inject_s"] = {
            "p25": round(float(np.percentile(times, 25)), 0),
            "p75": round(float(np.percentile(times, 75)), 0),
        }

    # Variable attention ranking
    var_vote: Dict[str, list] = defaultdict(list)
    for a in analyses:
        for e in a.get("var_ranking", []):
            var_vote[e["variable"]].append(abs(e.get("correlation_with_attn", 0)))

    var_ranking = sorted([
        {"variable": v, "mean_attention": round(float(np.mean(s)), 3)}
        for v, s in var_vote.items()
    ], key=lambda x: -x["mean_attention"])

    # Quality: is kill zone showing actual degradation?
    quality = 0.0
    if kill_zone:
        v_iqr   = kill_zone.get("v",        {}).get("iqr", 0)
        p_iqr   = kill_zone.get("pump_eff", {}).get("iqr", 0)
        f_iqr   = kill_zone.get("ff",       {}).get("iqr", 0)
        alt_iqr = kill_zone.get("alt",      {}).get("iqr", 0)
        rpm_iqr = kill_zone.get("rpm",      {}).get("iqr", 0)
        pit_iqr = kill_zone.get("pitch",    {}).get("iqr", 0)

        # Quality based on which variables actually vary during cascade.
        # v and pump_eff are step functions (instant drop at inject_t) → IQR=0
        # alt, rpm, pitch vary continuously → use these for quality measurement
        # Normalise: alt varies 0-16500ft, rpm 0-2400, pitch 0-45°
        dynamic_quality = min(1.0, (
            alt_iqr / 2000.0 +   # 2000ft IQR = fully dynamic altitude
            rpm_iqr / 800.0  +   # 800 RPM IQR = significant engine decay
            pit_iqr / 10.0       # 10° IQR = active pitch changes
        ) / 3.0)

        # Also reward voltage IQR if present (slow cascades)
        elec_quality = min(1.0, (v_iqr / 3.0 + p_iqr / 0.5 + f_iqr / 3.0) / 3.0)

        quality = round(max(dynamic_quality, elec_quality), 3)

    # Crash probability in kill zone
    crash_prob = rec_prob = None
    if kill_zone and rec_records and mid_states:
        zone = {}
        # Use dynamically-varying variables (alt, rpm, pitch) for zone definition
        # v and ff are step functions with IQR=0 — unusable for zone bounds
        for var in ("alt", "rpm", "pitch"):
            kz = kill_zone.get(var, {})
            if kz.get("iqr", 0) > 1.0:  # meaningful variance
                zone[var] = (kz["p25"], kz["p75"])
        # Fallback to v/ff if dynamic vars have no variance (shouldn't happen)
        if not zone:
            for var in ("v", "ff"):
                kz = kill_zone.get(var, {})
                if kz.get("iqr", 0) > 0.1:
                    zone[var] = (kz["p25"], kz["p75"])

        if zone:
            n_cat_in   = len([s for s in mid_states if all(
                lo <= s.get(v, 0) <= hi for v, (lo, hi) in zone.items())])
            cat_frac = n_cat_in / max(len(mid_states), 1)

            rec_in_zone = 0
            rec_total   = 0
            for rec in rec_records:
                fails    = rec.get("failures", [])
                inject_t = float(fails[0].get("trigger_time_s", 0)) if fails else 0
                for s in rec.get("states", []):
                    if float(s.get("t", 0)) < inject_t:
                        continue
                    rec_total += 1
                    if all(lo <= float(s.get(v, 0)) <= hi for v, (lo, hi) in zone.items()):
                        rec_in_zone += 1

            rec_frac = rec_in_zone / max(rec_total, 1)

            denom = cat_frac * 0.338 + rec_frac * 0.662
            if denom > 1e-9:
                crash_prob = round(cat_frac * 0.338 / denom, 3)
                rec_prob   = round(rec_frac * 0.662 / denom, 3)

    # Build context1-format induced rule
    v_kz  = kill_zone.get("v", {})
    ff_kz = kill_zone.get("ff", {})
    p_kz  = kill_zone.get("pitch", {})
    a_kz  = kill_zone.get("alt", {})
    t_kz  = kill_zone.get("time_since_inject_s", {})

    top_vars = [e["variable"] for e in var_ranking[:2]]
    induced_rel = " + ".join(top_vars) + " cascade interaction" if top_vars else f"{component}_cascade"

    v_iqr = v_kz.get("iqr", 0)
    iqr_note = (
        f"voltage IQR={v_iqr:.2f}V indicates actual electrical degradation during cascade"
        if v_iqr > 0.1 else
        f"voltage IQR={v_iqr:.2f}V — minimal electrical change in this cascade type"
    )

    induced_rule = {
        "induced_relationship":           induced_rel,
        "primary_failure":                component,
        "is_cascade_component":           component in CASCADE_COMPONENTS,
        "critical_state_combination": {
            "voltage_range_V":            [v_kz.get("p25"), v_kz.get("p75")],
            "fuel_flow_range_gph":        [ff_kz.get("p25"), ff_kz.get("p75")],
            "pitch_range_deg":            [p_kz.get("p25"), p_kz.get("p75")],
            "altitude_range_ft":          [a_kz.get("p25"), a_kz.get("p75")],
            "time_since_inject_s":        t_kz,
        },
        "voltage_iqr_V":                  v_kz.get("iqr", 0),
        "pump_eff_iqr":                   kill_zone.get("pump_eff", {}).get("iqr", 0),
        "crash_probability_in_this_zone": crash_prob,
        "recovery_probability_in_this_zone": rec_prob,
        "interpretation":                 (
            f"For '{component}' failures with cascade duration >= {MIN_CASCADE_DURATION_S:.0f}s: "
            f"during the cascade window, attention focuses on "
            f"voltage={v_kz.get('p25','?')}–{v_kz.get('p75','?')} V, "
            f"fuel_flow={ff_kz.get('p25','?')}–{ff_kz.get('p75','?')} GPH, "
            f"pitch={p_kz.get('p25','?')}°–{p_kz.get('p75','?')}°, "
            f"altitude={a_kz.get('p25','?')}–{a_kz.get('p75','?')} ft. "
            f"{iqr_note}. n={n} trajectories."
        ),
        "quality_score":                  quality,
        "attention_type":                 "CASCADE_WINDOW (no nominal flight contamination)",
        "validation_status":              "PENDING — Method 2 (physics) + Method 3 (NTSB)",
        "source":                         "cascade_transformer_attention",
    }

    return {
        "n_trajectories":            n,
        "timing_fractions":          timing_summary,
        "quality_score":             quality,
        "variable_attention_ranking": var_ranking,
        "kill_zone":                 kill_zone,
        "crash_probability_in_zone": crash_prob,
        "recovery_probability_in_zone": rec_prob,
        "induced_rule":              induced_rule,
    }