"""
FailureScope Layer 3 — Training Loop
======================================
Trains FailureScopeTransformer on 400K trajectories, validates on 100K.

Target (context1 [Success Metrics]): >90% accuracy on test set.

Key design decisions:
  - Mixed-precision (fp16) via torch.cuda.amp
  - Gradient checkpointing in model (enabled automatically during training)
  - Class-weighted loss: dataset is 33.8% catastrophic / 66.2% recoverable
    (per PROGRESS.md §4 Layer 2). Weights inverse to class frequency,
    normalised so they sum to 2.
  - OOM guard: if a batch triggers CUDA OOM the batch is skipped, cache is
    cleared, and training continues. If OOM occurs 3× consecutively, the
    script halts with a helpful message recommending batch_size reduction.
  - Best checkpoint saved to /opt/failurescope_data/models/trained_model.pt
    whenever val accuracy improves.
  - Pre-flight checks: disk space, GPU availability, Ollama (warns if running).

Usage:
  conda activate aviate
  systemctl stop ollama          # free ~2-3GB VRAM before training
  python train_layer3.py         # project root runner (calls train() here)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import (
    FEATURE_DIM,
    MAX_SEQ_LEN,
    TOTAL_SIZE,
    TRAIN_SIZE,
    TrajectoryDataset,
    collate_fn,
    dynamic_collate_fn,
)
from .model import FailureScopeTransformer
from .shard_dataset import ShardedDataset, SHARD_DIR, shards_exist, shard_info

# ── Paths ──────────────────────────────────────────────────────────────────────

TRAJECTORIES_PATH = Path(
    "/opt/failurescope_data/simulation_trajectories/trajectories.jsonl.gz"
)
MODEL_OUTPUT_DIR = Path("/opt/failurescope_data/models")
CHECKPOINT_PATH  = MODEL_OUTPUT_DIR / "trained_model.pt"
LOG_PATH         = MODEL_OUTPUT_DIR / "training_log.jsonl"

# ── Default training configuration ────────────────────────────────────────────

DEFAULT_CONFIG: Dict = {
    # Model — upgraded from d_model=128/4L to d_model=256/6L for richer attention
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "max_seq_len": MAX_SEQ_LEN,
    # Training — batch=64 safe for 8.2GB VRAM with d_model=256 (estimated peak ~1.76GB)
    # num_workers=4 to pipeline IO and reduce the IO bottleneck observed in session 3
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    # Class imbalance: 33.8% catastrophic / 66.2% recoverable (PROGRESS.md §4)
    "class_weights": [0.77, 1.43],
    # DataLoader — 4 workers to overlap gzip decompression with GPU compute
    "num_workers": 4,
    # OOM safety
    "max_consecutive_oom": 3,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preflight_checks(cfg: Dict) -> None:
    """Warn on potential issues before a long training run."""
    print("[Pre-flight] Running checks …")

    # Trajectories file
    if not TRAJECTORIES_PATH.exists():
        raise FileNotFoundError(
            f"Trajectories not found: {TRAJECTORIES_PATH}\n"
            "Run: python -m backend.layer2.batch_runner"
        )
    size_gb = TRAJECTORIES_PATH.stat().st_size / 1e9
    print(f"[Pre-flight] Trajectories: {size_gb:.2f} GB  ✓")

    # Disk space on /opt — hard guard per PROGRESS.md §2
    disk = shutil.disk_usage("/opt")
    free_gb = disk.free / 1e9
    if free_gb < 10.0:
        raise RuntimeError(
            f"[Pre-flight] /opt has only {free_gb:.1f} GB free. "
            "Need ≥10 GB. Free disk before training."
        )
    print(f"[Pre-flight] /opt free: {free_gb:.1f} GB  ✓")

    # GPU
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Pre-flight] GPU: {torch.cuda.get_device_name(0)}  ({vram_gb:.1f} GB VRAM)  ✓")
    else:
        print("[Pre-flight] ⚠  No CUDA GPU found — training on CPU (very slow)")

    # Ollama warning
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-x", "ollama"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(
                "[Pre-flight] ⚠  Ollama appears to be running. "
                "Run 'systemctl stop ollama' to free ~2-3 GB VRAM before training."
            )
    except Exception:
        pass

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Pre-flight] Output directory ready  ✓")

    # Shard detection — critical for IO performance
    if shards_exist(SHARD_DIR):
        info = shard_info(SHARD_DIR)
        print(
            f"[Pre-flight] ✓ Binary shards found  "
            f"({info['train_shards']} train + {info['val_shards']} val,  "
            f"{info['total_size_gb']} GB)  — fast IO mode"
        )
    else:
        print(
            "[Pre-flight] ⚠  No binary shards found. Training will use slow gz streaming.\n"
            "             Run first for ~10x speedup:\n"
            "               python preprocess_shards.py\n"
            "             Then re-run training."
        )


def _log(log_path: Path, entry: Dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Validation pass ────────────────────────────────────────────────────────────

def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (avg_loss, accuracy, per_class_recall).
    Per-class recall is logged to help spot if the model collapses to
    predicting the majority class (a common failure with imbalanced data).
    """
    model.eval()
    total_loss = 0.0
    tp = [0, 0]   # true positives per class
    fn = [0, 0]   # false negatives per class
    n_batches = 0

    with torch.no_grad():
        for features, pad_masks, labels in val_loader:
            try:
                features  = features.to(device, non_blocking=True)
                pad_masks = pad_masks.to(device, non_blocking=True)
                labels    = labels.to(device, non_blocking=True)

                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(features, pad_masks)
                    loss   = criterion(logits, labels)

                preds = logits.argmax(dim=-1)
                for cls in (0, 1):
                    tp[cls] += ((preds == cls) & (labels == cls)).sum().item()
                    fn[cls] += ((preds != cls) & (labels == cls)).sum().item()

                total_loss += loss.item()
                n_batches  += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

    n_batches = max(n_batches, 1)
    avg_loss  = total_loss / n_batches
    total_correct = sum(tp)
    total_samples = sum(tp) + sum(fn)
    accuracy  = total_correct / max(total_samples, 1)

    per_class_recall = {
        "recoverable_recall":  tp[0] / max(tp[0] + fn[0], 1),
        "catastrophic_recall": tp[1] / max(tp[1] + fn[1], 1),
    }
    return avg_loss, accuracy, per_class_recall


# ── Main training function ─────────────────────────────────────────────────────

def train(config: Optional[Dict] = None) -> float:
    """
    Train the FailureScopeTransformer.

    Returns best validation accuracy achieved.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    _preflight_checks(cfg)
    device = _get_device()

    print(f"\n[Layer 3] Device        : {device}")
    print(f"[Layer 3] Batch size    : {cfg['batch_size']}")
    print(f"[Layer 3] Max seq len   : {cfg['max_seq_len']}")
    print(f"[Layer 3] Epochs        : {cfg['epochs']}")
    print(f"[Layer 3] Train samples : {TRAIN_SIZE:,}")
    print(f"[Layer 3] Val samples   : {TOTAL_SIZE - TRAIN_SIZE:,}")

    # ── Datasets: use fast binary shards if available, else fall back to gz ────
    if shards_exist(SHARD_DIR):
        print(f"[Layer 3] IO mode       : FAST (binary shards)  ← ~10x faster\n")
        train_ds = ShardedDataset(
            shard_dir=SHARD_DIR, split="train",
            shuffle=True, max_seq_len=cfg["max_seq_len"],
        )
        val_ds = ShardedDataset(
            shard_dir=SHARD_DIR, split="val",
            shuffle=False, max_seq_len=cfg["max_seq_len"],
        )
        # With shards, workers each own disjoint files — use more workers
        effective_workers = min(cfg["num_workers"] * 2, 8)
    else:
        print(f"[Layer 3] IO mode       : SLOW (gz streaming)  "
              f"← run preprocess_shards.py first\n")
        train_ds = TrajectoryDataset(
            str(TRAJECTORIES_PATH), split="train", max_seq_len=cfg["max_seq_len"]
        )
        val_ds = TrajectoryDataset(
            str(TRAJECTORIES_PATH), split="val", max_seq_len=cfg["max_seq_len"]
        )
        effective_workers = cfg["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=effective_workers,
        collate_fn=dynamic_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=4 if effective_workers > 0 else None,
        persistent_workers=(effective_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        num_workers=effective_workers,
        collate_fn=dynamic_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=4 if effective_workers > 0 else None,
        persistent_workers=(effective_workers > 0),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = FailureScopeTransformer(
        feature_dim=FEATURE_DIM,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_seq_len=cfg["max_seq_len"],
        use_gradient_checkpointing=True,
    ).to(device)

    print(f"[Layer 3] Parameters: {model.count_parameters():,}")

    # ── Loss: weighted cross-entropy for class imbalance ───────────────────────
    class_weights = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["label_smoothing"],
    )

    # ── Optimiser + scaler ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
    )
    # Cosine annealing: gradually reduces LR from peak to 1% of peak over all epochs.
    # This prevents the model from overshooting when it transitions between shard types
    # and allows fine-grained convergence in later epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["learning_rate"] * 0.01,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    consecutive_oom = 0
    global_step    = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0
        epoch_start   = time.time()
        batch_idx     = 0

        for features, pad_masks, labels in train_loader:
            try:
                features  = features.to(device, non_blocking=True)
                pad_masks = pad_masks.to(device, non_blocking=True)
                labels    = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(features, pad_masks)
                    loss   = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()

                # ── Stats ──────────────────────────────────────────────────────
                with torch.no_grad():
                    preds          = logits.argmax(dim=-1)
                    batch_correct  = (preds == labels).sum().item()
                    epoch_correct += batch_correct
                    epoch_total   += labels.size(0)
                    epoch_loss    += loss.item()

                consecutive_oom = 0   # reset OOM counter on successful batch
                global_step     += 1
                batch_idx       += 1

                # ── Progress print every 200 batches ───────────────────────────
                if batch_idx % 200 == 0:
                    elapsed  = time.time() - epoch_start
                    avg_loss = epoch_loss / batch_idx
                    acc      = epoch_correct / max(epoch_total, 1)
                    vram_msg = ""
                    if device.type == "cuda":
                        alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                        vram_msg = f" | VRAM: {alloc_gb:.2f}GB"
                    print(
                        f"Epoch {epoch+1}/{cfg['epochs']} "
                        f"| Batch {batch_idx} "
                        f"| Loss {avg_loss:.4f} "
                        f"| TrainAcc {acc:.4f} "
                        f"| {elapsed:.0f}s"
                        f"{vram_msg}"
                    )
                    _log(LOG_PATH, {
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "step": global_step,
                        "loss": avg_loss,
                        "train_acc": acc,
                        "elapsed_s": round(elapsed, 1),
                    })

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                consecutive_oom += 1
                print(
                    f"[OOM] CUDA out-of-memory at epoch {epoch+1} "
                    f"batch {batch_idx} (#{consecutive_oom}). "
                    f"Batch skipped. Cache cleared."
                )
                if consecutive_oom >= cfg["max_consecutive_oom"]:
                    print(
                        f"\n[OOM HALT] {consecutive_oom} consecutive OOM errors.\n"
                        f"Recommended fix: reduce batch_size "
                        f"(currently {cfg['batch_size']}) to "
                        f"{cfg['batch_size'] // 2} and restart.\n"
                        f"  python train_layer3.py --batch_size {cfg['batch_size'] // 2}\n"
                        "Also ensure Ollama is stopped: systemctl stop ollama"
                    )
                    sys.exit(1)
                batch_idx += 1
                continue

        # ── Epoch complete: validate ───────────────────────────────────────────
        val_loss, val_acc, recalls = _validate(
            model, val_loader, criterion, device
        )
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed   = time.time() - epoch_start

        print(
            f"\n{'='*60}\n"
            f"Epoch {epoch+1}/{cfg['epochs']} complete  ({elapsed:.0f}s)\n"
            f"  Train acc   : {train_acc:.4f}\n"
            f"  Val acc     : {val_acc:.4f}\n"
            f"  Val loss    : {val_loss:.4f}\n"
            f"  Rec recall  : {recalls['recoverable_recall']:.4f}\n"
            f"  Cat recall  : {recalls['catastrophic_recall']:.4f}\n"
            f"  LR          : {scheduler.get_last_lr()[0]:.2e}\n"
            f"{'='*60}\n"
        )

        scheduler.step()

        _log(LOG_PATH, {
            "epoch": epoch + 1,
            "event": "epoch_end",
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss,
            **recalls,
        })

        # ── Save best model ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": cfg,
                    "model_config": {
                        "feature_dim": FEATURE_DIM,
                        "d_model": cfg["d_model"],
                        "nhead": cfg["nhead"],
                        "num_layers": cfg["num_layers"],
                        "dim_feedforward": cfg["dim_feedforward"],
                        "dropout": cfg["dropout"],
                        "max_seq_len": cfg["max_seq_len"],
                        "use_gradient_checkpointing": False,  # inference default
                    },
                },
                CHECKPOINT_PATH,
            )
            print(f"[Layer 3] ✓ Best model saved  val_acc={val_acc:.4f}  → {CHECKPOINT_PATH}")

    # ── Final report ──────────────────────────────────────────────────────────
    target = 0.90  # per context1 [Success Metrics]
    print(f"\n[Layer 3] Training complete.")
    print(f"[Layer 3] Best val accuracy: {best_val_acc:.4f}")

    if best_val_acc >= target:
        print(f"[Layer 3] ✓ Target met (≥{target:.0%}). Proceed to attention extraction.")
    else:
        print(
            f"[Layer 3] ✗ Target not met ({best_val_acc:.4f} < {target:.0%}).\n"
            "Options per PROGRESS.md §15B:\n"
            "  1. Regenerate trajectories with R006 fix applied (5% of engine_left\n"
            "     scenarios had incorrect voltage cascade — see PROGRESS.md §15B).\n"
            "     Command: python -m backend.layer2.batch_runner\n"
            "  2. Train more epochs: add --epochs 10 to train_layer3.py\n"
            "  3. Increase d_model to 256 (requires more VRAM — verify first)"
        )

    return best_val_acc