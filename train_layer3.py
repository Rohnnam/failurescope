#!/usr/bin/env python3
"""
FailureScope — Layer 3 Runner (v2)
====================================
Orchestrates the full Layer 3 pipeline: two models, working together.

Pipeline:
  1. Data inspection   (run once, validates data before ingestion)
  2. Preprocessing     (cascade windows + snapshots from one gz scan)
  3. Snapshot model    (XGBoost swiss cheese — which conditions at injection matter)
  4. Cascade model     (Transformer on cascade windows — which state changes matter)
  5. Relationship miner (combines both → final induced rules)

Usage:
  # Full pipeline (recommended for first run):
  python train_layer3.py --all

  # Step by step:
  python train_layer3.py --inspect          # understand data first
  python train_layer3.py --preprocess       # build cascade shards + snapshots
  python train_layer3.py --train-snapshot   # XGBoost swiss cheese model
  python train_layer3.py --train-cascade    # Transformer cascade model
  python train_layer3.py --mine             # combine -> induced rules
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_inspect(args) -> None:
    from backend.layer3.data_inspector import inspect
    inspect(n=50_000)


def cmd_preprocess(args) -> None:
    from preprocess_cascade import preprocess
    preprocess(inspect_first=args.inspect_first)


def cmd_train_snapshot(args) -> None:
    for pkg in ["xgboost", "shap", "scikit-learn"]:
        try:
            __import__(pkg.replace("-", "_").replace("scikit_learn", "sklearn"))
        except ImportError:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg,
                 "--break-system-packages", "-q"], check=True
            )

    from backend.layer3.snapshot_dataset import load_snapshots, snapshots_exist
    from backend.layer3.snapshot_model import extract_shap_rules, train_snapshot_model

    if not snapshots_exist():
        print("Snapshots not found. Run: python train_layer3.py --preprocess")
        sys.exit(1)

    train_X, train_y, val_X, val_y = load_snapshots()
    model = train_snapshot_model(train_X, train_y, val_X, val_y)
    extract_shap_rules(model, val_X, val_y)


def cmd_train_cascade(args) -> None:
    from backend.layer3.cascade_dataset import shards_exist
    if not shards_exist():
        print("Cascade shards not found. Run: python train_layer3.py --preprocess")
        sys.exit(1)
    from backend.layer3.train_cascade import train_cascade
    train_cascade(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)


def cmd_extract(args) -> None:
    from backend.layer3.train_cascade import extract_cascade_attention
    extract_cascade_attention(n_per_cascade=args.n_samples)


def cmd_mine(args) -> None:
    from backend.layer3.relationship_miner import mine_relationships
    result = mine_relationships()
    meta = result["meta"]
    print(f"\n✓ Layer 3 complete.")
    print(f"  {meta['n_induced']} INDUCED  {meta['n_novel']} NOVEL  {meta['n_confirmed']} CONFIRMED")
    print(f"  Results -> /opt/failurescope_data/models/induced_relationships.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="FailureScope Layer 3")
    parser.add_argument("--all",            action="store_true")
    parser.add_argument("--inspect",        action="store_true")
    parser.add_argument("--preprocess",     action="store_true")
    parser.add_argument("--inspect-first",  action="store_true")
    parser.add_argument("--train-snapshot", action="store_true")
    parser.add_argument("--train-cascade",  action="store_true")
    parser.add_argument("--extract",        action="store_true")
    parser.add_argument("--mine",           action="store_true")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--n_samples",  type=int,   default=200)
    args = parser.parse_args()

    if args.all:
        cmd_inspect(args)
        cmd_preprocess(args)
        cmd_train_snapshot(args)
        cmd_train_cascade(args)
        cmd_extract(args)
        cmd_mine(args)
        return

    ran = False
    if args.inspect:        cmd_inspect(args);        ran = True
    if args.preprocess:     cmd_preprocess(args);     ran = True
    if args.train_snapshot: cmd_train_snapshot(args); ran = True
    if args.train_cascade:  cmd_train_cascade(args);  ran = True
    if args.extract:        cmd_extract(args);        ran = True
    if args.mine:           cmd_mine(args);           ran = True

    if not ran:
        parser.print_help()
        print("\nQuick start:  python train_layer3.py --all")
        print("Fast start:   python train_layer3.py --preprocess --train-snapshot")


if __name__ == "__main__":
    main()