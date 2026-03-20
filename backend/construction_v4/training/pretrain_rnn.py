"""
RNN Pre-training — SMILES language model on ZINC250k
=====================================================
Trains a CharRNN to generate valid SMILES strings by next-character
prediction on the ZINC250k dataset. Produces a pre-trained checkpoint
that serves as both:
  1. The base for RL fine-tuning (policy initialisation)
  2. The frozen prior for KL regularisation (prevents mode collapse)

Usage:
    # Full training on Kaggle GPU:
    python training/pretrain_rnn.py --epochs 30 --device cuda

    # Quick smoke test on CPU (2 epochs, small batch):
    python training/pretrain_rnn.py --epochs 2 --batch-size 32 --device cpu

    # Resume from checkpoint:
    python training/pretrain_rnn.py --resume checkpoints/rnn_epoch_10.pt

Output:
    checkpoints/rnn_pretrained.pt  — best validation loss
    checkpoints/rnn_epoch_{n}.pt   — every 5 epochs
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import (
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH,
    PRETRAIN_LR,
    PRETRAIN_PATIENCE,
    V4_CHECKPOINT_DIR,
    ZINC_DATA_PATH,
    VOCAB_SIZE,
    RANDOM_STATE,
)
from data.smiles_dataset import SMILESVocab, SMILESDataset, collate_fn
from models.char_rnn import CharRNN


def compute_validity(smiles_list):
    """Compute validity, uniqueness, and novelty of generated SMILES."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return {
            "validity": -1,
            "uniqueness": -1,
            "n_valid": 0,
            "n_unique": 0,
            "n_total": len(smiles_list),
        }

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            valid.append(canonical)

    n_total = len(smiles_list)
    n_valid = len(valid)
    n_unique = len(set(valid))

    return {
        "validity": n_valid / max(n_total, 1),
        "uniqueness": n_unique / max(n_valid, 1),
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_total": n_total,
    }


def train_epoch(model, dataloader, optimizer, criterion, device, vocab):
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (input_ids, target_ids, lengths) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits, _ = model(input_ids)
        # logits: (batch, seq_len, vocab_size)
        # target_ids: (batch, seq_len)

        # Flatten for cross-entropy
        loss = criterion(
            logits.reshape(-1, model.vocab_size),
            target_ids.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, target_ids, lengths in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, _ = model(input_ids)
            loss = criterion(
                logits.reshape(-1, model.vocab_size),
                target_ids.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Pre-train SMILES CharRNN")
    parser.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH)
    parser.add_argument("--lr", type=float, default=PRETRAIN_LR)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (for Kaggle)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(V4_CHECKPOINT_DIR),
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--sample-every", type=int, default=5, help="Sample and evaluate every N epochs"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=512,
        help="Number of molecules to sample for evaluation",
    )
    args = parser.parse_args()

    # ── Device setup ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"{'=' * 60}")
    print(f"  SMILES RNN Pre-training")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {args.batch_size}")
    print(f"  LR:     {args.lr}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    # ── Data ──
    data_path = args.data_dir or str(ZINC_DATA_PATH)
    if args.data_dir:
        # Kaggle mode: look for csv in the specified dir
        import glob

        csvs = glob.glob(os.path.join(args.data_dir, "*.csv"))
        if csvs:
            data_path = csvs[0]
        print(f"  Data (Kaggle): {data_path}")
    else:
        print(f"  Data: {data_path}")

    if not os.path.exists(data_path):
        print(f"\n  [ERROR] Data file not found: {data_path}")
        print(f"  Run: python data/zinc_downloader.py")
        sys.exit(1)

    vocab = SMILESVocab()
    print(f"  Vocab size: {vocab.size}")

    full_dataset = SMILESDataset(data_path, vocab)

    # 90/10 train/val split
    n_total = len(full_dataset)
    n_val = max(int(0.1 * n_total), 1000)
    n_train = n_total - n_val

    torch.manual_seed(RANDOM_STATE)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatible
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Model ──
    model = CharRNN(vocab_size=vocab.size, vocab=vocab).to(device)

    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        model = CharRNN.load(args.resume, device=str(device))
        # Try to extract epoch number from filename
        try:
            basename = os.path.basename(args.resume)
            if "epoch_" in basename:
                start_epoch = int(basename.split("epoch_")[1].split(".")[0])
        except (ValueError, IndexError):
            pass

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # ── Checkpoint dir ──
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ──
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'─' * 60}")
    print(f"  Starting training...")
    print(f"{'─' * 60}\n")

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, vocab
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # LR scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - t_epoch
        total_elapsed = time.time() - t_start
        eta = (
            (total_elapsed / (epoch - start_epoch + 1)) * (args.epochs - epoch - 1)
            if epoch > start_epoch
            else 0
        )

        print(
            f"  Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s  ETA={eta / 60:.0f}m"
        )

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(ckpt_dir, "rnn_pretrained.pt"))
            print(f"  ★ New best val_loss={val_loss:.4f}, saved rnn_pretrained.pt")
        else:
            patience_counter += 1

        # Periodic sampling and evaluation
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            print(f"\n  Sampling {args.n_samples} molecules...")
            samples = model.sample(args.n_samples, temperature=1.0, device=str(device))
            metrics = compute_validity(samples)
            print(
                f"  Validity:   {metrics['validity']:.1%} "
                f"({metrics['n_valid']}/{metrics['n_total']})"
            )
            print(
                f"  Uniqueness: {metrics['uniqueness']:.1%} "
                f"({metrics['n_unique']}/{metrics['n_valid']})"
            )
            print(f"  Sample molecules:")
            for s in samples[:5]:
                print(f"    {s}")
            print()

            # Save epoch checkpoint
            model.save(os.path.join(ckpt_dir, f"rnn_epoch_{epoch + 1}.pt"))

        # Early stopping
        if patience_counter >= PRETRAIN_PATIENCE:
            print(
                f"\n  Early stopping at epoch {epoch + 1} (patience={PRETRAIN_PATIENCE})"
            )
            break

    # ── Final summary ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Pre-training Complete")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Total time:    {total_time / 60:.1f} minutes")
    print(f"  Checkpoint:    {os.path.join(ckpt_dir, 'rnn_pretrained.pt')}")
    print(f"{'=' * 60}")

    # Final evaluation with best model
    print(f"\n  Loading best model for final evaluation...")
    best_model = CharRNN.load(
        os.path.join(ckpt_dir, "rnn_pretrained.pt"), device=str(device)
    )
    final_samples = best_model.sample(
        args.n_samples, temperature=1.0, device=str(device)
    )
    final_metrics = compute_validity(final_samples)
    print(f"\n  FINAL METRICS (best model):")
    print(f"    Validity:   {final_metrics['validity']:.1%}")
    print(f"    Uniqueness: {final_metrics['uniqueness']:.1%}")

    # Check targets
    targets_met = True
    if final_metrics["validity"] < 0.95:
        print(f"    ⚠ Validity below target (95%)")
        targets_met = False
    if final_metrics["uniqueness"] < 0.98:
        print(f"    ⚠ Uniqueness below target (98%)")
        targets_met = False

    if targets_met:
        print(f"\n  ✓ All pre-training targets met!")
    else:
        print(f"\n  ⚠ Some targets not met — consider training longer")


if __name__ == "__main__":
    main()
