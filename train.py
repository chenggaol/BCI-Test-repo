"""
Train CBraMod on the Ofner2017 elbow-intent binary classification task.

Usage
-----
    python train.py --n_chans 19 --n_outputs 2 --epochs 20 --batch_size 64
"""

import argparse

import torch
import torch.nn as nn

from dataload import load_ofner2017_binary, make_dataloaders
from model import build_model, PRETRAINED_REPO_ID


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_chans", type=int, default=19,
                        help="Number of EEG channels fed to the model")
    parser.add_argument("--n_outputs", type=int, default=2,
                        help="Number of target classes")
    parser.add_argument("--n_times", type=int, default=200,
                        help="Time samples per input window (n_segments * patch_size)")
    parser.add_argument("--pretrained_repo_id", type=str, default=PRETRAINED_REPO_ID,
                        help="Hugging Face Hub repo to load pretrained CBraMod weights from")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze everything except the classification head")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_correct, n_total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            n_correct += (logits.argmax(dim=-1) == y_batch).sum().item()
            n_total += X_batch.size(0)

    return total_loss / n_total, n_correct / n_total


def main():
    args = parse_args()

    X, y = load_ofner2017_binary()
    train_loader, val_loader = make_dataloaders(
        X, y, batch_size=args.batch_size, test_size=args.test_size, seed=args.seed
    )

    model, device = build_model(
        n_chans=args.n_chans,
        n_outputs=args.n_outputs,
        n_times=args.n_times,
        pretrained_repo_id=args.pretrained_repo_id,
        freeze_backbone=args.freeze_backbone,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
