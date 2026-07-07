"""
EEG Binary Classification Pipeline — Data Loading
====================================================
Dataset:  Ofner2017 (motor imagery session)
Task:     Elbow intent (1) vs. everything else (0)

Output shape to model:  (batch, 19, 200)
    - 19 channels  (10-20 montage subset)
    - 200 points   (1s MI window at 200 Hz; CBraMod patches this internally)

Reads .gdf files from dataset/ (see DATASET_DIR below) — never downloads.
Place all Ofner2017 .gdf files there before running this module.

See model.py for model construction and train.py for the training loop.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import mne
mne.set_log_level("WARNING")

DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def _read_local_gdf(url, sign, path=None, force_update=False, verbose=None):
    """
    Drop-in replacement for moabb.datasets.download.data_dl.

    moabb normally fetches Ofner2017 .gdf files from Zenodo on demand; this
    reads them from DATASET_DIR instead and never touches the network.
    """
    fname = url.rsplit("/", 1)[-1]
    local_path = DATASET_DIR / fname
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset file: {local_path}\n"
            f"Expected all Ofner2017 .gdf files under {DATASET_DIR}"
        )
    return str(local_path)


def _patch_moabb_to_read_local_dataset():
    from moabb.datasets import download as moabb_download

    moabb_download.data_dl = _read_local_gdf


_patch_moabb_to_read_local_dataset()


def load_ofner2017_binary():
    """
    Load Ofner2017, extract the 1s MI window (2-3s in the raw trial),
    resample to 200 Hz, select 19 channels matching the 10-20 montage,
    and relabel into binary: elbow intent (1) vs everything else (0).

    Returns
    -------
    X : np.ndarray, shape (n_trials, 19, 200)
        EEG ready for braindecode's CBraMod.
    y : np.ndarray, shape (n_trials,)
        Binary labels: 1 = elbow intent, 0 = rest/other movement.
    """
    from moabb.datasets import Ofner2017
    from moabb.paradigms import MotorImagery

    # Load the MI session only (session 2 — imagined movements)
    dataset = Ofner2017(imagined=True, executed=False)

    # The dataset's internal interval starts at the cue (t=2s in raw trial).
    # tmin=0.0 means "cue onset", tmax=1.0 means "1s after cue" → the 2-3s window.
    paradigm = MotorImagery(
        n_classes=7,
        fmin=8, fmax=35,        # mu + beta bands
        tmin=0.0, tmax=1.0,     # 1s MI window (relative to cue onset)
        resample=200,           # resample to 200 Hz
    )

    # Load all 15 subjects
    subjects = list(range(1, 16))
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)

    print(f"Raw loaded: X.shape={X.shape}, classes={np.unique(y)}")

    # --- Select 19 channels matching the 10-20 montage ---
    # CBraMod was pretrained on 19 channels in this order.
    ten_twenty_channels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2",
    ]

    # Get actual channel names from the paradigm
    # (load one subject's epochs to read channel names)
    epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
    all_ch_names = list(epochs.ch_names)

    ch_indices = []
    matched_names = []
    for ch in ten_twenty_channels:
        if ch in all_ch_names:
            ch_indices.append(all_ch_names.index(ch))
            matched_names.append(ch)
        else:
            print(f"  Warning: channel '{ch}' not found in Ofner2017, skipping")

    print(f"Matched {len(ch_indices)}/{len(ten_twenty_channels)} channels: {matched_names}")

    X = X[:, ch_indices, :]  # (n_trials, n_matched_channels, 200)

    # --- Relabel into binary ---
    elbow_classes = {"right_elbow_flexion", "right_elbow_extension"}
    y_binary = np.array([1 if label in elbow_classes else 0 for label in y])

    n_elbow = np.sum(y_binary == 1)
    n_other = np.sum(y_binary == 0)
    print(f"Binary labels: {n_elbow} elbow intent, {n_other} other/rest")

    # braindecode's CBraMod expects (n_trials, n_channels, n_times) and does
    # its own internal patching (via patch_size), so no extra segment axis.
    print(f"Final X shape: {X.shape}")  # (n_trials, 19, 200)
    return X, y_binary


def make_dataloaders(X, y, batch_size=64, test_size=0.2, seed=42):
    """
    Z-score normalize, split into train/val, and wrap in DataLoaders.
    """
    # Per-trial, per-channel z-score normalization
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=seed
    )

    print(f"Train: {X_train.shape[0]} trials | Val: {X_val.shape[0]} trials")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    X, y = load_ofner2017_binary()
    train_loader, val_loader = make_dataloaders(X, y, batch_size=64)

    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Input shape:  {sample_batch.shape}")   # (64, 19, 200)
    print(f"Label shape:  {sample_labels.shape}")  # (64,)