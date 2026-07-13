#!/usr/bin/env python3
"""
train_mobilecnn_1d.py  --  Train MobileCNN-1D and MobileCNN-SE-1D on WEMAC.

Usage
-----
  # Train both models on all 10 folds (default):
  python train_mobilecnn_1d.py

  # Train only one model:
  python train_mobilecnn_1d.py --model mobile1d
  python train_mobilecnn_1d.py --model mobile_se

  # Train a single fold:
  python train_mobilecnn_1d.py --model mobile_se --fold 0

  # Override epochs / learning rate:
  python train_mobilecnn_1d.py --epochs 100 --lr 1e-3

Output
------
  results/deepBindi/mobile1d/   type4_fold0.pth ... type4_fold9.pth
  results/deepBindi/mobile_se/  type4_fold0.pth ... type4_fold9.pth
  Each directory also contains: best_model.pth, fold_metrics.csv, training_log.txt

Architecture
------------
  MobileCNN-1D (Option A):
    Input: (N, 57, 10)
    Block 1: DWConv(57, k=5, pad=1) -> BN -> ReLU
             PWConv(57->32, k=1)    -> BN -> ReLU -> MaxPool(2)
    Block 2: DWConv(32, k=5, pad=1) -> BN -> ReLU
             PWConv(32->64, k=1)    -> BN -> ReLU -> MaxPool(2)
    Head:    Dense(64->1) -> Sigmoid

  MobileCNN-SE-1D (Option B):
    Same as A, plus after Block 1 MaxPool:
    SE: GlobalAvgPool -> Dense(32->8) -> ReLU -> Dense(8->32) -> Sigmoid
        channel-wise multiply

  Both have ~4.9K parameters (vs 19.5K for CNN_1D_v2 baseline).

Requirements
------------
  pip install torch numpy
  Python src path must include src/deep -> src/CH07_DeepLearning junction.
"""

import sys
import os
import argparse
import time
import csv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PY_SRC = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "EPFL_STAY_LAURA", "python_code"
))
_SRC_DIR = os.path.join(_PY_SRC, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    sys.exit(f"Missing dependency: {e}  (pip install numpy torch)")

try:
    from AN02_Databases.prepare_dataset import prepare_file
    from deep.feature_preparation.feature_maps import create_feature_maps
except ImportError as e:
    sys.exit(
        f"Cannot import WEMAC loaders: {e}\n"
        f"  Check that src/deep -> src/CH07_DeepLearning junction exists and\n"
        f"  that the old 'deep' pip package is not installed."
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

_DROP = 0.2
_KSIZE = 5
_PAD   = 1
_POOL  = 2
_SE_RATIO = 4   # reduction: ch1 / SE_RATIO = 8 SE units


class MobileCNN1D(nn.Module):
    """
    Depthwise-Separable 1D CNN (MobileNetV1 style) for WEMAC binary classification.

    Each block: DWConv -> BN -> ReLU -> PWConv -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_ch=57, ch1=32, ch2=64, featmap_resolution=10):
        super().__init__()
        s = 1
        # Block 1 output widths
        w1_dw = (featmap_resolution + 2*_PAD - _KSIZE)//s + 1  # 8
        w1_pw = w1_dw                                            # 8 (k=1 pw)
        w1    = (w1_pw - _POOL)//_POOL + 1                      # 4
        # Block 2 output widths
        w2_dw = (w1 + 2*_PAD - _KSIZE)//s + 1                  # 2
        w2_pw = w2_dw                                            # 2
        w2    = (w2_pw - _POOL)//_POOL + 1                      # 1

        self.final_out = w2 * ch2  # 64

        self.dw1    = nn.Conv1d(in_ch, in_ch, kernel_size=_KSIZE, stride=s,
                                padding=_PAD, groups=in_ch)
        self.bn_dw1 = nn.BatchNorm1d(in_ch)
        self.pw1    = nn.Conv1d(in_ch, ch1, kernel_size=1)
        self.bn_pw1 = nn.BatchNorm1d(ch1)

        self.dw2    = nn.Conv1d(ch1, ch1, kernel_size=_KSIZE, stride=s,
                                padding=_PAD, groups=ch1)
        self.bn_dw2 = nn.BatchNorm1d(ch1)
        self.pw2    = nn.Conv1d(ch1, ch2, kernel_size=1)
        self.bn_pw2 = nn.BatchNorm1d(ch2)

        self.pool    = nn.MaxPool1d(_POOL)
        self.relu    = nn.ReLU()
        self.drop    = nn.Dropout(_DROP)
        self.fc      = nn.Linear(self.final_out, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn_dw1(self.dw1(x)))
        x = self.drop(x)
        x = self.pool(self.relu(self.bn_pw1(self.pw1(x))))
        # Block 2
        x = self.relu(self.bn_dw2(self.dw2(x)))
        x = self.drop(x)
        x = self.pool(self.relu(self.bn_pw2(self.pw2(x))))
        # Head
        x = x.view(-1, self.final_out)
        return self.sigmoid(self.fc(x))


class MobileCNN_SE_1D(nn.Module):
    """
    MobileCNN-SE-1D: same blocks as MobileCNN1D, plus SE channel attention
    after Block 1's MaxPool output.

    SE block: GlobalAvgPool -> Dense(32->8) -> ReLU -> Dense(8->32) -> Sigmoid
              -> channel-wise multiply onto Block 1 output.
    """
    def __init__(self, in_ch=57, ch1=32, ch2=64, featmap_resolution=10):
        super().__init__()
        s = 1
        w1 = ((featmap_resolution + 2*_PAD - _KSIZE)//s + 1 - _POOL)//_POOL + 1  # 4
        w2_dw = ((w1 + 2*_PAD - _KSIZE)//s + 1)                                   # 2
        w2    = (w2_dw - _POOL)//_POOL + 1                                          # 1

        self.final_out = w2 * ch2  # 64
        se_inner = max(1, ch1 // _SE_RATIO)   # 8

        self.dw1    = nn.Conv1d(in_ch, in_ch, kernel_size=_KSIZE, stride=s,
                                padding=_PAD, groups=in_ch)
        self.bn_dw1 = nn.BatchNorm1d(in_ch)
        self.pw1    = nn.Conv1d(in_ch, ch1, kernel_size=1)
        self.bn_pw1 = nn.BatchNorm1d(ch1)

        # SE branch
        self.se_avg  = nn.AdaptiveAvgPool1d(1)
        self.se_fc1  = nn.Linear(ch1, se_inner)
        self.se_fc2  = nn.Linear(se_inner, ch1)

        self.dw2    = nn.Conv1d(ch1, ch1, kernel_size=_KSIZE, stride=s,
                                padding=_PAD, groups=ch1)
        self.bn_dw2 = nn.BatchNorm1d(ch1)
        self.pw2    = nn.Conv1d(ch1, ch2, kernel_size=1)
        self.bn_pw2 = nn.BatchNorm1d(ch2)

        self.pool    = nn.MaxPool1d(_POOL)
        self.relu    = nn.ReLU()
        self.drop    = nn.Dropout(_DROP)
        self.fc      = nn.Linear(self.final_out, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn_dw1(self.dw1(x)))
        x = self.drop(x)
        x = self.pool(self.relu(self.bn_pw1(self.pw1(x))))  # (N, ch1, W/2)

        # SE attention
        se = self.se_avg(x).squeeze(-1)           # (N, ch1)
        se = self.relu(self.se_fc1(se))            # (N, se_inner)
        se = self.sigmoid(self.se_fc2(se))         # (N, ch1), values in [0,1]
        x  = x * se.unsqueeze(-1)                  # channel-wise scale

        # Block 2
        x = self.relu(self.bn_dw2(self.dw2(x)))
        x = self.drop(x)
        x = self.pool(self.relu(self.bn_pw2(self.pw2(x))))
        # Head
        x = x.view(-1, self.final_out)
        return self.sigmoid(self.fc(x))


MODEL_REGISTRY = {
    "mobile1d":  MobileCNN1D,
    "mobile_se": MobileCNN_SE_1D,
}


# ---------------------------------------------------------------------------
# Data loading  (same pipeline as extract_test_inputs.py)
# ---------------------------------------------------------------------------

def load_wemac_folds(data_dir, feat_size=10, n_folds=10):
    """
    Load WEMAC data and return list of (X_train, y_train, X_val, y_val, X_test, y_test)
    tuples, one per fold, all as float32 tensors shaped (N, 57, feat_size).
    """
    print("Loading WEMAC dataset...")
    loader = prepare_file(data_dir, 'wemac_50overlapping')
    print(f"  dataset shape: {loader.dataset.shape}")

    INPUT_RES = (57, feat_size)
    deep_loader = create_feature_maps(loader, feat_size, INPUT_RES)
    c, d, h, w = deep_loader.X_data.size()
    print(f"  feature map tensor: ({c}, {d}, {h}, {w})")

    vol_partitions = loader.CV_LOSO_generator_ids(
        cv_folds=n_folds,
        test_partition=0.3,
        eval_partition=0.1,
    )

    folds = []
    for fold, (train_ids, val_ids, test_ids, _) in enumerate(vol_partitions):
        X_tr, y_tr, _ = deep_loader.select_vol(volunteers_id=train_ids)
        X_va, y_va, _ = deep_loader.select_vol(volunteers_id=val_ids)
        X_te, y_te, _ = deep_loader.select_vol(volunteers_id=test_ids)
        # Squeeze depth dim: (N,1,57,W) -> (N,57,W)
        X_tr = X_tr.squeeze(1)
        X_va = X_va.squeeze(1)
        X_te = X_te.squeeze(1)
        y_tr = y_tr[:, 0, :].squeeze(-1)
        y_va = y_va[:, 0, :].squeeze(-1)
        y_te = y_te[:, 0, :].squeeze(-1)
        folds.append((X_tr, y_tr, X_va, y_va, X_te, y_te))
        print(f"  Fold {fold}: train={X_tr.shape[0]}  val={X_va.shape[0]}  test={X_te.shape[0]}")
    return folds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def f1_score_binary(preds, labels, threshold=0.5):
    """Compute binary F1 from float predictions and float labels."""
    p = (preds > threshold).float()
    l = labels.float()
    tp = (p * l).sum().item()
    fp = (p * (1 - l)).sum().item()
    fn = ((1 - p) * l).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def train_fold(model, X_tr, y_tr, X_va, y_va,
               epochs=150, lr=1e-3, batch_size=64, device="cpu"):
    """
    Train `model` for `epochs` epochs. Returns best val-F1 and the
    corresponding state_dict.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    ds = TensorDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    X_va  = X_va.to(device)
    y_va  = y_va.to(device)

    best_f1    = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va_preds = model(X_va).squeeze(-1)
                va_loss  = criterion(va_preds, y_va).item()
                va_f1    = f1_score_binary(va_preds, y_va)
                va_acc   = ((va_preds > 0.5).float() == y_va).float().mean().item()
            if va_f1 > best_f1:
                best_f1    = va_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ep {ep+1:4d}/{epochs}  val_loss={va_loss:.4f}  "
                  f"val_f1={va_f1:.3f}  val_acc={va_acc:.3f}")

    return best_f1, best_state


def evaluate_fold(model, state_dict, X_te, y_te, device="cpu"):
    """Load state_dict and evaluate on test set. Returns (f1, acc)."""
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)
    with torch.no_grad():
        preds = model(X_te).squeeze(-1)
    f1  = f1_score_binary(preds, y_te)
    acc = ((preds > 0.5).float() == y_te).float().mean().item()
    return f1, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MobileCNN-1D / MobileCNN-SE-1D on WEMAC for X-HEEP port.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all)")
    parser.add_argument("--fold", type=int, default=None,
        help="Train a single fold (0-9); default: all 10 folds")
    parser.add_argument("--epochs", type=int, default=150,
        help="Training epochs per fold (default: 150)")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64,
        help="Batch size (default: 64)")
    parser.add_argument("--device", default="cpu",
        help="PyTorch device string (default: cpu)")
    args = parser.parse_args()

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    data_dir = _PY_SRC + os.sep
    folds = load_wemac_folds(data_dir)

    device = torch.device(args.device)
    fold_range = range(10) if args.fold is None else [args.fold]

    for model_name in models_to_train:
        model_cls = MODEL_REGISTRY[model_name]
        out_dir = os.path.normpath(os.path.join(
            _PY_SRC, "results", "deepBindi", model_name
        ))
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "training_log.txt")
        csv_path = os.path.join(out_dir, "fold_metrics.csv")

        print(f"\n{'='*60}")
        print(f"Training {model_name}  ({args.epochs} epochs, {len(fold_range)} folds)")
        # Count params
        _m = model_cls()
        n_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")
        print(f"Output dir: {out_dir}")
        print(f"{'='*60}")

        all_metrics = []
        best_f1_global = -1.0
        best_fold_idx  = -1

        with open(log_path, "w") as log_fh:
            log_fh.write(f"model={model_name}  params={n_params}  epochs={args.epochs}\n\n")

            for fold in fold_range:
                X_tr, y_tr, X_va, y_va, X_te, y_te = folds[fold]
                print(f"\n-- Fold {fold} --")
                t0 = time.time()

                model = model_cls()
                val_f1, best_state = train_fold(
                    model, X_tr, y_tr, X_va, y_va,
                    epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, device=str(device)
                )

                te_f1, te_acc = evaluate_fold(model_cls(), best_state, X_te, y_te, device=str(device))
                elapsed = time.time() - t0
                print(f"  Fold {fold}: test_f1={te_f1:.3f}  test_acc={te_acc:.3f}  "
                      f"val_best_f1={val_f1:.3f}  time={elapsed:.1f}s")

                log_fh.write(
                    f"fold={fold}  test_f1={te_f1:.4f}  test_acc={te_acc:.4f}  "
                    f"val_f1={val_f1:.4f}  elapsed={elapsed:.1f}s\n"
                )
                log_fh.flush()

                pth_path = os.path.join(out_dir, f"type4_fold{fold}.pth")
                torch.save(best_state, pth_path)
                print(f"  Saved {pth_path}")

                all_metrics.append({
                    "fold": fold, "test_f1": te_f1, "test_acc": te_acc,
                    "val_f1": val_f1,
                })

                if te_f1 > best_f1_global:
                    best_f1_global = te_f1
                    best_fold_idx  = fold

        # Write CSV
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["fold", "test_f1", "test_acc", "val_f1"])
            writer.writeheader()
            writer.writerows(all_metrics)

        # Save best
        if best_fold_idx >= 0:
            import shutil
            best_src = os.path.join(out_dir, f"type4_fold{best_fold_idx}.pth")
            best_dst = os.path.join(out_dir, "best_model.pth")
            shutil.copy(best_src, best_dst)
            print(f"\nBest fold for {model_name}: fold {best_fold_idx}  "
                  f"test_f1={best_f1_global:.3f}")
            print(f"Saved best_model.pth  (copy of type4_fold{best_fold_idx}.pth)")

        if all_metrics:
            mean_f1  = sum(m["test_f1"]  for m in all_metrics) / len(all_metrics)
            mean_acc = sum(m["test_acc"] for m in all_metrics) / len(all_metrics)
            print(f"\n{model_name} summary over {len(all_metrics)} folds:")
            print(f"  mean test F1  = {mean_f1:.3f}")
            print(f"  mean test Acc = {mean_acc:.3f}")

        print(f"\nNext step: python export_mobile_weights.py --model {model_name} "
              f"--pth {os.path.join(out_dir, 'type4_fold0.pth')}")


if __name__ == "__main__":
    main()
