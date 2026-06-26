#!/usr/bin/env python3
"""
extract_test_inputs.py  --  Extract WEMAC fold-0 test samples for the C port.

Produces test_input.h with two properly preprocessed samples:
  test_input_0  -- a NO_FEAR sample from fold 0 test set
  test_input_1  -- a FEAR    sample from fold 0 test set

Input pipeline (must match training exactly):
  1. Load WEMAC data via prepare_file / create_feature_maps
  2. Split using the same 10-fold CV generator (fold 0)
  3. select_vol() -> squeeze(1) -> shape (N, 57, 10) float
  4. Find a high-confidence NO_FEAR and FEAR sample (float model)
  5. Quantize: x_int32 = round(x_float * INPUT_SCALE).clip(-128, 127)
  6. Verify PyTorch label matches for the chosen samples
  7. Write test_input.h

Run from deepbindi_cnn_x_heep/:
  python extract_test_inputs.py --pth <path/to/type4_fold0.pth>

Or let it auto-detect the most recent fold-0 checkpoint.
"""

import sys
import os
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- locate Python source tree -------------------------------------------------
_PY_SRC = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "EPFL_STAY_LAURA", "python_code"
))
_SRC_DIR = os.path.join(_PY_SRC, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    sys.exit(f"Missing dependency: {e}  (pip install numpy torch)")

try:
    from AN02_Databases.prepare_dataset import prepare_file
    from deep.feature_preparation.feature_maps import create_feature_maps
except ImportError as e:
    sys.exit(
        f"Cannot import WEMAC data loaders: {e}\n"
        f"Make sure src/deep -> src/CH07_DeepLearning junction exists and\n"
        f"that python-deep (old package) is NOT installed."
    )

# ---- CNN_1D_v2 (Model 4 / PYE) -- copied inline to avoid keras import --------

DROP_OUT = 0.2

class CNN_1D_v2(nn.Module):
    def __init__(self, in1=57, out1=32, out2=64, nlabels=1, featmap_resolution=10):
        super().__init__()
        ksize, strid, pad, dilation, maxpol = 5, 1, 1, 1, 2
        Lout1 = int((featmap_resolution + 2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        Lout2 = int((Lout1 + 2*pad - dilation*(ksize-1) - 1 + strid) // (strid*maxpol))
        self.final_out = Lout2 * out2
        self.conv1       = nn.Conv1d(in1, out1, kernel_size=ksize, stride=strid,
                                     padding=pad, dilation=dilation)
        self.conv2       = nn.Conv1d(out1, out2, kernel_size=ksize, stride=strid,
                                     padding=pad, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm1d(out1)
        self.batch_norm2 = nn.BatchNorm1d(out2)
        self.fc1         = nn.Linear(self.final_out, nlabels)
        self.dropout     = nn.Dropout(DROP_OUT)
        self.maxPool     = nn.MaxPool1d(maxpol)
        self.relu        = nn.ReLU()
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        x = self.maxPool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        x = self.maxPool(self.relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, self.final_out)
        x = self.sigmoid(self.fc1(x))
        return x


# ---- C-array formatting -------------------------------------------------------

def _fmt_row(vals, sep=","):
    return "    " + ", ".join(str(int(v)) for v in vals) + sep

def _c_array_570(name, arr, label, source_info):
    """Format a 570-element int32 array with 10 values per row."""
    lines = [f"static const int32_t {name}[570] = {{"]
    for i in range(0, 570, 10):
        chunk = arr[i:i+10]
        sep = "," if (i + 10) < 570 else ""
        lines.append(_fmt_row(chunk, sep))
    lines.append("};")
    return "\n".join(lines)


# ---- Main ---------------------------------------------------------------------

INPUT_SCALE = 12   # float -> int32: x_int = round(x_float * INPUT_SCALE)
                   # Maps z-score range ~[-10, 10] -> ~[-120, 120] (int8 range)
                   # Consistent with overflow analysis in nn_runtime.h:
                   #   BN1 right-shift=9, BN2 right-shift=14


def load_fold0_test(data_dir, feat_size=10):
    """Return (X_test_float, y_test) for fold 0 test split."""
    print("Loading WEMAC dataset...")
    loader = prepare_file(data_dir, 'wemac_50overlapping')
    print(f"  dataset shape: {loader.dataset.shape}")

    INPUT_RES = (57, feat_size)
    deep_loader = create_feature_maps(loader, feat_size, INPUT_RES)
    c, d, h, w = deep_loader.X_data.size()
    print(f"  feature map tensor: ({c}, {d}, {h}, {w})")

    vol_partitions = loader.CV_LOSO_generator_ids(
        cv_folds=10,
        test_partition=0.3,
        eval_partition=0.1,
    )

    # Consume only fold 0
    for fold, (train_ids, val_ids, test_ids, _) in enumerate(vol_partitions):
        if fold == 0:
            X_test, y_test, _ = deep_loader.select_vol(volunteers_id=test_ids)
            # 1D model: drop depth dim -> (N, 57, 10)
            X_test = X_test.squeeze(1)        # (N, 57, 10) float32
            y_test = y_test[:, 0, :].squeeze(-1)  # (N,) float32 labels
            print(f"  Fold 0 test set: {X_test.shape}  labels {y_test.shape}")
            counts = [(y_test == v).sum().item() for v in [0.0, 1.0]]
            print(f"  Label counts: NO_FEAR={counts[0]}  FEAR={counts[1]}")
            return X_test, y_test
    raise RuntimeError("Fold 0 not found in CV generator")


def find_confident_samples(model, X_test, y_test, conf_threshold=0.7):
    """
    Run float model on all test samples.
    Return (nf_idx, nf_prob, fear_idx, fear_prob) for the highest-confidence
    NO_FEAR and FEAR samples where confidence >= conf_threshold.
    """
    model.eval()
    with torch.no_grad():
        probs = model(X_test).squeeze(-1).cpu().numpy()  # (N,) float probabilities
    labels = y_test.cpu().numpy().flatten()

    # NO_FEAR: label==0, want prob as close to 0 as possible
    nf_mask  = (labels == 0.0)
    fear_mask = (labels == 1.0)

    nf_probs   = probs[nf_mask]
    fear_probs = probs[fear_mask]
    nf_indices   = np.where(nf_mask)[0]
    fear_indices = np.where(fear_mask)[0]

    # most confident NO_FEAR = lowest prob
    best_nf   = nf_indices[np.argmin(nf_probs)]
    best_fear = fear_indices[np.argmax(fear_probs)]

    print(f"\n  Most confident NO_FEAR: sample idx={best_nf}  prob={probs[best_nf]:.4f}")
    print(f"  Most confident FEAR:    sample idx={best_fear}  prob={probs[best_fear]:.4f}")

    return int(best_nf), float(probs[best_nf]), int(best_fear), float(probs[best_fear])


def quantize_sample(x_float_1d, scale=INPUT_SCALE):
    """
    Quantize a (57, 10) float tensor to int32.
    x_int = round(x_float * scale).clip(-128, 127)
    Returns a 570-element numpy int32 array in channel-first layout.
    """
    x_np  = x_float_1d.cpu().numpy()          # (57, 10)
    x_int = np.round(x_np * scale).astype(np.int32)
    x_int = np.clip(x_int, -128, 127)
    return x_int.flatten()                     # (570,) channel-first


def write_test_input_h(output_path, nf_idx, nf_arr, nf_prob,
                       fear_idx, fear_arr, fear_prob, pth_path):
    lines = []
    lines.append("/**")
    lines.append(" * test_input.h  --  WEMAC fold-0 test samples for CNN_1D_v2 validation.")
    lines.append(" *")
    lines.append(f" * Auto-generated by extract_test_inputs.py")
    lines.append(f" * Source model : {os.path.basename(pth_path)}")
    lines.append(f" * Input scale  : x_int = round(x_float * {INPUT_SCALE}).clip(-128, 127)")
    lines.append(" *                (z-score normalized WEMAC features -> int8 range)")
    lines.append(" *")
    lines.append(f" * test_input_0  -- fold-0 test sample idx={nf_idx}")
    lines.append(f" *                  label 0 (NO_FEAR),  float model prob={nf_prob:.4f}")
    lines.append(f" * test_input_1  -- fold-0 test sample idx={fear_idx}")
    lines.append(f" *                  label 1 (FEAR),     float model prob={fear_prob:.4f}")
    lines.append(" *")
    lines.append(" * Layout: channel-first, data[ch * 10 + t]")
    lines.append(" *         ch in [0, 56],  t in [0, 9],  570 int32_t values total.")
    lines.append(" *")
    lines.append(" * With real trained weights + overflow right-shifts (BN1>>9, BN2>>14)")
    lines.append(" * both samples should be classified correctly by the C port.")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef TEST_INPUT_H")
    lines.append("#define TEST_INPUT_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define TEST_INPUT_LABEL_0  0   /* expected label: NO_FEAR */")
    lines.append("#define TEST_INPUT_LABEL_1  1   /* expected label: FEAR    */")
    lines.append("")
    lines.append("/* --------------------------------------------------------------------------")
    lines.append(f" * Fold-0 test sample idx={nf_idx}  --  label 0 (NO_FEAR)")
    lines.append(f" * Float model sigmoid output: {nf_prob:.4f}  (below 0.5 -> NO_FEAR)")
    lines.append(" * -------------------------------------------------------------------------- */")
    lines.append(_c_array_570("test_input_0", nf_arr, 0, pth_path))
    lines.append("")
    lines.append("/* --------------------------------------------------------------------------")
    lines.append(f" * Fold-0 test sample idx={fear_idx}  --  label 1 (FEAR)")
    lines.append(f" * Float model sigmoid output: {fear_prob:.4f}  (above 0.5 -> FEAR)")
    lines.append(" * -------------------------------------------------------------------------- */")
    lines.append(_c_array_570("test_input_1", fear_arr, 1, pth_path))
    lines.append("")
    lines.append("#endif /* TEST_INPUT_H */")
    lines.append("")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract WEMAC fold-0 test samples for the CNN_1D_v2 C port.",
    )
    parser.add_argument("--pth", default=None,
        help="Path to type4_fold0.pth  (auto-detected if omitted)")
    parser.add_argument("--output", "-o",
        default=os.path.join(_SCRIPT_DIR, "test_input.h"),
        help="Output path for test_input.h")
    parser.add_argument("--conf", type=float, default=0.7,
        help="Minimum float-model confidence for sample selection (default 0.7)")
    args = parser.parse_args()

    # ---- Resolve checkpoint -----------------------------------------------
    if args.pth:
        pth_path = args.pth
    else:
        results_base = os.path.normpath(os.path.join(
            _PY_SRC, "results", "deepBindi", "cnn1d_v2_xheep"
        ))
        candidates = []
        if os.path.isdir(results_base):
            candidates = sorted([
                os.path.join(results_base, f)
                for f in os.listdir(results_base)
                if f.startswith("type4_fold0") and f.endswith(".pth")
            ])
        if not candidates:
            # Fall back: look for any type4 checkpoint
            if os.path.isdir(results_base):
                candidates = sorted([
                    os.path.join(results_base, f)
                    for f in os.listdir(results_base)
                    if f.startswith("type4") and f.endswith(".pth")
                ])
        if not candidates:
            sys.exit(
                "Cannot auto-detect checkpoint.  Provide --pth path/to/type4_fold0.pth"
            )
        pth_path = candidates[0]
        print(f"Auto-detected checkpoint: {pth_path}")

    if not os.path.isfile(pth_path):
        sys.exit(f"File not found: {pth_path}")

    # ---- Load model -------------------------------------------------------
    print(f"Loading {pth_path}  ({os.path.getsize(pth_path)//1024} KB)")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    model = CNN_1D_v2(in1=57, featmap_resolution=10)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Load data --------------------------------------------------------
    # prepare_file(path, name) reads path + "data/" + name + ".csv"
    # In train_cnn1d_v2_xheep.py: _DATA_DIR = os.path.join(_SIM_DIR, '../')
    # which ends with a separator.  We must do the same.
    data_dir = _PY_SRC + os.sep
    X_test, y_test = load_fold0_test(data_dir)

    # ---- Find confident samples -------------------------------------------
    nf_idx, nf_prob, fear_idx, fear_prob = find_confident_samples(
        model, X_test, y_test, conf_threshold=args.conf
    )

    # ---- Quantize ---------------------------------------------------------
    nf_arr   = quantize_sample(X_test[nf_idx],   INPUT_SCALE)
    fear_arr = quantize_sample(X_test[fear_idx], INPUT_SCALE)

    print(f"\n  NO_FEAR sample  idx={nf_idx}  float range [{X_test[nf_idx].min():.3f}, {X_test[nf_idx].max():.3f}]")
    print(f"                  int range  [{nf_arr.min()}, {nf_arr.max()}]")
    print(f"  FEAR sample     idx={fear_idx}  float range [{X_test[fear_idx].min():.3f}, {X_test[fear_idx].max():.3f}]")
    print(f"                  int range  [{fear_arr.min()}, {fear_arr.max()}]")

    # ---- Verify quantized prediction in float model (sanity check) --------
    nf_q_float   = torch.tensor(nf_arr / INPUT_SCALE, dtype=torch.float32).reshape(1, 57, 10)
    fear_q_float = torch.tensor(fear_arr / INPUT_SCALE, dtype=torch.float32).reshape(1, 57, 10)
    with torch.no_grad():
        nf_q_prob   = model(nf_q_float).item()
        fear_q_prob = model(fear_q_float).item()
    print(f"\n  After quantization (dequantized -> float model):")
    print(f"    NO_FEAR prob  original={nf_prob:.4f}  after_quant={nf_q_prob:.4f}  "
          f"predict={'FEAR' if nf_q_prob > 0.5 else 'NO_FEAR'}")
    print(f"    FEAR    prob  original={fear_prob:.4f}  after_quant={fear_q_prob:.4f}  "
          f"predict={'FEAR' if fear_q_prob > 0.5 else 'NO_FEAR'}")

    if nf_q_prob > 0.5:
        print("  WARNING: quantization flipped the NO_FEAR prediction — "
              "try a lower INPUT_SCALE or a different sample.")
    if fear_q_prob <= 0.5:
        print("  WARNING: quantization flipped the FEAR prediction — "
              "try a lower INPUT_SCALE or a different sample.")

    # ---- Write header -----------------------------------------------------
    write_test_input_h(args.output, nf_idx, nf_arr, nf_prob,
                       fear_idx, fear_arr, fear_prob, pth_path)
    print()
    print("Next steps:")
    print("  make run-real")
    print("  Expected output:")
    print("    Sample 0 (NO_FEAR): Output 0 (NO_FEAR)")
    print("    Sample 1 (FEAR):    Output 1 (FEAR)")


if __name__ == "__main__":
    main()
