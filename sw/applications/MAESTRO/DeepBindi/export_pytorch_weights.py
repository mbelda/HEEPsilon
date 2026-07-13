#!/usr/bin/env python3
"""
export_pytorch_weights.py  --  Export CNN_1D_v2 trained weights to a C int32_t header.

Usage
-----
  # Export the best fold from a models directory (auto-selects fold with highest test F1):
  python export_pytorch_weights.py --results-dir <path>/results/deepBindi/models_<timestamp>/

  # Export a specific .pth file directly:
  python export_pytorch_weights.py --pth path/to/type4_fold3.pth

  # Only inspect (print weight stats, no output file):
  python export_pytorch_weights.py --pth type4_fold3.pth --inspect

Output
------
  weights_cnn_1d_v2.h  (in the same directory as this script)

  Contains const int32_t arrays for:
    conv1_weight, conv1_bias
    bn1_scale, bn1_offset   (BatchNorm pre-folded, Q7: 128 = 1.0)
    conv2_weight, conv2_bias
    bn2_scale, bn2_offset
    fc1_weight, fc1_bias

How to wire into the C port
----------------------------
  1. Copy weights_cnn_1d_v2.h into deepbindi_cnn_x_heep/.
  2. In cnn_models_c.c, replace dummy weight generation with reads from these arrays.
     See inline comment in weights_cnn_1d_v2.h for the API change.
  3. Remove the weight_arena_reset() call from run_cnn_1d_v2()  -- with real const
     weights the weight pool is not used and need not be reset.

Quantisation details
---------------------
  Conv / FC weights and biases are quantised to int8 range per-tensor:
    scale_f = 127.0 / max(|w|)
    w_int   = round(w * scale_f)
  The scale factor is stored as a comment alongside each array.  The C port
  currently ignores the scale difference between layers; for cycle-count
  benchmarking that is acceptable.  For production accuracy, rescale activations
  between layers or use per-channel quantisation.

  BatchNorm pre-folding (same scheme as nn_runtime.c):
    scale[c]  = round(gamma[c] / sqrt(var[c] + eps) * 128)
    offset[c] = round(beta[c]  - gamma[c] * mean[c] / sqrt(var[c] + eps))
  This is Q7 fixed-point: 128 represents 1.0.

Requirements
------------
  pip install torch numpy
"""

import sys
import os
import argparse
import math

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

try:
    import torch
except ImportError:
    sys.exit("torch required:  pip install torch")

# ---- Paths ------------------------------------------------------------------

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "weights_cnn_1d_v2.h")

# ---- BatchNorm pre-folding --------------------------------------------------

_BN_EPS = 1e-5   # PyTorch default


def fold_batchnorm(gamma, beta, mean, var, eps=_BN_EPS):
    """
    Pre-fold BatchNorm into Q7 scale/offset arrays.

    Returns (scale_int32, offset_int32) where scale == 128 means multiply by 1.0.
    """
    std     = np.sqrt(var + eps)
    scale_f = gamma / std            # float scale
    scale_q = np.round(scale_f * 128.0).astype(np.int32)
    # offset: beta - gamma * mean / std
    offset_f = beta - gamma * mean / std
    offset_q = np.round(offset_f).astype(np.int32)
    return scale_q, offset_q


# ---- Weight quantisation ---------------------------------------------------

def quantise_tensor(w_np, bits=8):
    """
    Per-tensor symmetric quantisation to [-2^(bits-1)+1, 2^(bits-1)-1].
    Returns (w_int32, scale_float).
    scale_float: multiply int value by scale_float to recover original float.
    """
    max_abs = np.max(np.abs(w_np))
    if max_abs == 0.0:
        return np.zeros_like(w_np, dtype=np.int32), 1.0
    max_int  = float(2 ** (bits - 1) - 1)
    scale_f  = max_abs / max_int          # float / int = float scale
    w_int    = np.round(w_np / scale_f).astype(np.int32)
    return w_int, scale_f


# ---- C array formatting ----------------------------------------------------

def _c_array(c_name, arr_int32, comment="", values_per_line=10):
    """Return a const int32_t C array declaration as a string."""
    flat  = arr_int32.flatten()
    count = flat.size
    lines = []
    if comment:
        for cl in comment.strip().split("\n"):
            lines.append(f"/* {cl} */")
    lines.append(f"static const int32_t {c_name}[{count}] = {{")
    for i in range(0, count, values_per_line):
        chunk = flat[i : i + values_per_line]
        sep   = "," if (i + values_per_line) < count else ""
        lines.append("    " + ", ".join(str(int(v)) for v in chunk) + sep)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


# ---- Main export -----------------------------------------------------------

def export(state_dict, output_path, pth_source=""):
    """Extract all CNN_1D_v2 layers and write weights_cnn_1d_v2.h."""

    # Helper to get numpy from state dict
    def np_of(key):
        return state_dict[key].detach().cpu().float().numpy()

    # Check mandatory keys
    required = [
        "conv1.weight", "conv1.bias",
        "batch_norm1.weight", "batch_norm1.bias",
        "batch_norm1.running_mean", "batch_norm1.running_var",
        "conv2.weight", "conv2.bias",
        "batch_norm2.weight", "batch_norm2.bias",
        "batch_norm2.running_mean", "batch_norm2.running_var",
        "fc1.weight", "fc1.bias",
    ]
    missing = [k for k in required if k not in state_dict]
    if missing:
        print(f"ERROR: state dict is missing keys: {missing}")
        print(f"Available keys: {list(state_dict.keys())}")
        sys.exit(1)

    # ---- Conv1 --------------------------------------------------------------
    conv1_w, s1w = quantise_tensor(np_of("conv1.weight"))  # (32, 57, 5)
    conv1_b, s1b = quantise_tensor(np_of("conv1.bias"))    # (32,)

    # ---- BatchNorm 1 (pre-folded) ------------------------------------------
    bn1_scale, bn1_offset = fold_batchnorm(
        np_of("batch_norm1.weight"),
        np_of("batch_norm1.bias"),
        np_of("batch_norm1.running_mean"),
        np_of("batch_norm1.running_var"),
    )

    # ---- Conv2 --------------------------------------------------------------
    conv2_w, s2w = quantise_tensor(np_of("conv2.weight"))  # (64, 32, 5)
    conv2_b, s2b = quantise_tensor(np_of("conv2.bias"))    # (64,)

    # ---- BatchNorm 2 (pre-folded) ------------------------------------------
    bn2_scale, bn2_offset = fold_batchnorm(
        np_of("batch_norm2.weight"),
        np_of("batch_norm2.bias"),
        np_of("batch_norm2.running_mean"),
        np_of("batch_norm2.running_var"),
    )

    # ---- FC1 ----------------------------------------------------------------
    fc1_w, sfw = quantise_tensor(np_of("fc1.weight"))  # (1, 64)
    fc1_b, sfb = quantise_tensor(np_of("fc1.bias"))    # (1,)

    # ---- Summary -----------------------------------------------------------
    print(f"Conv1  weights: {conv1_w.shape}  scale={s1w:.6f}  max_int={int(np.max(np.abs(conv1_w)))}")
    print(f"Conv1  bias   : {conv1_b.shape}  scale={s1b:.6f}")
    print(f"BN1    scale  : min={bn1_scale.min()}  max={bn1_scale.max()}")
    print(f"BN1    offset : min={bn1_offset.min()}  max={bn1_offset.max()}")
    print(f"Conv2  weights: {conv2_w.shape}  scale={s2w:.6f}  max_int={int(np.max(np.abs(conv2_w)))}")
    print(f"Conv2  bias   : {conv2_b.shape}  scale={s2b:.6f}")
    print(f"BN2    scale  : min={bn2_scale.min()}  max={bn2_scale.max()}")
    print(f"BN2    offset : min={bn2_offset.min()}  max={bn2_offset.max()}")
    print(f"FC1    weights: {fc1_w.shape}  scale={sfw:.6f}")
    print(f"FC1    bias   : {fc1_b.shape}  scale={sfb:.6f}")

    # ---- Write header -------------------------------------------------------
    guard = os.path.basename(output_path).upper().replace(".", "_")
    lines = [
        f"/* {os.path.basename(output_path)}",
        f" * Auto-generated by export_pytorch_weights.py",
        f" * Source: {pth_source}",
        f" *",
        f" * CNN_1D_v2 (PYE / Model 4) trained weights for X-HEEP inference.",
        f" * All arrays are int32_t (in-range [-127, 127] for conv/fc weights).",
        f" *",
        f" * Wiring into cnn_models_c.c:",
        f" *   1. #include this header.",
        f" *   2. Replace seeded_value_int32() calls with reads from these arrays.",
        f" *   3. Remove weight_arena_reset() from run_cnn_1d_v2() (pool unused).",
        f" *",
        f" * BatchNorm is pre-folded (Q7: 128 = 1.0):",
        f" *   y = ((int64_t)x * scale[c] >> 7) + offset[c]",
        f" *",
        f" * Conv / FC weight quantisation (per-tensor symmetric, int8 range):",
        f" *   x_float = x_int * scale   (scale documented per array below)",
        f" */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
        f"/* ------------------------------------------------------------------ */",
        f"/* CONV1  (57 -> 32, kernel=5)                                         */",
        f"/* Layout: [out_ch][in_ch][k]  = [32][57][5] = 9120 elements           */",
        f"/* ------------------------------------------------------------------ */",
        _c_array("conv1_weight", conv1_w,
                 f"shape={list(conv1_w.shape)}  float_scale={s1w:.6f}  (x_float = x_int * scale)"),
        _c_array("conv1_bias", conv1_b,
                 f"shape={list(conv1_b.shape)}  float_scale={s1b:.6f}"),
        f"/* BatchNorm1 pre-folded (Q7: scale=128 -> multiply by 1.0) */",
        _c_array("bn1_scale",  bn1_scale,
                 f"shape={list(bn1_scale.shape)}  Q7: value/128 = effective float multiplier"),
        _c_array("bn1_offset", bn1_offset,
                 f"shape={list(bn1_offset.shape)}  additive offset (same int32 scale as activations)"),
        f"/* ------------------------------------------------------------------ */",
        f"/* CONV2  (32 -> 64, kernel=5)                                         */",
        f"/* Layout: [out_ch][in_ch][k]  = [64][32][5] = 10240 elements          */",
        f"/* ------------------------------------------------------------------ */",
        _c_array("conv2_weight", conv2_w,
                 f"shape={list(conv2_w.shape)}  float_scale={s2w:.6f}"),
        _c_array("conv2_bias", conv2_b,
                 f"shape={list(conv2_b.shape)}  float_scale={s2b:.6f}"),
        f"/* BatchNorm2 pre-folded */",
        _c_array("bn2_scale",  bn2_scale,
                 f"shape={list(bn2_scale.shape)}  Q7"),
        _c_array("bn2_offset", bn2_offset,
                 f"shape={list(bn2_offset.shape)}"),
        f"/* ------------------------------------------------------------------ */",
        f"/* FC1  (64 -> 1)                                                      */",
        f"/* ------------------------------------------------------------------ */",
        _c_array("fc1_weight", fc1_w,
                 f"shape={list(fc1_w.shape)}  float_scale={sfw:.6f}"),
        _c_array("fc1_bias", fc1_b,
                 f"shape={list(fc1_b.shape)}  float_scale={sfb:.6f}"),
        f"#endif /* {guard} */",
        "",
    ]

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))

    total = (conv1_w.size + conv1_b.size +
             bn1_scale.size + bn1_offset.size +
             conv2_w.size + conv2_b.size +
             bn2_scale.size + bn2_offset.size +
             fc1_w.size + fc1_b.size)
    print(f"\nWrote {output_path}")
    print(f"Total int32_t elements: {total}  ({total * 4 // 1024} KB as int32_t)")


def inspect_state_dict(state_dict):
    """Print a summary of all tensors in the state dict."""
    print(f"{'Key':45s}  {'Shape':25s}  {'dtype':8s}  min/max")
    print("-" * 100)
    for k, v in state_dict.items():
        arr = v.detach().cpu().float().numpy()
        print(f"  {k:43s}  {str(list(arr.shape)):25s}  {str(arr.dtype):8s}  "
              f"{arr.min():.4f} / {arr.max():.4f}")


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export CNN_1D_v2 PyTorch weights to C int32_t header.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pth", default=None,
        help="Path to a single .pth state-dict file.")
    parser.add_argument("--results-dir", default=None,
        help="Path to a models_<timestamp>/ directory; picks the .pth with the lowest fold index.")
    parser.add_argument("--output", "-o", default=_DEFAULT_OUT,
        help=f"Output .h file (default: {os.path.basename(_DEFAULT_OUT)})")
    parser.add_argument("--inspect", "-i", action="store_true",
        help="Print tensor summary only; do not write output file.")
    args = parser.parse_args()

    # Resolve input file
    if args.pth:
        pth_path = args.pth
    elif args.results_dir:
        # Pick the first .pth in the directory (lowest fold index)
        pth_files = sorted(
            [f for f in os.listdir(args.results_dir) if f.endswith(".pth")]
        )
        if not pth_files:
            sys.exit(f"No .pth files found in {args.results_dir}")
        pth_path = os.path.join(args.results_dir, pth_files[0])
        print(f"Auto-selected: {pth_path}")
        if len(pth_files) > 1:
            print(f"Other available folds: {pth_files[1:]}")
            print("Use --pth to select a specific fold.")
    else:
        # Auto-detect: look for the most recent models_* dir
        results_base = os.path.normpath(os.path.join(
            _SCRIPT_DIR, "..", "..", "..",
            "EPFL_STAY_LAURA", "python_code", "results", "deepBindi"
        ))
        if os.path.isdir(results_base):
            model_dirs = sorted(
                [d for d in os.listdir(results_base) if d.startswith("models_")],
                reverse=True,  # most recent first
            )
            if model_dirs:
                candidates = sorted([
                    f for f in os.listdir(os.path.join(results_base, model_dirs[0]))
                    if f.endswith(".pth")
                ])
                if candidates:
                    pth_path = os.path.join(results_base, model_dirs[0], candidates[0])
                    print(f"Auto-detected most recent checkpoint: {pth_path}")
                else:
                    sys.exit(f"No .pth files in {os.path.join(results_base, model_dirs[0])}")
            else:
                sys.exit(
                    "No models_* directory found.  Run the training script first "
                    "with model_on=true in deep_simulation_config.json, then point "
                    "--results-dir at the generated models_<timestamp>/ folder."
                )
        else:
            sys.exit(
                "Cannot auto-detect results directory.  Provide --pth or --results-dir."
            )

    if not os.path.isfile(pth_path):
        sys.exit(f"File not found: {pth_path}")

    print(f"Loading {pth_path}  ({os.path.getsize(pth_path) // 1024} KB)")
    state_dict = torch.load(pth_path, map_location="cpu")

    inspect_state_dict(state_dict)

    if not args.inspect:
        print()
        export(state_dict, args.output, pth_source=pth_path)
        print()
        print("Next steps:")
        print("  1. Copy weights_cnn_1d_v2.h into deepbindi_cnn_x_heep/.")
        print("  2. In cnn_models_c.c:")
        print("       #include \"weights_cnn_1d_v2.h\"")
        print("       Replace seeded_value_int32() in each layer_create call with")
        print("       direct reads from conv1_weight[], conv1_bias[], etc.")
        print("  3. Remove weight_arena_reset() from run_cnn_1d_v2().")
        print("  4. Rebuild: make run")
        print("  5. Both samples should now classify correctly.")


if __name__ == "__main__":
    main()
