#!/usr/bin/env python3
"""
export_mobile_weights.py  --  Export MobileCNN-1D / MobileCNN-SE-1D trained weights
                               to C int32_t headers for the X-HEEP int32 port.

Usage
-----
  # Export MobileCNN-1D (Option A):
  python export_mobile_weights.py --model mobile1d --pth path/to/type4_fold0.pth

  # Export MobileCNN-SE-1D (Option B):
  python export_mobile_weights.py --model mobile_se --pth path/to/type4_fold0.pth

  # Auto-detect most recent fold-0 checkpoint:
  python export_mobile_weights.py --model mobile1d

Output
------
  weights_mobile1d.h    (for --model mobile1d)
  weights_mobile_se.h   (for --model mobile_se)

  Each header contains:
    - const int32_t arrays for all layer weights / biases
    - Pre-folded BatchNorm as Q7 scale + offset
    - #define MOBILE_SHIFT_* for overflow-preventing right-shifts
    - For mobile_se: additional SE Dense arrays and #define MOBILE_SE_*

Overflow analysis
-----------------
  The MOBILE_SHIFT_* values are computed from the actual trained BN scale ranges
  using a conservative worst-case accumulator analysis. See inline comments.
  The C forward pass calls tensor_rshift_inplace(x, MOBILE_SHIFT_*) after each
  BN+ReLU to keep accumulators within INT32_MAX for the next layer.
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PY_SRC = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "EPFL_STAY_LAURA", "python_code"
))

_BN_EPS = 1e-5

# ---------------------------------------------------------------------------
# Quantisation helpers (same as export_pytorch_weights.py)
# ---------------------------------------------------------------------------

def fold_batchnorm(gamma, beta, mean, var, eps=_BN_EPS):
    std     = np.sqrt(var + eps)
    scale_f = gamma / std
    scale_q = np.round(scale_f * 128.0).astype(np.int32)
    offset_q = np.round(beta - gamma * mean / std).astype(np.int32)
    return scale_q, offset_q


def quantise_tensor(w_np, bits=8):
    max_abs = np.max(np.abs(w_np))
    if max_abs == 0.0:
        return np.zeros_like(w_np, dtype=np.int32), 1.0
    max_int = float(2 ** (bits - 1) - 1)
    scale_f = max_abs / max_int
    w_int   = np.round(w_np / scale_f).astype(np.int32)
    return w_int, scale_f


def _c_array(c_name, arr_int32, comment="", values_per_line=10):
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


# ---------------------------------------------------------------------------
# Overflow-aware right-shift computation
# ---------------------------------------------------------------------------

INT32_MAX = 2147483647


# ---------------------------------------------------------------------------
# Data-driven calibration helpers
# ---------------------------------------------------------------------------

def load_test_inputs_from_header(header_path):
    """
    Parse test_input.h and return (sample0, sample1) as (57,10) int32 arrays.
    Returns (None, None) if the file cannot be parsed.
    """
    import re
    try:
        with open(header_path, 'r') as f:
            text = f.read()
    except OSError:
        return None, None

    pattern = re.compile(
        r'static const int32_t (test_input_\d+)\[\d+\]\s*=\s*\{([^}]+)\}',
        re.DOTALL)
    arrays = {}
    for m in pattern.finditer(text):
        name = m.group(1)
        vals = [int(x.strip()) for x in m.group(2).split(',') if x.strip()]
        if len(vals) == 570:
            arrays[name] = np.array(vals, dtype=np.int32).reshape(57, 10)

    s0 = arrays.get('test_input_0')
    s1 = arrays.get('test_input_1')
    return s0, s1


def simulate_block1_pool1_max(sample_int32,
                               dw1_w, dw1_b, bn_dw1_sc, bn_dw1_off, shift_dw1,
                               pw1_w, pw1_b, bn_pw1_sc, bn_pw1_off, shift_pw1):
    """
    Run int32 Block 1 (DW1+BN+relu+rshift -> PW1+BN+relu+rshift -> MaxPool)
    on a single (57, 10) int32 input and return the (32, 4) int32 pool1 tensor.

    Mirrors the C forward pass exactly: int64_t BN intermediate, arithmetic rshift.
    """
    C_dw, W_in = 57, 10
    K, pad = 5, 1
    W_dw = (W_in + 2 * pad - K) + 1   # = 8

    # ---- DW1: depthwise conv, groups=57, k=5, pad=1 ----
    x_pad = np.pad(sample_int32, ((0, 0), (pad, pad)), mode='constant')  # (57,12)
    w_dw1 = dw1_w.reshape(C_dw, K)                                        # (57, 5)
    dw1_out = np.zeros((C_dw, W_dw), dtype=np.int64)
    for c in range(C_dw):
        for t in range(W_dw):
            dw1_out[c, t] = (
                int(np.dot(x_pad[c, t:t+K].astype(np.int64),
                           w_dw1[c].astype(np.int64)))
                + int(dw1_b[c])
            )
    dw1_out = np.clip(dw1_out, -(2**31), 2**31 - 1).astype(np.int32)

    # ---- BN_DW1 rshift + ReLU ----
    bn_dw1 = np.zeros((C_dw, W_dw), dtype=np.int64)
    for c in range(C_dw):
        bn_dw1[c] = (dw1_out[c].astype(np.int64) * int(bn_dw1_sc[c])) >> 7
        bn_dw1[c] += int(bn_dw1_off[c])
        bn_dw1[c] >>= shift_dw1
    bn_dw1 = np.maximum(bn_dw1, 0).astype(np.int32)

    # ---- PW1: pointwise conv, (57->32), k=1 ----
    pw1_w_2d = pw1_w.reshape(32, 57)
    pw1_out = pw1_w_2d.astype(np.int64) @ bn_dw1.astype(np.int64)   # (32, 8)
    pw1_out += pw1_b.reshape(-1, 1).astype(np.int64)
    pw1_out = np.clip(pw1_out, -(2**31), 2**31 - 1).astype(np.int32)

    # ---- BN_PW1 rshift + ReLU ----
    bn_pw1 = np.zeros((32, W_dw), dtype=np.int64)
    for c in range(32):
        bn_pw1[c] = (pw1_out[c].astype(np.int64) * int(bn_pw1_sc[c])) >> 7
        bn_pw1[c] += int(bn_pw1_off[c])
        bn_pw1[c] >>= shift_pw1
    bn_pw1 = np.maximum(bn_pw1, 0).astype(np.int32)

    # ---- MaxPool(k=2, s=2): 8 -> 4 ----
    pool1 = np.maximum(bn_pw1[:, 0::2], bn_pw1[:, 1::2])   # (32, 4)
    return pool1.astype(np.int32)


def compute_shift_to_fit(max_value, next_macs, next_weight_max=127,
                         bn_scale_max=None, safety_margin=2):
    """
    Compute the minimum right-shift S such that:
      (max_value >> S) * next_macs * next_weight_max < INT32_MAX

    If bn_scale_max is given, also ensures that after BN:
      ((max_value >> S) * bn_scale_max) >> 7  < INT32_MAX

    safety_margin: add this many extra bits to avoid edge cases.
    Returns int shift (0 if no shift needed).
    """
    for S in range(0, 32):
        reduced = max_value >> S
        acc = reduced * next_macs * next_weight_max
        ok = acc < INT32_MAX
        if bn_scale_max is not None:
            bn_out = (reduced * bn_scale_max) >> 7
            ok = ok and bn_out < INT32_MAX
        if ok:
            return max(0, S + safety_margin)
    return 31


def compute_all_shifts(bn_scales, actual_pool1_max=None, pool1_safety=8):
    """
    Compute MOBILE_SHIFT_* values from a dict of BN scale arrays.

    Keys expected: 'bn_dw1', 'bn_pw1', 'bn_dw2', 'bn_pw2'
    Returns dict:
      'dw1'      -> int  (global shift after BN_DW1)
      'pw1'      -> int  (global shift after BN_PW1)
      'dw2_per'  -> np.ndarray int32 (per-channel shifts after BN_DW2)
      'pw2'      -> int  (global shift after BN_PW2)

    SHIFT_DW1 and SHIFT_PW1 are calibrated from theoretical worst-case inputs
    to guarantee no int32_t overflow in the conv accumulators.

    SHIFT_DW2 (per-channel) and SHIFT_PW2 are calibrated from actual_pool1_max
    when provided (data-driven), otherwise from the theoretical V2.  Data-driven
    calibration is strongly preferred: the theoretical V2 is ~1000x the observed
    pool1 maximum for real biosignal data, causing signal collapse in DW2 when
    using theoretical shifts.

    actual_pool1_max: actual max |pool1| value measured by simulate_block1_pool1_max().
    pool1_safety: multiply actual_pool1_max by this factor before computing DW2
                  shifts, to provide headroom for samples beyond the calibration set.
                  Default 8x (3 extra bits).
    """
    S_dw1_arr = bn_scales['bn_dw1']   # per-channel Q7 scales (numpy array)
    S_pw1_arr = bn_scales['bn_pw1']
    S_dw2_arr = bn_scales['bn_dw2']
    S_pw2_arr = bn_scales['bn_pw2']

    S_dw1 = int(S_dw1_arr.max())
    S_pw1 = int(S_pw1_arr.max())
    S_dw2 = int(S_dw2_arr.max())
    S_pw2 = int(S_pw2_arr.max())

    print(f"  BN scales (max Q7): dw1={S_dw1} pw1={S_pw1} dw2={S_dw2} pw2={S_pw2}")

    max_input = 128
    max_w     = 127

    # --- SHIFT_DW1 (global, theoretical) ---
    max_dw1    = 5 * max_w * max_input           # 80,960
    max_bn_dw1 = (max_dw1 * S_dw1) >> 7

    shift_dw1 = 0
    for S in range(0, 32):
        v = max_bn_dw1 >> S
        if v < INT32_MAX and (57 * max_w * v) < INT32_MAX:
            shift_dw1 = S
            break
    V1 = max_bn_dw1 >> shift_dw1
    print(f"  SHIFT_DW1={shift_dw1}:  after_BN_DW1={max_bn_dw1} >> {shift_dw1} = {V1}")

    # --- SHIFT_PW1 (global, theoretical) ---
    max_pw1    = 57 * max_w * V1
    max_bn_pw1 = (max_pw1 * S_pw1) >> 7

    shift_pw1 = 0
    for S in range(0, 32):
        v = max_bn_pw1 >> S
        max_dw2_check = 5 * max_w * v
        if v < INT32_MAX and max_dw2_check < INT32_MAX:
            shift_pw1 = S
            break
    V2_theoretical = max_bn_pw1 >> shift_pw1
    print(f"  SHIFT_PW1={shift_pw1}:  after_BN_PW1={max_bn_pw1} >> {shift_pw1} = {V2_theoretical}")

    # --- DW2 calibration anchor ---
    # If actual pool1 max is known from simulation, use it * safety factor.
    # Otherwise fall back to theoretical V2 (very conservative).
    if actual_pool1_max is not None:
        V2_cal = actual_pool1_max * pool1_safety
        print(f"  Data-driven calibration: actual_pool1_max={actual_pool1_max}  "
              f"x{pool1_safety} safety -> V2_cal={V2_cal:,}  "
              f"(theoretical V2={V2_theoretical:,}, ratio={V2_theoretical//max(1,V2_cal)}x)")
    else:
        V2_cal = V2_theoretical
        print(f"  WARNING: no actual_pool1_max provided; using theoretical V2={V2_cal:,}."
              f"  DW2 shifts will be over-conservative (signal collapse risk).")

    # --- SHIFT_DW2 (per-channel, data-driven) ---
    max_dw2 = 5 * max_w * V2_cal
    V3_MAX  = INT32_MAX // (32 * max_w)        # ~528K

    shifts_dw2 = np.zeros(len(S_dw2_arr), dtype=np.int32)
    for c, s_c in enumerate(S_dw2_arr):
        max_bn_dw2_c = (max_dw2 * int(s_c)) >> 7
        shift_c = 0
        for S in range(0, 32):
            v = max_bn_dw2_c >> S
            if v < V3_MAX and v < INT32_MAX:
                shift_c = S
                break
        shifts_dw2[c] = shift_c

    V3_max_cal = 0
    for c, s_c in enumerate(S_dw2_arr):
        max_bn_dw2_c = (max_dw2 * int(s_c)) >> 7
        v3_c = max_bn_dw2_c >> int(shifts_dw2[c])
        if v3_c > V3_max_cal:
            V3_max_cal = v3_c

    print(f"  SHIFT_DW2 per-channel: [{shifts_dw2.min()}..{shifts_dw2.max()}]  "
          f"max V3 (calibrated) = {V3_max_cal:,}  (limit {V3_MAX:,})")

    # --- SHIFT_PW2 (global) ---
    max_pw2    = 32 * max_w * V3_max_cal
    max_bn_pw2 = (max_pw2 * S_pw2) >> 7

    shift_pw2 = 0
    for S in range(0, 32):
        v = max_bn_pw2 >> S
        if v < INT32_MAX and (64 * max_w * v) < INT32_MAX:
            shift_pw2 = S
            break
    V4 = max_bn_pw2 >> shift_pw2
    print(f"  SHIFT_PW2={shift_pw2}:  after_BN_PW2={max_bn_pw2} >> {shift_pw2} = {V4}")
    print(f"  FC accumulator max: {64 * max_w * V4:,}  "
          f"({'OK' if 64*max_w*V4 < INT32_MAX else 'OVERFLOW'})")

    return {
        'dw1': shift_dw1, 'pw1': shift_pw1,
        'dw2_per': shifts_dw2, 'pw2': shift_pw2,
    }


def compute_se_shifts(V2_pool1, se_fc1_max_scale, se_fc2_max_scale):
    """
    Compute SE block shifts.

    V2_pool1: max activation value entering the SE block (after BN_PW1+rshift).
    se_fc1_max_scale: max weight magnitude in SE Dense1 (int8, typically <= 127).
    se_fc2_max_scale: max weight magnitude in SE Dense2.

    The SE GlobalAvgPool reduces spatial dim but not value range.
    SE_DENSE1: 32 inputs -> max_acc = 32 * fc1_w * V2
    SE_DENSE2: 8  inputs -> max_acc = 8  * fc2_w * (SE_D1_out >> SE_SHIFT_D1)
    HardSigmoid: y_q7 = clip((x >> SE_SHIFT_HS) + 64, 0, 128)
    """
    max_w = 127

    # SE Dense1 input: V2_pool1 (same as pool1 output)
    max_se_d1 = 32 * max_w * V2_pool1

    # SE_SHIFT_D1: reduce SE Dense1 output to allow Dense2 to fit
    shift_d1 = 0
    for S in range(0, 32):
        v = max_se_d1 >> S
        if (8 * max_w * v) < INT32_MAX:
            shift_d1 = S
            break
    V_d1 = max_se_d1 >> shift_d1
    max_se_d2 = 8 * max_w * V_d1

    # SE_SHIFT_HS: map SE Dense2 output to [0, 128] Q7
    # y = clip((x >> shift_hs) + 64, 0, 128)
    # Want: max_se_d2 >> shift_hs <= 64 (so max y = 128)
    shift_hs = 0
    for S in range(0, 64):
        if (max_se_d2 >> S) <= 64:
            shift_hs = S
            break

    print(f"  SE: pool1_max={V2_pool1}  d1_acc_max={max_se_d1}")
    print(f"  SE_SHIFT_D1={shift_d1}  (d1_out -> {V_d1})")
    print(f"  d2_acc_max={max_se_d2}")
    print(f"  SE_SHIFT_HS={shift_hs}  (maps {max_se_d2} >> {shift_hs} = "
          f"{max_se_d2 >> shift_hs} to y_q7 = {min(128, (max_se_d2 >> shift_hs) + 64)})")

    return {'shift_d1': shift_d1, 'shift_hs': shift_hs}


# ---------------------------------------------------------------------------
# State dict key helpers
# ---------------------------------------------------------------------------

def np_of(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"Missing key '{key}' in state dict.\n"
                       f"Available: {sorted(state_dict.keys())}")
    return state_dict[key].detach().cpu().float().numpy()


def check_keys(state_dict, required):
    missing = [k for k in required if k not in state_dict]
    if missing:
        print(f"ERROR: missing state dict keys: {missing}")
        print(f"Available: {sorted(state_dict.keys())}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Export: MobileCNN-1D
# ---------------------------------------------------------------------------

REQUIRED_MOBILE1D = [
    "dw1.weight", "dw1.bias",
    "bn_dw1.weight", "bn_dw1.bias", "bn_dw1.running_mean", "bn_dw1.running_var",
    "pw1.weight", "pw1.bias",
    "bn_pw1.weight", "bn_pw1.bias", "bn_pw1.running_mean", "bn_pw1.running_var",
    "dw2.weight", "dw2.bias",
    "bn_dw2.weight", "bn_dw2.bias", "bn_dw2.running_mean", "bn_dw2.running_var",
    "pw2.weight", "pw2.bias",
    "bn_pw2.weight", "bn_pw2.bias", "bn_pw2.running_mean", "bn_pw2.running_var",
    "fc.weight", "fc.bias",
]

REQUIRED_MOBILE_SE = REQUIRED_MOBILE1D + [
    "se_fc1.weight", "se_fc1.bias",
    "se_fc2.weight", "se_fc2.bias",
]


def export_mobile1d(state_dict, output_path, pth_source=""):
    check_keys(state_dict, REQUIRED_MOBILE1D)

    # Quantise conv/dense weights
    dw1_w, s = quantise_tensor(np_of(state_dict, "dw1.weight"))   # (57,1,5)
    dw1_b, _ = quantise_tensor(np_of(state_dict, "dw1.bias"))     # (57,)
    pw1_w, s = quantise_tensor(np_of(state_dict, "pw1.weight"))   # (32,57,1)
    pw1_b, _ = quantise_tensor(np_of(state_dict, "pw1.bias"))     # (32,)
    dw2_w, s = quantise_tensor(np_of(state_dict, "dw2.weight"))   # (32,1,5)
    dw2_b, _ = quantise_tensor(np_of(state_dict, "dw2.bias"))     # (32,)
    pw2_w, s = quantise_tensor(np_of(state_dict, "pw2.weight"))   # (64,32,1)
    pw2_b, _ = quantise_tensor(np_of(state_dict, "pw2.bias"))     # (64,)
    fc_w,  _ = quantise_tensor(np_of(state_dict, "fc.weight"))    # (1,64)
    fc_b,  _ = quantise_tensor(np_of(state_dict, "fc.bias"))      # (1,)

    # Pre-fold BatchNorm
    bn_dw1_sc, bn_dw1_off = fold_batchnorm(
        np_of(state_dict, "bn_dw1.weight"), np_of(state_dict, "bn_dw1.bias"),
        np_of(state_dict, "bn_dw1.running_mean"), np_of(state_dict, "bn_dw1.running_var"))
    bn_pw1_sc, bn_pw1_off = fold_batchnorm(
        np_of(state_dict, "bn_pw1.weight"), np_of(state_dict, "bn_pw1.bias"),
        np_of(state_dict, "bn_pw1.running_mean"), np_of(state_dict, "bn_pw1.running_var"))
    bn_dw2_sc, bn_dw2_off = fold_batchnorm(
        np_of(state_dict, "bn_dw2.weight"), np_of(state_dict, "bn_dw2.bias"),
        np_of(state_dict, "bn_dw2.running_mean"), np_of(state_dict, "bn_dw2.running_var"))
    bn_pw2_sc, bn_pw2_off = fold_batchnorm(
        np_of(state_dict, "bn_pw2.weight"), np_of(state_dict, "bn_pw2.bias"),
        np_of(state_dict, "bn_pw2.running_mean"), np_of(state_dict, "bn_pw2.running_var"))

    bn_scales = {
        'bn_dw1': bn_dw1_sc, 'bn_pw1': bn_pw1_sc,
        'bn_dw2': bn_dw2_sc, 'bn_pw2': bn_pw2_sc,
    }

    # --- Data-driven calibration: find actual pool1 max from test samples ---
    print("\n--- Data-driven Block 1 calibration ---")
    actual_pool1_max = None
    test_header = os.path.join(_SCRIPT_DIR, "test_input.h")
    s0, s1 = load_test_inputs_from_header(test_header)
    if s0 is not None:
        # Compute shifts_dw1 / shift_pw1 first (need them for simulation)
        _S_dw1 = int(bn_dw1_sc.max());  _S_pw1 = int(bn_pw1_sc.max())
        _max_bn_dw1 = (5 * 127 * 128 * _S_dw1) >> 7
        _shift_dw1 = next(S for S in range(32)
                          if (_max_bn_dw1 >> S) < INT32_MAX
                          and 57 * 127 * (_max_bn_dw1 >> S) < INT32_MAX)
        _V1 = _max_bn_dw1 >> _shift_dw1
        _max_bn_pw1 = (57 * 127 * _V1 * _S_pw1) >> 7
        _shift_pw1 = next(S for S in range(32)
                          if (_max_bn_pw1 >> S) < INT32_MAX
                          and 5 * 127 * (_max_bn_pw1 >> S) < INT32_MAX)
        pool1_max = 0
        for sample in (s0, s1):
            p = simulate_block1_pool1_max(
                sample,
                dw1_w, dw1_b, bn_dw1_sc, bn_dw1_off, _shift_dw1,
                pw1_w, pw1_b, bn_pw1_sc, bn_pw1_off, _shift_pw1)
            pool1_max = max(pool1_max, int(np.abs(p).max()))
        actual_pool1_max = pool1_max
        print(f"  Simulated pool1 max over {2} test samples: {actual_pool1_max}")
    else:
        print(f"  test_input.h not found at {test_header} -- using theoretical calibration")

    print("\n--- Computing overflow-safe right-shifts for MobileCNN-1D ---")
    shifts = compute_all_shifts(bn_scales, actual_pool1_max=actual_pool1_max)

    # Summaries
    print(f"\nDW1  weights: {dw1_w.shape}  BN_DW1 scale: [{bn_dw1_sc.min()}, {bn_dw1_sc.max()}]")
    print(f"PW1  weights: {pw1_w.shape}  BN_PW1 scale: [{bn_pw1_sc.min()}, {bn_pw1_sc.max()}]")
    print(f"DW2  weights: {dw2_w.shape}  BN_DW2 scale: [{bn_dw2_sc.min()}, {bn_dw2_sc.max()}]")
    print(f"PW2  weights: {pw2_w.shape}  BN_PW2 scale: [{bn_pw2_sc.min()}, {bn_pw2_sc.max()}]")
    print(f"FC   weights: {fc_w.shape}")

    guard = os.path.basename(output_path).upper().replace(".", "_")
    total = (dw1_w.size + dw1_b.size + bn_dw1_sc.size + bn_dw1_off.size +
             pw1_w.size + pw1_b.size + bn_pw1_sc.size + bn_pw1_off.size +
             dw2_w.size + dw2_b.size + bn_dw2_sc.size + bn_dw2_off.size +
             shifts['dw2_per'].size +
             pw2_w.size + pw2_b.size + bn_pw2_sc.size + bn_pw2_off.size +
             fc_w.size  + fc_b.size)
    print(f"\nTotal int32_t elements: {total}  ({total*4//1024} KB)")

    lines = [
        f"/* {os.path.basename(output_path)}",
        f" * Auto-generated by export_mobile_weights.py",
        f" * Source: {pth_source}",
        f" *",
        f" * MobileCNN-1D (Option A) trained weights for X-HEEP int32 inference.",
        f" * Architecture: DWConv+BN+ReLU+PWConv+BN+ReLU+MaxPool x 2, then FC.",
        f" * All conv/fc weight arrays are int8-range int32_t [-127, 127].",
        f" * BatchNorm is pre-folded (Q7: 128 = 1.0).",
        f" *",
        f" * MOBILE_SHIFT_* values are computed from actual BN scale ranges:",
        f" *   tensor_rshift_inplace(x, MOBILE_SHIFT_X) is called after BN_X+ReLU.",
        f" *   This prevents accumulator overflow in the next layer.",
        f" */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
        f"/* Right-shift amounts (calibrated from trained BN scales) */",
        f"#define MOBILE_SHIFT_DW1  {shifts['dw1']}",
        f"#define MOBILE_SHIFT_PW1  {shifts['pw1']}",
        f"/* DW2 uses per-channel shifts -- see mobile_bn_dw2_shift[] below */",
        f"#define MOBILE_SHIFT_PW2  {shifts['pw2']}",
        f"",
        f"/* ---- Block 1: Depthwise (57 -> 57, k=5) ---- */",
        _c_array("mobile_dw1_weight", dw1_w,
                 f"shape={list(dw1_w.shape)}  DWConv k=5, groups=57, layout [57][1][5]"),
        _c_array("mobile_dw1_bias",   dw1_b,   f"shape={list(dw1_b.shape)}"),
        _c_array("mobile_bn_dw1_scale",  bn_dw1_sc,
                 f"shape={list(bn_dw1_sc.shape)}  Q7 BN scale (range [{bn_dw1_sc.min()},{bn_dw1_sc.max()}])"),
        _c_array("mobile_bn_dw1_offset", bn_dw1_off, f"shape={list(bn_dw1_off.shape)}"),
        f"/* ---- Block 1: Pointwise (57 -> 32, k=1) ---- */",
        _c_array("mobile_pw1_weight", pw1_w,
                 f"shape={list(pw1_w.shape)}  PWConv k=1, layout [32][57][1]"),
        _c_array("mobile_pw1_bias",   pw1_b,   f"shape={list(pw1_b.shape)}"),
        _c_array("mobile_bn_pw1_scale",  bn_pw1_sc,
                 f"shape={list(bn_pw1_sc.shape)}  Q7 (range [{bn_pw1_sc.min()},{bn_pw1_sc.max()}])"),
        _c_array("mobile_bn_pw1_offset", bn_pw1_off, f"shape={list(bn_pw1_off.shape)}"),
        f"/* ---- Block 2: Depthwise (32 -> 32, k=5) ---- */",
        _c_array("mobile_dw2_weight", dw2_w,
                 f"shape={list(dw2_w.shape)}  DWConv k=5, groups=32, layout [32][1][5]"),
        _c_array("mobile_dw2_bias",   dw2_b,   f"shape={list(dw2_b.shape)}"),
        _c_array("mobile_bn_dw2_scale",  bn_dw2_sc,
                 f"shape={list(bn_dw2_sc.shape)}  Q7 (range [{bn_dw2_sc.min()},{bn_dw2_sc.max()}])"),
        _c_array("mobile_bn_dw2_offset", bn_dw2_off, f"shape={list(bn_dw2_off.shape)}"),
        _c_array("mobile_bn_dw2_shift",  shifts['dw2_per'],
                 f"Per-channel post-shift after BN_DW2 (range [{int(shifts['dw2_per'].min())},{int(shifts['dw2_per'].max())}])\n"
                 f"Used by batchnorm_rshift_perchannel() -- one entry per DW2 output channel."),
        f"/* ---- Block 2: Pointwise (32 -> 64, k=1) ---- */",
        _c_array("mobile_pw2_weight", pw2_w,
                 f"shape={list(pw2_w.shape)}  PWConv k=1, layout [64][32][1]"),
        _c_array("mobile_pw2_bias",   pw2_b,   f"shape={list(pw2_b.shape)}"),
        _c_array("mobile_bn_pw2_scale",  bn_pw2_sc,
                 f"shape={list(bn_pw2_sc.shape)}  Q7 (range [{bn_pw2_sc.min()},{bn_pw2_sc.max()}])"),
        _c_array("mobile_bn_pw2_offset", bn_pw2_off, f"shape={list(bn_pw2_off.shape)}"),
        f"/* ---- Head: FC (64 -> 1) ---- */",
        _c_array("mobile_fc_weight", fc_w,
                 f"shape={list(fc_w.shape)}  layout [1][64]"),
        _c_array("mobile_fc_bias",   fc_b,   f"shape={list(fc_b.shape)}"),
        f"#endif /* {guard} */",
        "",
    ]
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {output_path}")
    return shifts


def export_mobile_se(state_dict, output_path, pth_source=""):
    check_keys(state_dict, REQUIRED_MOBILE_SE)

    # All shared mobile1d layers (same quantisation)
    dw1_w, _ = quantise_tensor(np_of(state_dict, "dw1.weight"))
    dw1_b, _ = quantise_tensor(np_of(state_dict, "dw1.bias"))
    pw1_w, _ = quantise_tensor(np_of(state_dict, "pw1.weight"))
    pw1_b, _ = quantise_tensor(np_of(state_dict, "pw1.bias"))
    dw2_w, _ = quantise_tensor(np_of(state_dict, "dw2.weight"))
    dw2_b, _ = quantise_tensor(np_of(state_dict, "dw2.bias"))
    pw2_w, _ = quantise_tensor(np_of(state_dict, "pw2.weight"))
    pw2_b, _ = quantise_tensor(np_of(state_dict, "pw2.bias"))
    fc_w,  _ = quantise_tensor(np_of(state_dict, "fc.weight"))
    fc_b,  _ = quantise_tensor(np_of(state_dict, "fc.bias"))

    bn_dw1_sc, bn_dw1_off = fold_batchnorm(
        np_of(state_dict, "bn_dw1.weight"), np_of(state_dict, "bn_dw1.bias"),
        np_of(state_dict, "bn_dw1.running_mean"), np_of(state_dict, "bn_dw1.running_var"))
    bn_pw1_sc, bn_pw1_off = fold_batchnorm(
        np_of(state_dict, "bn_pw1.weight"), np_of(state_dict, "bn_pw1.bias"),
        np_of(state_dict, "bn_pw1.running_mean"), np_of(state_dict, "bn_pw1.running_var"))
    bn_dw2_sc, bn_dw2_off = fold_batchnorm(
        np_of(state_dict, "bn_dw2.weight"), np_of(state_dict, "bn_dw2.bias"),
        np_of(state_dict, "bn_dw2.running_mean"), np_of(state_dict, "bn_dw2.running_var"))
    bn_pw2_sc, bn_pw2_off = fold_batchnorm(
        np_of(state_dict, "bn_pw2.weight"), np_of(state_dict, "bn_pw2.bias"),
        np_of(state_dict, "bn_pw2.running_mean"), np_of(state_dict, "bn_pw2.running_var"))

    # SE Dense layers
    se_fc1_w, _ = quantise_tensor(np_of(state_dict, "se_fc1.weight"))  # (8, 32)
    se_fc1_b, _ = quantise_tensor(np_of(state_dict, "se_fc1.bias"))    # (8,)
    se_fc2_w, _ = quantise_tensor(np_of(state_dict, "se_fc2.weight"))  # (32, 8)
    se_fc2_b, _ = quantise_tensor(np_of(state_dict, "se_fc2.bias"))    # (32,)

    bn_scales = {
        'bn_dw1': bn_dw1_sc, 'bn_pw1': bn_pw1_sc,
        'bn_dw2': bn_dw2_sc, 'bn_pw2': bn_pw2_sc,
    }

    # --- Data-driven calibration: find actual pool1 max from test samples ---
    print("\n--- Data-driven Block 1 calibration ---")
    actual_pool1_max = None
    test_header = os.path.join(_SCRIPT_DIR, "test_input.h")
    s0, s1 = load_test_inputs_from_header(test_header)
    if s0 is not None:
        _S_dw1 = int(bn_dw1_sc.max());  _S_pw1 = int(bn_pw1_sc.max())
        _max_bn_dw1 = (5 * 127 * 128 * _S_dw1) >> 7
        _shift_dw1 = next(S for S in range(32)
                          if (_max_bn_dw1 >> S) < INT32_MAX
                          and 57 * 127 * (_max_bn_dw1 >> S) < INT32_MAX)
        _V1 = _max_bn_dw1 >> _shift_dw1
        _max_bn_pw1 = (57 * 127 * _V1 * _S_pw1) >> 7
        _shift_pw1 = next(S for S in range(32)
                          if (_max_bn_pw1 >> S) < INT32_MAX
                          and 5 * 127 * (_max_bn_pw1 >> S) < INT32_MAX)
        pool1_max = 0
        for sample in (s0, s1):
            p = simulate_block1_pool1_max(
                sample,
                dw1_w, dw1_b, bn_dw1_sc, bn_dw1_off, _shift_dw1,
                pw1_w, pw1_b, bn_pw1_sc, bn_pw1_off, _shift_pw1)
            pool1_max = max(pool1_max, int(np.abs(p).max()))
        actual_pool1_max = pool1_max
        print(f"  Simulated pool1 max over {2} test samples: {actual_pool1_max}")
    else:
        print(f"  test_input.h not found at {test_header} -- using theoretical calibration")

    print("\n--- Computing overflow-safe right-shifts for MobileCNN-SE-1D ---")
    shifts = compute_all_shifts(bn_scales, actual_pool1_max=actual_pool1_max)

    # Compute V2_pool1 to feed into SE shift analysis
    max_input = 128
    max_w     = 127
    S_dw1 = int(bn_dw1_sc.max())
    S_pw1 = int(bn_pw1_sc.max())
    max_dw1    = 5 * max_w * max_input
    max_bn_dw1 = (max_dw1 * S_dw1) >> 7
    V1 = max_bn_dw1 >> shifts['dw1']
    max_pw1    = 57 * max_w * V1
    max_bn_pw1 = (max_pw1 * S_pw1) >> 7
    V2 = max_bn_pw1 >> shifts['pw1']   # pool1 output range (same after MaxPool)

    print("\n--- Computing SE block shifts ---")
    se_shifts = compute_se_shifts(V2, 127, 127)

    guard = os.path.basename(output_path).upper().replace(".", "_")
    total = (dw1_w.size + dw1_b.size + bn_dw1_sc.size + bn_dw1_off.size +
             pw1_w.size + pw1_b.size + bn_pw1_sc.size + bn_pw1_off.size +
             se_fc1_w.size + se_fc1_b.size + se_fc2_w.size + se_fc2_b.size +
             dw2_w.size + dw2_b.size + bn_dw2_sc.size + bn_dw2_off.size +
             shifts['dw2_per'].size +
             pw2_w.size + pw2_b.size + bn_pw2_sc.size + bn_pw2_off.size +
             fc_w.size  + fc_b.size)
    print(f"\nTotal int32_t elements: {total}  ({total*4//1024} KB)")

    lines = [
        f"/* {os.path.basename(output_path)}",
        f" * Auto-generated by export_mobile_weights.py",
        f" * Source: {pth_source}",
        f" *",
        f" * MobileCNN-SE-1D (Option B) trained weights for X-HEEP int32 inference.",
        f" * Architecture: DWConv+BN+ReLU+PWConv+BN+ReLU+MaxPool x 2 + SE after Block 1.",
        f" */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
        f"/* Right-shift amounts (calibrated from trained BN scales) */",
        f"#define MOBILE_SHIFT_DW1    {shifts['dw1']}",
        f"#define MOBILE_SHIFT_PW1    {shifts['pw1']}",
        f"/* DW2 uses per-channel shifts -- see mobile_bn_dw2_shift[] below */",
        f"#define MOBILE_SHIFT_PW2    {shifts['pw2']}",
        f"",
        f"/* SE block: hardsigmoid maps x -> clip((x >> SE_SHIFT_HS) + 64, 0, 128) */",
        f"#define MOBILE_SE_SHIFT_D1  {se_shifts['shift_d1']}  /* rshift after SE Dense1+ReLU */",
        f"#define MOBILE_SE_SHIFT_HS  {se_shifts['shift_hs']}  /* hard-sigmoid shift for SE Dense2 output */",
        f"",
        f"/* ---- Block 1: Depthwise (57->57, k=5) ---- */",
        _c_array("mobile_dw1_weight", dw1_w, f"shape={list(dw1_w.shape)}"),
        _c_array("mobile_dw1_bias",   dw1_b, f"shape={list(dw1_b.shape)}"),
        _c_array("mobile_bn_dw1_scale",  bn_dw1_sc, f"Q7 [{bn_dw1_sc.min()},{bn_dw1_sc.max()}]"),
        _c_array("mobile_bn_dw1_offset", bn_dw1_off, f"shape={list(bn_dw1_off.shape)}"),
        f"/* ---- Block 1: Pointwise (57->32, k=1) ---- */",
        _c_array("mobile_pw1_weight", pw1_w, f"shape={list(pw1_w.shape)}"),
        _c_array("mobile_pw1_bias",   pw1_b, f"shape={list(pw1_b.shape)}"),
        _c_array("mobile_bn_pw1_scale",  bn_pw1_sc, f"Q7 [{bn_pw1_sc.min()},{bn_pw1_sc.max()}]"),
        _c_array("mobile_bn_pw1_offset", bn_pw1_off, f"shape={list(bn_pw1_off.shape)}"),
        f"/* ---- SE: Dense1 (32->8) and Dense2 (8->32) ---- */",
        _c_array("mobile_se_fc1_weight", se_fc1_w,
                 f"shape={list(se_fc1_w.shape)}  SE squeeze layer"),
        _c_array("mobile_se_fc1_bias",   se_fc1_b, f"shape={list(se_fc1_b.shape)}"),
        _c_array("mobile_se_fc2_weight", se_fc2_w,
                 f"shape={list(se_fc2_w.shape)}  SE excitation layer"),
        _c_array("mobile_se_fc2_bias",   se_fc2_b, f"shape={list(se_fc2_b.shape)}"),
        f"/* ---- Block 2: Depthwise (32->32, k=5) ---- */",
        _c_array("mobile_dw2_weight", dw2_w, f"shape={list(dw2_w.shape)}"),
        _c_array("mobile_dw2_bias",   dw2_b, f"shape={list(dw2_b.shape)}"),
        _c_array("mobile_bn_dw2_scale",  bn_dw2_sc, f"Q7 [{bn_dw2_sc.min()},{bn_dw2_sc.max()}]"),
        _c_array("mobile_bn_dw2_offset", bn_dw2_off, f"shape={list(bn_dw2_off.shape)}"),
        _c_array("mobile_bn_dw2_shift",  shifts['dw2_per'],
                 f"Per-channel post-shift after BN_DW2 (range [{int(shifts['dw2_per'].min())},{int(shifts['dw2_per'].max())}])\n"
                 f"Used by batchnorm_rshift_perchannel() -- one entry per DW2 output channel."),
        f"/* ---- Block 2: Pointwise (32->64, k=1) ---- */",
        _c_array("mobile_pw2_weight", pw2_w, f"shape={list(pw2_w.shape)}"),
        _c_array("mobile_pw2_bias",   pw2_b, f"shape={list(pw2_b.shape)}"),
        _c_array("mobile_bn_pw2_scale",  bn_pw2_sc, f"Q7 [{bn_pw2_sc.min()},{bn_pw2_sc.max()}]"),
        _c_array("mobile_bn_pw2_offset", bn_pw2_off, f"shape={list(bn_pw2_off.shape)}"),
        f"/* ---- Head: FC (64->1) ---- */",
        _c_array("mobile_fc_weight", fc_w, f"shape={list(fc_w.shape)}"),
        _c_array("mobile_fc_bias",   fc_b, f"shape={list(fc_b.shape)}"),
        f"#endif /* {guard} */",
        "",
    ]
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_pth(model_name, pth_arg):
    if pth_arg:
        return pth_arg
    out_dir = os.path.normpath(os.path.join(
        _PY_SRC, "results", "deepBindi", model_name
    ))
    for fname in [f"type4_fold0.pth", "best_model.pth"]:
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p):
            print(f"Auto-detected checkpoint: {p}")
            return p
    sys.exit(
        f"Cannot auto-detect checkpoint for '{model_name}'.\n"
        f"Run:  python train_mobilecnn_1d.py --model {model_name}\n"
        f"Then: python export_mobile_weights.py --model {model_name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export MobileCNN-1D / SE weights to C int32_t header.",
    )
    parser.add_argument("--model", required=True,
        choices=["mobile1d", "mobile_se"],
        help="Which model to export")
    parser.add_argument("--pth", default=None,
        help="Path to .pth checkpoint (auto-detected if omitted)")
    parser.add_argument("--output", "-o", default=None,
        help="Output .h file (default: weights_<model>.h in this directory)")
    args = parser.parse_args()

    pth_path = resolve_pth(args.model, args.pth)
    if not os.path.isfile(pth_path):
        sys.exit(f"File not found: {pth_path}")

    out_name = args.output or os.path.join(_SCRIPT_DIR, f"weights_{args.model}.h")

    print(f"Loading {pth_path}  ({os.path.getsize(pth_path)//1024} KB)")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    if args.model == "mobile1d":
        export_mobile1d(state_dict, out_name, pth_source=pth_path)
    else:
        export_mobile_se(state_dict, out_name, pth_source=pth_path)

    print(f"\nNext: rebuild with -DDEEPBINDI_MODEL=1 (mobile1d) or -DDEEPBINDI_MODEL=2 (mobile_se)")
    print(f"      and -DDEEPBINDI_REAL_WEIGHTS to use {os.path.basename(out_name)}")


if __name__ == "__main__":
    main()
