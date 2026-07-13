#!/usr/bin/env python3
"""
extract_weights.py  --  Inspect and extract weights from a DeepBindi TFLite model.

IMPORTANT - ARCHITECTURE NOTE
------------------------------
The TFLite models in saved_model/tflite/ are based on CNN_2d_tensorflow_softmax
(Keras TF3 in cnn_models.py), NOT the PyTorch CNN_1D_v2 (PYE) that the C port
implements.  Key differences:

  TFLite micro (CNN_2d_tensorflow_softmax / model_quant_1FC.tflite):
    Input  : (1, 57, 10, 1)  NHWC (channel-last, TensorFlow convention)
    Conv   : Conv2D(1->64, kernel=(1,5), padding=valid)
    Pool   : MaxPool2D((1,2))  -> output (1, 57, 3, 64)
    Flatten: 57 * 3 * 64 = 10 944 features
    Dense  : 10944 -> 2  (softmax, 2-class output)
    Predict: argmax(output)  =>  0=NO_FEAR, 1=FEAR

  C port (CNN_1D_v2 PyTorch / cnn_models_c.c):
    Input  : (1, 57, 1, 10)  NCHW (channel-first, PyTorch convention)
    Conv1  : Conv1d(57->32, k=5, pad=1) -> BN -> ReLU -> MaxPool(2)
    Conv2  : Conv1d(32->64, k=5, pad=1) -> BN -> ReLU -> MaxPool(2)
    Flatten: 64 features
    Dense  : 64 -> 1  (sigmoid / threshold at 0)

Consequently, the weights extracted from the TFLite models CANNOT be loaded
directly into the C port.  Options:
  A) Export the PyTorch CNN_1D_v2 trained model to ONNX and extract weights.
  B) Retrain CNN_2d_tensorflow_softmax and port that architecture to C instead.
  C) Use the dummy weights currently in the C port for cycle-count benchmarking
     and replace with real weights once a compatible checkpoint is available.

This script is still useful for:
  1. Inspecting the TFLite model structure (operators, tensor shapes, dtypes).
  2. Verifying the input/output quantization parameters (scale, zero_point).
  3. Extracting all constant tensors as int32_t C arrays for reference.

Usage:
    # Inspect model structure only (no output file):
    python extract_weights.py --inspect path/to/model_quant_1FC.tflite

    # Full extraction to C header:
    python extract_weights.py path/to/model_quant_1FC.tflite [output.h]

Default model path (relative to this script):
    ../../../EPFL_STAY_LAURA/python_code/src/CH07_TFLite/saved_model/tflite/model_quant_1FC.tflite

Quantisation convention (from utils.py tflite_inference):
    x_quantised = x_float / scale + zero_point   (float -> int8)
    x_float     = (x_quantised - zero_point) * scale  (int8 -> float)

Requirements:
    pip install tensorflow numpy
    (or: pip install tflite-runtime numpy  for a lighter install)
"""

import sys
import os
import argparse

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

# ---- Default paths -------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.normpath(os.path.join(
    _SCRIPT_DIR,
    "..", "..", "..",          # DeepBindi_PortC -> DeepBindi
    "EPFL_STAY_LAURA", "python_code", "src",
    "CH07_TFLite", "saved_model", "tflite",
    "model_quant_1FC.tflite"
))
_DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "weights_tflite_1FC.h")


# ---- TFLite loader -------------------------------------------------------

def _load_interpreter(model_path):
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=model_path)
        except ImportError:
            sys.exit(
                "TFLite interpreter not found.\n"
                "Install with:  pip install tensorflow\n"
                "or (lighter):  pip install tflite-runtime"
            )
    interp.allocate_tensors()
    return interp


# ---- Model inspection -------------------------------------------------------

def inspect_model(interp):
    """Print a human-readable summary: I/O, operators, all tensors."""
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    print("=" * 60)
    print("MODEL INPUTS")
    print("=" * 60)
    for d in input_details:
        q = d.get("quantization", (0.0, 0))
        print(f"  [{d['index']:3d}] {d['name']}")
        print(f"        shape={d['shape']}  dtype={d['dtype'].__name__}")
        print(f"        quantisation: scale={q[0]:.6f}  zero_point={q[1]}")
        print(f"        dequant formula: x_float = (x_int8 - {q[1]}) * {q[0]:.6f}")

    print()
    print("=" * 60)
    print("MODEL OUTPUTS")
    print("=" * 60)
    for d in output_details:
        q = d.get("quantization", (0.0, 0))
        print(f"  [{d['index']:3d}] {d['name']}")
        print(f"        shape={d['shape']}  dtype={d['dtype'].__name__}")
        print(f"        quantisation: scale={q[0]:.6f}  zero_point={q[1]}")
        oc = interp.get_output_details()[0]["shape"][-1]
        if oc == 2:
            print(f"        2-class softmax -> use argmax() for prediction")
        elif oc == 1:
            print(f"        1-output sigmoid -> threshold at 0.5 (or 0 in logit space)")

    # Operator list
    try:
        print()
        print("=" * 60)
        print("OPERATORS  (execution order)")
        print("=" * 60)
        for i, op in enumerate(interp._get_ops_details()):
            inames = [interp.get_tensor_details()[t]["name"] for t in op.get("inputs", [])]
            onames = [interp.get_tensor_details()[t]["name"] for t in op.get("outputs", [])]
            print(f"  [{i:2d}] op_type={op.get('op_name','?')}")
            print(f"       inputs : {inames}")
            print(f"       outputs: {onames}")
    except AttributeError:
        print("  (operator list not available in this TFLite version)")

    # All tensors
    print()
    print("=" * 60)
    print("ALL TENSORS")
    print("=" * 60)
    print(f"  {'idx':>4}  {'dtype':>6}  {'shape':>25}  {'quant(scale,zp)':>22}  name")
    for td in interp.get_tensor_details():
        q = td.get("quantization", (0.0, 0))
        shape_str = str(list(td["shape"]))
        q_str = f"({q[0]:.4f}, {q[1]})"
        print(f"  {td['index']:4d}  {td['dtype'].__name__:>6}  {shape_str:>25}  {q_str:>22}  {td['name']}")


# ---- Weight extraction ---------------------------------------------------

def _sanitise(name):
    name = name.replace("/", "_").replace(";", "_").replace(":", "_")
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if name and name[0].isdigit():
        name = "t_" + name
    return name.lower()


def _array_to_c(c_name, arr, quant, values_per_line=12):
    flat  = arr.flatten().astype("int32")
    count = flat.size
    scale, zp = quant
    lines = [
        f"/* shape={list(arr.shape)}  dtype={arr.dtype}  scale={scale:.6f}  zero_point={zp} */",
        f"/* dequant: x_float = (x_int - {zp}) * {scale:.6f} */",
        f"const int32_t {c_name}[{count}] = {{",
    ]
    for i in range(0, count, values_per_line):
        chunk = flat[i:i + values_per_line]
        line  = ", ".join(str(int(v)) for v in chunk)
        if i + values_per_line < count:
            line += ","
        lines.append("    " + line)
    lines.append("};\n")
    return "\n".join(lines)


def extract_weights(interp, output_path):
    """Write a C header with all constant (weight) tensors."""
    input_indices  = {d["index"] for d in interp.get_input_details()}
    output_indices = {d["index"] for d in interp.get_output_details()}
    io_indices     = input_indices | output_indices

    guard = os.path.basename(output_path).upper().replace(".", "_")
    header_lines = [
        f"/* {os.path.basename(output_path)}",
        f" * Auto-generated by extract_weights.py",
        f" *",
        f" * ARCHITECTURE NOTE: these weights are from CNN_2d_tensorflow_softmax",
        f" * (TFLite micro / model_quant_1FC.tflite), NOT from CNN_1D_v2 (PyTorch).",
        f" * They CANNOT be loaded directly into cnn_models_c.c as-is.",
        f" * See extract_weights.py header comment for options.",
        f" *",
        f" * Dequant formula: x_float = (x_int8 - zero_point) * scale",
        f" * (scale and zero_point are in the comment above each array)",
        f" */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
    ]

    total = 0
    for td in interp.get_tensor_details():
        if td["index"] in io_indices:
            continue
        try:
            data = interp.get_tensor(td["index"])
        except Exception:
            continue
        if data is None or data.size <= 1:
            continue
        # Only keep weight/bias tensors (constant, not activations)
        if data.dtype not in (np.int8, np.int32, np.float32):
            continue

        c_name = _sanitise(td["name"]) or f"tensor_{td['index']}"
        quant  = td.get("quantization", (0.0, 0))
        print(f"  [{td['index']:3d}] {td['name']:50s}  shape={list(data.shape)}  {data.dtype}  "
              f"scale={quant[0]:.4f}  zp={quant[1]}")
        header_lines.append(_array_to_c(c_name, data, quant))
        total += data.size

    header_lines += [f"#endif /* {guard} */", ""]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines))

    print(f"\nWrote {output_path}")
    print(f"Total extracted elements: {total}  ({total * 4 / 1024:.1f} KB as int32_t)")


# ---- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect / extract DeepBindi TFLite weights to C int32_t arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=_DEFAULT_MODEL,
        help=f"Path to .tflite file (default: model_quant_1FC.tflite)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=_DEFAULT_OUTPUT,
        help=f"Output .h file (default: {os.path.basename(_DEFAULT_OUTPUT)})",
    )
    parser.add_argument(
        "--inspect", "-i",
        action="store_true",
        help="Print model structure only; do not write output file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model file not found: {args.model}")
        print(f"\nAvailable .tflite files:")
        tflite_dir = os.path.dirname(_DEFAULT_MODEL)
        if os.path.isdir(tflite_dir):
            for f in sorted(os.listdir(tflite_dir)):
                if f.endswith(".tflite"):
                    fpath = os.path.join(tflite_dir, f)
                    print(f"  {fpath}  ({os.path.getsize(fpath)//1024} KB)")
        sys.exit(1)

    print(f"Loading {args.model}  ({os.path.getsize(args.model)//1024} KB)\n")
    interp = _load_interpreter(args.model)

    inspect_model(interp)

    if not args.inspect:
        print()
        print("=" * 60)
        print("EXTRACTING CONSTANT TENSORS")
        print("=" * 60)
        extract_weights(interp, args.output)
        print()
        print("Next steps:")
        print("  1. Review the architecture note at the top of this script.")
        print("  2. If porting CNN_2d_tensorflow_softmax (1FC) to C, the conv weights")
        print("     shape is (1, 5, 1, 64) NHWC -> map to (64, 1, 1, 5) NCHW for conv2d_forward.")
        print("  3. The dense layer has shape (10944, 2); for argmax classification")
        print("     run two Dense(10944->1) passes and pick the larger output.")
        print("  4. For CNN_1D_v2 real weights: export the PyTorch checkpoint instead.")


if __name__ == "__main__":
    main()
