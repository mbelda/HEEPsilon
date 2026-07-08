#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

# -------------------------------------------------
# Reference computation (Polybench gesummv)
# -------------------------------------------------
def gesummv_reference(A, B, x, y, alpha, beta, N):
    """
    Computes Polybench gesummv entirely with native integers.
    """
    tmp = np.zeros(N, dtype=np.int32)
    y_out = np.zeros(N, dtype=np.int32)

    for i in range(N):
        s_tmp = 0
        s_y = 0
        for j in range(N):
            s_tmp += A[i * N + j] * x[j]
            s_y += B[i * N + j] * x[j]
        
        tmp[i] = s_tmp
        y_out[i] = alpha * tmp[i] + beta * s_y

    return y_out


# -------------------------------------------------
# Generate + write
# -------------------------------------------------
def generate_and_write(N, seed=None):

    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")

    h_filename = data_dir / f"data_{N}.h"
    npz_filename = data_dir / f"data_{N}.npz"

    # Generate int32 arrays directly
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    B = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(N), dtype=np.int32)
    y = np.random.randint(-5, 6, size=(N), dtype=np.int32)
    
    alpha = np.random.randint(1, 5)
    beta = np.random.randint(1, 5)

    # Reference computation without explicit casts
    y_expected = gesummv_reference(A, B, x, y, alpha, beta, N)

    # -------------------------------------------------
    # Write C header
    # -------------------------------------------------
    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_H\n")
        f.write("#define DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"/* Dataset Size: {N} */\n")
        f.write(f"#define N {N}\n\n")
        f.write(f"#define ALPHA {alpha}\n")
        f.write(f"#define BETA {beta}\n\n")

        # In/Out variables marked as interleaved for X-HEEP memory banking
        write_array(f, "A", A, interleaved=False)
        write_array(f, "B", B, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        write_array(f, "y", y, interleaved=False)
        write_array(f, "y_expected", y_expected, interleaved=False)

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ for CGRA simulator
    # -------------------------------------------------
    npz_payload = {
        "A": A,
        "B": B,
        "x": x,
        "y": y,
        "alpha": alpha,
        "beta": beta,
        "NI": N,    
        "NJ": N,    
        "y_expected": y_expected
    }

    np.savez(npz_filename, **npz_payload)

    print(f"Generated size {N}:")
    print(f"  -> {h_filename}")
    print(f"  -> {npz_filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate Gesummv Polybench Data variants by exact size N")
    
    parser.add_argument("--N", type=int, required=True, help="Exact matrix/vector size dimension N")
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()

    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()