#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def gemver_p3_reference(A, x, w, alpha, N):
    A_mat = A.reshape((N, N))
    w_out = w.copy()
    
    for i in range(N):
        aux = 0
        for j in range(N):
            aux += alpha * A_mat[i, j] * x[j]
        w_out[i] += aux
        
    return w_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_{N}.h"
    npz_filename = data_dir / f"data_{N}.npz"

    # A y x representan aquí los datos resultantes de las partes 1 y 2
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    w = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    alpha = np.random.randint(1, 5)

    w_expected = gemver_p3_reference(A, x, w, alpha, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_GEMVER_P3_H\n")
        f.write("#define DATA_GEMVER_P3_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: {N} */\n")
        f.write(f"#define N {N}\n")
        f.write(f"#define ALPHA {alpha}\n\n")

        write_array(f, "A", A, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        write_array(f, "w", w, interleaved=False)
        write_array(f, "w_expected", w_expected, interleaved=False)
        f.write("#endif\n")

    npz_payload = {
        "A": A, "x": x, "w": w, "alpha": alpha,
        "NI": N, "NJ": N, "w_expected": w_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"Part 3 Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate GEMVER Part 3 Data")
    parser.add_argument("--N", type=int, required=True, help="Matrix/vector size N")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()