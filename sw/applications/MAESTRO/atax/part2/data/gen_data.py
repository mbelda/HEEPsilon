#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def atax_p2_reference(A, tmp, M, N):
    """Calcula la Parte 2 de ATAX: y = A^T * tmp"""
    y_out = np.zeros(N, dtype=np.int32)
    for j in range(N):
        acc = 0
        for i in range(M):
            acc += A[i * N + j] * tmp[i]
        y_out[j] = acc
    return y_out

def generate_and_write(M, N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    M, N = int(M), int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_atax_p2_{M}_{N}.h"
    npz_filename = data_dir / f"data_atax_p2_{M}_{N}.npz"

    A = np.random.randint(-5, 6, size=(M * N), dtype=np.int32)
    tmp = np.random.randint(-5, 6, size=(M,), dtype=np.int32)

    y_expected = atax_p2_reference(A, tmp, M, N)

    def write_array(f, name, arr):
        f.write(f"int {name}[{len(arr)}] = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_ATAX_P2_H\n#define DATA_ATAX_P2_H\n\n")
        f.write(f"#define M {M}\n#define N {N}\n\n")
        write_array(f, "A", A)
        write_array(f, "tmp", tmp)
        write_array(f, "y_expected", y_expected)
        f.write("#endif\n")

    npz_payload = {"A": A, "tmp": tmp, "M": M, "N": N, "y_expected": y_expected}
    np.savez(npz_filename, **npz_payload)
    print(f"ATAX P2 Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate ATAX Part 2 Data")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    generate_and_write(M=args.M, N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()