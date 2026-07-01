#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def gemver_p2_reference(A, x, y, z, beta, N):
    A_mat = A.reshape((N, N))
    x_out = x.copy()
    
    for i in range(N):
        aux = 0
        for j in range(N):
            aux += beta * A_mat[j, i] * y[j]
        x_out[i] += aux + z[i]
        
    return x_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_{N}.h"
    npz_filename = data_dir / f"data_{N}.npz"

    # A representa aquí la matriz ya modificada por la parte 1
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    y = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    z = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    beta = np.random.randint(1, 5)

    x_expected = gemver_p2_reference(A, x, y, z, beta, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_GEMVER_P2_H\n")
        f.write("#define DATA_GEMVER_P2_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: {N} */\n")
        f.write(f"#define N {N}\n")
        f.write(f"#define BETA {beta}\n\n")

        write_array(f, "A", A, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        write_array(f, "y", y, interleaved=False)
        write_array(f, "z", z, interleaved=False)
        write_array(f, "x_expected", x_expected, interleaved=False)
        f.write("#endif\n")

    npz_payload = {
        "A": A, "x": x, "y": y, "z": z, "beta": beta,
        "NI": N, "NJ": N, "x_expected": x_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"Part 2 Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate GEMVER Part 2 Data")
    parser.add_argument("--N", type=int, required=True, help="Matrix/vector size N")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()