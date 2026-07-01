#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def gemver_p1_reference(A, u1, v1, u2, v2, N):
    # Copia para no modificar el original en memoria
    A_out = A.copy().reshape((N, N))
    
    for i in range(N):
        for j in range(N):
            A_out[i, j] += u1[i] * v1[j] + u2[i] * v2[j]
            
    return A_out.flatten()

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_{N}.h"
    npz_filename = data_dir / f"data_{N}.npz"

    # Generación de datos iniciales
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    u1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    v1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    u2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    v2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    A_expected = gemver_p1_reference(A, u1, v1, u2, v2, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_GEMVER_P1_H\n")
        f.write("#define DATA_GEMVER_P1_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: {N} */\n")
        f.write(f"#define N {N}\n\n")

        write_array(f, "A", A, interleaved=False)
        write_array(f, "u1", u1, interleaved=False)
        write_array(f, "v1", v1, interleaved=False)
        write_array(f, "u2", u2, interleaved=False)
        write_array(f, "v2", v2, interleaved=False)
        write_array(f, "A_expected", A_expected, interleaved=False)
        f.write("#endif\n")

    npz_payload = {
        "A": A, "u1": u1, "v1": v1, "u2": u2, "v2": v2,
        "NI": N, "NJ": N, "A_expected": A_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"Part 1 Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate GEMVER Part 1 Data")
    parser.add_argument("--N", type=int, required=True, help="Matrix/vector size N")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()