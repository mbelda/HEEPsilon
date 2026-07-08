#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def mvt_full_reference(A, x1_init, x2_init, y1, y2, N):
    """
    Calcula el benchmark MVT completo (Parte 1 y Parte 2).
    A es un array 1D que representa una matriz NxN en formato Row-Major.
    """
    x1_out = np.copy(x1_init)
    x2_out = np.copy(x2_init)
    
    # Parte 1: x1[i] = x1[i] + A[i][j] * y1[j]
    for i in range(N):
        for j in range(N):
            x1_out[i] += A[i * N + j] * y1[j]
            
    # Parte 2: x2[i] = x2[i] + A[j][i] * y2[j]
    for i in range(N):
        for j in range(N):
            x2_out[i] += A[j * N + i] * y2[j]
            
    return x1_out, x2_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_mvt_full_{N}.h"
    npz_filename = data_dir / f"data_mvt_full_{N}.npz"

    # 1. Generación de estructuras de datos iniciales (Inputs)
    # Comparten la matriz A (N x N)
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    
    # Vectores para la Parte 1
    x1_init = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    y1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    
    # Vectores para la Parte 2
    x2_init = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    y2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 2. Ejecución de la referencia para obtener los Golden Outputs
    x1_expected, x2_expected = mvt_full_reference(A, x1_init, x2_init, y1, y2, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 3. Escritura del C Header (.h) unificado
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_MVT_FULL_H\n")
        f.write("#define DATA_MVT_FULL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: N={N} (Matrix N x N) */\n")
        f.write(f"#define N {N}\n\n")

        # Matriz compartida
        write_array(f, "A", A, interleaved=False)
        
        # Estructuras de la Parte 1
        f.write("/* --- Parte 1 Data --- */\n")
        write_array(f, "x1", x1_init, interleaved=False)
        write_array(f, "y1", y1, interleaved=False)
        write_array(f, "x1_expected", x1_expected, interleaved=False)
        
        # Estructuras de la Parte 2
        f.write("/* --- Parte 2 Data --- */\n")
        write_array(f, "x2", x2_init, interleaved=False)
        write_array(f, "y2", y2, interleaved=False)
        write_array(f, "x2_expected", x2_expected, interleaved=False)
        
        f.write("#endif\n")

    # 4. Volcado comprimido (.npz) para verificación en Python
    npz_payload = {
        "A": A, 
        "N": N,
        "x1_init": x1_init, "y1": y1, "x1_expected": x1_expected,
        "x2_init": x2_init, "y2": y2, "x2_expected": x2_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"MVT Full Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate Complete MVT (Matrix Vector Product) Data")
    parser.add_argument("--N", type=int, required=True, help="Dimension N (Matriz NxN, Vectores tamaño N)")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()