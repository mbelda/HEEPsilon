#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def mvt_p2_reference(A, y2, x2_init, N):
    """
    Calcula la Parte 2 de MVT: x2[i] = x2[i] + Aᵀ * y2
    En formato 1D (Row-Major), A[j][i] se traduce como A[j * N + i].
    """
    x2_out = np.copy(x2_init)
    
    for i in range(N):
        acc = x2_out[i]
        for j in range(N):
            # A[j][i] en un array unidimensional es j * N + i
            acc += A[j * N + i] * y2[j]
        x2_out[i] = acc
            
    return x2_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_mvt_p2_{N}.h"
    npz_filename = data_dir / f"data_mvt_p2_{N}.npz"

    # 1. Generación de datos iniciales (Inputs)
    # Valores pequeños para evitar desbordamientos en entornos embebidos/CGRA
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    y2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    x2_init = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 2. Ejecución de la referencia para obtener el Golden Output (x2_expected)
    x2_expected = mvt_p2_reference(A, y2, x2_init, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 3. Escritura del C Header (.h)
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_MVT_P2_H\n")
        f.write("#define DATA_MVT_P2_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: N={N} (Matrix N x N) */\n")
        f.write(f"#define N {N}\n\n")

        # Inputs del sistema
        write_array(f, "A", A, interleaved=False)
        write_array(f, "y2", y2, interleaved=False)
        write_array(f, "x2", x2_init, interleaved=False) # x2 inicial en la memoria
        
        # Golden output esperado
        write_array(f, "x2_expected", x2_expected, interleaved=False)
        
        f.write("#endif\n")

    # 4. Volcado comprimido (.npz) para debug o tests en Python
    npz_payload = {
        "A": A, 
        "y2": y2, 
        "x2_init": x2_init,
        "N": N, 
        "x2_expected": x2_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"MVT P2 Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate MVT Part 2 Data (Transposed Matrix-Vector Multiplication)")
    parser.add_argument("--N", type=int, required=True, help="Dimension N (Matriz NxN, Vectores tamaño N)")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()