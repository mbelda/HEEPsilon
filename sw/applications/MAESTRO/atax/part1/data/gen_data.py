#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def atax_p1_reference(A, x, M, N):
    """
    Calcula la Parte 1 de ATAX: tmp = A * x con arrays 1D.
    Cada elemento tmp[i] acumula el producto escalar de la fila i por x.
    """
    tmp_out = np.zeros(M, dtype=np.int32)
    
    # Procesamiento por filas
    for i in range(M):
        acc = 0
        for j in range(N):
            acc += A[i * N + j] * x[j]
        tmp_out[i] = acc
            
    return tmp_out

def generate_and_write(M, N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    M = int(M)
    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_atax_p1_{M}_{N}.h"
    npz_filename = data_dir / f"data_atax_p1_{M}_{N}.npz"

    # 1. Generación de estructuras de datos iniciales (Inputs)
    # Usamos enteros pequeños para evitar desbordamientos en la simulación
    A = np.random.randint(-5, 6, size=(M * N), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 2. Ejecución de la referencia de la Parte 1 (Golden output es tmp)
    tmp_expected = atax_p1_reference(A, x, M, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 3. Escritura del C Header (.h) unificado para la Parte 1
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_ATAX_P1_H\n")
        f.write("#define DATA_ATAX_P1_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: M={M}, N={N} */\n")
        f.write(f"#define M {M}\n")
        f.write(f"#define N {N}\n\n")

        # Inputs del sistema
        write_array(f, "A", A, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        
        # Golden output esperado en la memoria del CGRA
        write_array(f, "tmp_expected", tmp_expected, interleaved=False)
        
        f.write("#endif\n")

    # 4. Volcado comprimido (.npz) para el script de testeo/debug de la P1
    npz_payload = {
        "A": A, 
        "x": x, 
        "M": M, 
        "N": N, 
        "tmp_expected": tmp_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"ATAX P1 Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate ATAX Part 1 Data (Matrix-Vector Multiplication)")
    parser.add_argument("--M", type=int, required=True, help="Dimension M (filas de A / tamaño de tmp)")
    parser.add_argument("--N", type=int, required=True, help="Dimension N (columnas de A / tamaño de x)")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(M=args.M, N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()