#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def atax_reference(A, x, M, N):
    """
    Calcula el kernel ATAX fundido con vectores y matrices 1D.
    Elimina el almacenamiento global del vector intermedio 'tmp'.
    """
    y_out = np.zeros(N, dtype=np.int32)
    
    # Procesamiento por filas
    for i in range(M):
        tmp_acum = 0
        # Primer bucle: acumula el producto de la fila i por el vector x
        for j in range(N):
            tmp_acum += A[i * N + j] * x[j]
            
        # Segundo bucle: distribuye el escalar acumulado sobre y usando la fila i (traspuesta implícita)
        for j in range(N):
            y_out[j] += A[i * N + j] * tmp_acum
            
    return y_out

def generate_and_write(M, N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    M = int(M)
    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_atax_{M}_{N}.h"
    npz_filename = data_dir / f"data_atax_{M}_{N}.npz"

    # 1. Generación de estructuras de datos iniciales (Inputs)
    A = np.random.randint(-5, 6, size=(M * N), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 2. Pipeline de ejecución de referencia sin vector tmp
    y_expected = atax_reference(A, x, M, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 3. Escritura del C Header (.h) unificado sin tmp_expected
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_ATAX_H\n")
        f.write("#define DATA_ATAX_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: M={M}, N={N} */\n")
        f.write(f"#define M {M}\n")
        f.write(f"#define N {N}\n\n")

        # Inputs del sistema
        write_array(f, "A", A, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        
        # Golden output único para verificación
        write_array(f, "y_expected", y_expected, interleaved=False)
        
        f.write("#endif\n")

    # 4. Volcado comprimido (.npz) listo para el nuevo script de test
    npz_payload = {
        "A": A, 
        "x": x, 
        "M": M, 
        "N": N, 
        "y_expected": y_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"ATAX Fused Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate ATAX Data without tmp array (1D Arrays)")
    parser.add_argument("--M", type=int, required=True, help="Dimension M (filas de A)")
    parser.add_argument("--N", type=int, required=True, help="Dimension N (columnas de A / tamaño de x)")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(M=args.M, N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()