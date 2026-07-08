#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def test_reference(A, y1, x1_init, N):
    """
    Calcula la referencia del bucle:
    for i in range(N):
        acc = 0
        for j in range(N):
            acc += A[i * N + j] * y1[j]
        x1[i] += acc
    """
    # Copiamos para no modificar el array inicial
    x1_out = np.copy(x1_init)
    
    for i in range(N):
        acc = 0
        for j in range(N):
            acc += A[i * N + j] * y1[j]
        x1_out[i] += acc
            
    return x1_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_test_{N}.h"
    npz_filename = data_dir / f"data_test_{N}.npz"

    # 1. Generación de estructuras de datos iniciales (Inputs)
    # Matriz A de tamaño N*N, vectores y1 y x1 de tamaño N
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    y1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    x1_init = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 2. Ejecución de la referencia para obtener el Golden Output (x1_expected)
    x1_expected = test_reference(A, y1, x1_init, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 3. Escritura del C Header (.h) unificado
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_TEST_H\n")
        f.write("#define DATA_TEST_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: N={N} (Matrix N x N) */\n")
        f.write(f"#define N {N}\n\n")

        # Inputs del sistema
        write_array(f, "A", A, interleaved=False)
        write_array(f, "y1", y1, interleaved=False)
        write_array(f, "x1", x1_init, interleaved=False) # El array en C se llamará x1 e iniciará con estos valores
        
        # Golden output esperado para la verificación de tu test
        write_array(f, "x1_expected", x1_expected, interleaved=False)
        
        f.write("#endif\n")

    # 4. Volcado comprimido (.npz) para verificación/debug en Python
    npz_payload = {
        "A": A, 
        "y1": y1, 
        "x1_init": x1_init,
        "N": N, 
        "x1_expected": x1_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"Dataset Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate Data for Custom Gemv-Accumulate Test")
    parser.add_argument("--N", type=int, required=True, help="Dimension N (Matriz NxN, Vectores tamaño N)")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()