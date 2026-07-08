#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

def gemver_p1_reference(A, u1, v1, u2, v2, N):
    """Calcula la Parte 1 de GEMVER: A = A + u1*v1^T + u2*v2^T"""
    A_out = A.copy().reshape((N, N))
    
    for i in range(N):
        for j in range(N):
            A_out[i, j] += u1[i] * v1[j] + u2[i] * v2[j]
            
    return A_out.flatten()

def gemver_p2_reference(A_modified, x, y, z, beta, N):
    """Calcula la Parte 2 de GEMVER: x = x + beta * A_modified^T * y + z"""
    A_mat = A_modified.reshape((N, N))
    x_out = x.copy()
    
    for i in range(N):
        aux = 0
        for j in range(N):
            # Acceso traspuesto (j, i)
            aux += beta * A_mat[j, i] * y[j]
        x_out[i] += aux + z[i]
        
    return x_out

def gemver_p3_reference(A_modified, x_modified, w, alpha, N):
    """Calcula la Parte 3 de GEMVER: w = w + alpha * A_modified * x_modified"""
    A_mat = A_modified.reshape((N, N))
    w_out = w.copy()
    
    for i in range(N):
        aux = 0
        for j in range(N):
            # Acceso normal (i, j) pero usando el vector x ya modificado de la P2
            aux += alpha * A_mat[i, j] * x_modified[j]
        w_out[i] += aux
        
    return w_out

def generate_and_write(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(N)
    data_dir = Path(".")
    h_filename = data_dir / f"data_{N}.h"
    npz_filename = data_dir / f"data_{N}.npz"

    # 1. Generación de escalares de configuración
    beta = np.random.randint(1, 5)
    alpha = np.random.randint(1, 5)

    # 2. Generación de estructuras de datos iniciales
    A = np.random.randint(-5, 6, size=(N * N), dtype=np.int32)
    u1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    v1 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    u2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    v2 = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    
    x = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    y = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    z = np.random.randint(-5, 6, size=(N,), dtype=np.int32)
    
    w = np.random.randint(-5, 6, size=(N,), dtype=np.int32)

    # 3. Pipeline de ejecución secuencial encadenado (Flujo de dependencias reales)
    A_expected = gemver_p1_reference(A, u1, v1, u2, v2, N)
    x_expected = gemver_p2_reference(A_expected, x, y, z, beta, N)
    w_expected = gemver_p3_reference(A_expected, x_expected, w, alpha, N)

    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    # 4. Escritura del C Header (.h) global unificado
    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_GEMVER_FUSED_H\n")
        f.write("#define DATA_GEMVER_FUSED_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* Dataset Size: {N} */\n")
        f.write(f"#define N {N}\n")
        f.write(f"#define BETA {beta}\n")
        f.write(f"#define ALPHA {alpha}\n\n")

        # Inputs del sistema completo
        write_array(f, "A", A, interleaved=False)
        write_array(f, "u1", u1, interleaved=False)
        write_array(f, "v1", v1, interleaved=False)
        write_array(f, "u2", u2, interleaved=False)
        write_array(f, "v2", v2, interleaved=False)
        write_array(f, "x", x, interleaved=False)
        write_array(f, "y", y, interleaved=False)
        write_array(f, "z", z, interleaved=False)
        write_array(f, "w", w, interleaved=False)
        
        # Golden outputs para verificación por etapas o final
        write_array(f, "A_expected", A_expected, interleaved=False)
        write_array(f, "x_expected", x_expected, interleaved=False)
        write_array(f, "w_expected", w_expected, interleaved=False)
        
        f.write("#endif\n")

    # 5. Volcado comprimido (.npz) para las simulaciones y el simulador de trazas del CGRA
    npz_payload = {
        "A": A, "u1": u1, "v1": v1, "u2": u2, "v2": v2,
        "x": x, "y": y, "z": z, "w": w, 
        "beta": beta, "alpha": alpha,
        "NI": N, "NJ": N, 
        "A_expected": A_expected,
        "x_expected": x_expected,
        "w_expected": w_expected
    }
    np.savez(npz_filename, **npz_payload)
    print(f"Full Fused Dataset (P1 + P2 + P3) Generated -> {h_filename}, {npz_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate Full Fused GEMVER (P1 + P2 + P3) Data")
    parser.add_argument("--N", type=int, required=True, help="Matrix/vector size N")
    parser.add_argument("--seed", type=int, default=3, help="Random generation seed")
    args = parser.parse_args()
    generate_and_write(N=args.N, seed=args.seed)

if __name__ == "__main__":
    main()