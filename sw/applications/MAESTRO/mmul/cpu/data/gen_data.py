#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

# -------------------------------------------------
# Reference computation (Simple MMUL: C = A * B)
# -------------------------------------------------
def mmul_reference(A, B, NI, NK, NJ):
    """
    Computes standard integer MMUL (C_out = A * B) 
    using row-major flattened 1D arrays.
    A: NI x NK, B: NK x NJ, C_out: NI x NJ
    """
    C_out = np.zeros(NI * NJ, dtype=np.int32)

    for i in range(NI):
        for j in range(NJ):
            sum_val = 0
            for k in range(NK):
                sum_val += A[i * NK + k] * B[k * NJ + j]
            C_out[i * NJ + j] = sum_val

    return C_out


# -------------------------------------------------
# Generate + write
# -------------------------------------------------
def generate_and_write(NI, NK, NJ, seed=None):

    if seed is not None:
        np.random.seed(seed)

    NI = int(NI)
    NK = int(NK)
    NJ = int(NJ)
    data_dir = Path(".")

    # Usamos NI_NK_NJ para el nombre si difieren, o solo NI si es cuadrada
    suffix = f"{NI}" if (NI == NK == NJ) else f"{NI}_{NK}_{NJ}"
    h_filename = data_dir / f"data_{suffix}.h"
    npz_filename = data_dir / f"data_{suffix}.npz"

    # Matrices aplanadas en 1D (Row-Major). C ya no se genera como entrada.
    A = np.random.randint(-5, 6, size=(NI * NK), dtype=np.int32)
    B = np.random.randint(-5, 6, size=(NK * NJ), dtype=np.int32)
    
    # Cálculo de referencia (C_expected = A * B)
    C_expected = mmul_reference(A, B, NI, NK, NJ)

    # -------------------------------------------------
    # Write C header
    # -------------------------------------------------
    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_H\n")
        f.write("#define DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"/* Dataset Dimensions: NI={NI}, NK={NK}, NJ={NJ} */\n")
        f.write(f"#define NI {NI}\n")
        f.write(f"#define NK {NK}\n")
        f.write(f"#define NJ {NJ}\n\n")

        # Variables de entrada y salida esperada (eliminada la matriz C de entrada)
        write_array(f, "A", A, interleaved=False)
        write_array(f, "B", B, interleaved=False)
        write_array(f, "C_expected", C_expected, interleaved=False)

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ for CGRA simulator
    # -------------------------------------------------
    npz_payload = {
        "A": A,
        "B": B,
        "rowsA": NI,    
        "colsA": NK,    
        "colsB": NJ,    
        "C_expected": C_expected
    }

    np.savez(npz_filename, **npz_payload)

    print(f"Generated MMUL structure ({NI}x{NK}x{NJ}):")
    print(f"  -> {h_filename}")
    print(f"  -> {npz_filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate MMUL Data variants for X-HEEP and CGRA")
    
    # Permite especificar una NJ para matrices cuadradas, o definir NI, NK, NJ individualmente
    parser.add_argument("--NJ", type=int, required=True, help="Size dimension NJ (or default for NI and NK if they are not set)")
    parser.add_argument("--NI", type=int, help="Rows of A / Rows of C (defaults to NJ)")
    parser.add_argument("--NK", type=int, help="Cols of A / Rows of B (defaults to NJ)")
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()

    # Si NI o NK no se envían por parámetro, asume que es una matriz cuadrada usando NJ
    ni_dim = args.NI if args.NI is not None else args.NJ
    nk_dim = args.NK if args.NK is not None else args.NJ
    nj_dim = args.NJ

    generate_and_write(NI=ni_dim, NK=nk_dim, NJ=nj_dim, seed=args.seed)

if __name__ == "__main__":
    main()