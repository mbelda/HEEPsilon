#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

# -------------------------------------------------
# Reference computation (CGRA 3-Loop Kernel)
# -------------------------------------------------
def cgra_kernel_3loops_reference(
    ptr_in_lider,
    ptr_w,
    channels_per_group,
    kernel_h,
    kernel_w,
    input_w,
    tam_canal_input
):
    """
    Simulación exacta bit-a-bit del recorrido de punteros de la función C.
    Calcula la suma acumulada (dot product con saltos espaciales).
    """
    sum_val = 0
    idx_in = 0
    idx_w = 0

    salto_fila_filtro = input_w - kernel_w
    salto_siguiente_canal = tam_canal_input - (kernel_h * input_w)

    for c in range(channels_per_group):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Multiplicación y acumulación
                sum_val += int(ptr_in_lider[idx_in]) * int(ptr_w[idx_w])
                
                # Desplazamiento de punteros (+1 elemento = +4 bytes en C)
                idx_in += 1
                idx_w += 1
            
            # Fin de fila del filtro: salto
            idx_in += salto_fila_filtro
        
        # Fin de canal: salto al siguiente canal
        idx_in += salto_siguiente_canal

    # Forzar desborde de 32 bits con signo (mismo comportamiento que int32_t en C)
    return np.int32(sum_val)


# -------------------------------------------------
# Generate + write
# -------------------------------------------------
def generate_and_write(channels, kernel_h, kernel_w, input_h, input_w, seed=None):

    if seed is not None:
        np.random.seed(seed)

    channels = int(channels)
    kernel_h = int(kernel_h)
    kernel_w = int(kernel_w)
    input_h = int(input_h)
    input_w = int(input_w)
    
    # Parámetros derivados
    tam_canal_input = input_h * input_w
    size_input = channels * tam_canal_input
    size_weights = channels * kernel_h * kernel_w

    data_dir = Path(".")
    suffix = f"c{channels}_kh{kernel_h}_kw{kernel_w}_iw{input_w}"
    h_filename = data_dir / f"data_{suffix}.h"
    npz_filename = data_dir / f"data_{suffix}.npz"

    # Generación de datos aleatorios acotados (para evitar desbordamientos masivos si no se desea)
    input_lider = np.random.randint(-10, 11, size=size_input, dtype=np.int32)
    weights = np.random.randint(-10, 11, size=size_weights, dtype=np.int32)
    
    # Cálculo del valor de salida esperado
    expected_sum = cgra_kernel_3loops_reference(
        input_lider,
        weights,
        channels,
        kernel_h,
        kernel_w,
        input_w,
        tam_canal_input
    )

    # -------------------------------------------------
    # Write C header
    # -------------------------------------------------
    def write_array(f, name, arr, interleaved=False):
        attr = ' __attribute__((section(".xheep_data_interleaved")))' if interleaved else ''
        f.write(f"int32_t {name}[{len(arr)}]{attr} = {{\n    ")
        f.write(", ".join(str(v) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:
        f.write("#ifndef DATA_H\n")
        f.write("#define DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"/* Dataset Dimensions */\n")
        f.write(f"#define CHANNELS_PER_GROUP {channels}\n")
        f.write(f"#define KERNEL_H           {kernel_h}\n")
        f.write(f"#define KERNEL_W           {kernel_w}\n")
        f.write(f"#define INPUT_H            {input_h}\n")
        f.write(f"#define INPUT_W            {input_w}\n")
        f.write(f"#define TAM_CANAL_INPUT    {tam_canal_input}\n\n")

        # Variables de entrada y salida esperada
        write_array(f, "input_lider", input_lider, interleaved=False)
        write_array(f, "weights", weights, interleaved=False)
        
        f.write(f"/* Salida esperada */\n")
        f.write(f"const int32_t EXPECTED_SUM = {expected_sum};\n\n")

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ for CGRA simulator
    # -------------------------------------------------
    npz_payload = {
        "input_lider": input_lider,
        "weights": weights,
        "channels_per_group": channels,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "input_h": input_h,
        "input_w": input_w,
        "tam_canal_input": tam_canal_input,
        "expected_sum": expected_sum
    }

    np.savez(npz_filename, **npz_payload)

    print(f"Generated CGRA 3-Loop Kernel Data ({channels} ch, {kernel_h}x{kernel_w} kernel, {input_h}x{input_w} input):")
    print(f"  -> {h_filename}")
    print(f"  -> {npz_filename}")
    print(f"  -> Expected Sum: {expected_sum}")


def main():
    parser = argparse.ArgumentParser(description="Generate Data for CGRA 3-loop kernel")
    
    parser.add_argument("--channels", type=int, required=True, help="Number of channels per group (C)")
    parser.add_argument("--kh", type=int, required=True, help="Kernel Height (KH)")
    parser.add_argument("--kw", type=int, required=True, help="Kernel Width (KW)")
    parser.add_argument("--ih", type=int, required=True, help="Input Height (IH)")
    parser.add_argument("--iw", type=int, required=True, help="Input Width (IW)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validación básica de dimensiones
    if args.kh > args.ih or args.kw > args.iw:
        parser.error("El tamaño del kernel (KHxKW) no puede ser mayor que el de la entrada (IHxIW).")

    generate_and_write(
        channels=args.channels,
        kernel_h=args.kh,
        kernel_w=args.kw,
        input_h=args.ih,
        input_w=args.iw,
        seed=args.seed
    )

if __name__ == "__main__":
    main()