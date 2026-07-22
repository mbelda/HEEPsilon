#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

# -------------------------------------------------
# Reference computation (Full vs CGRA-only)
# -------------------------------------------------
def compute_convolution_references(
    input_lider,
    weights,
    channels,
    kernel_h,
    kernel_w,
    input_h,
    input_w,
    tam_canal_input
):
    """
    Calcula tanto la suma total de la convolución (con bordes) 
    como la porción exacta que el CGRA procesará en su zona segura.
    """
    # Parámetros por defecto para el testeo unitario aislado
    stride_h = 1
    stride_w = 1
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    out_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    # Límites de la zona segura del CGRA
    oh_seguro_inicio = (pad_h + stride_h - 1) // stride_h
    oh_seguro_fin = (input_h + pad_h - kernel_h) // stride_h + 1
    if oh_seguro_fin < oh_seguro_inicio: oh_seguro_fin = oh_seguro_inicio

    ow_seguro_inicio = (pad_w + stride_w - 1) // stride_w
    ow_seguro_fin = (input_w + pad_w - kernel_w) // stride_w + 1
    if ow_seguro_fin < ow_seguro_inicio: ow_seguro_fin = ow_seguro_inicio

    global_sum = 0
    cgra_sum = 0

    # Reestructuramos temporalmente para facilitar el indexado espacial en Python
    # input: [channels, input_h, input_w]
    input_3d = input_lider.reshape(channels, input_h, input_w)
    # weights: [channels, kernel_h, kernel_w]
    weights_3d = weights.reshape(channels, kernel_h, kernel_w)

    for oh in range(out_h):
        for ow in range(out_w):
            
            # Verificar si esta coordenada de salida pertenece al CGRA
            es_zona_cgra = (oh_seguro_inicio <= oh < oh_seguro_fin) and (ow_seguro_inicio <= ow < ow_seguro_fin)
            
            # Solo los múltiplos de 16 entran al CGRA por canal
            canales_cgra = (channels // 16) * 16

            for icg in range(channels):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        
                        ih = oh * stride_h + kh - pad_h
                        int_iw = ow * stride_w + kw - pad_w

                        # Control de padding (borde exterior virtual)
                        if ih < 0 or ih >= input_h or int_iw < 0 or int_iw >= input_w:
                            continue

                        val_mac = int(input_3d[icg, ih, int_iw]) * int(weights_3d[icg, kh, kw])
                        
                        # Acumulación global (Toda la imagen)
                        global_sum += val_mac

                        # Acumulación exclusiva del CGRA:
                        # Debe estar en la zona segura espacial Y dentro de los canales múltiplos de 16
                        if es_zona_cgra and (icg < canales_cgra):
                            cgra_sum += val_mac

    return np.int32(global_sum), np.int32(cgra_sum)


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
    
    tam_canal_input = input_h * input_w
    size_input = channels * tam_canal_input
    size_weights = channels * kernel_h * kernel_w

    data_dir = Path(".")
    suffix = f"c{channels}_kh{kernel_h}_kw{kernel_w}_iw{input_w}"
    h_filename = data_dir / f"data_{suffix}.h"
    npz_filename = data_dir / f"data_{suffix}.npz"

    input_lider = np.random.randint(-10, 11, size=size_input, dtype=np.int32)
    weights = np.random.randint(-10, 11, size=size_weights, dtype=np.int32)
    
    # Cálculo de ambas referencias matemáticas (Total vs CGRA)
    expected_sum, expected_cgra_sum = compute_convolution_references(
        input_lider, weights, channels, kernel_h, kernel_w, input_h, input_w, tam_canal_input
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

        write_array(f, "input_lider", input_lider, interleaved=False)
        write_array(f, "weights", weights, interleaved=False)
        
        f.write(f"/* Salidas esperadas */\n")
        f.write(f"const int32_t EXPECTED_SUM      = {expected_sum};\n")
        f.write(f"const int32_t EXPECTED_CGRA_SUM = {expected_cgra_sum};\n\n")

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
        "expected_sum": expected_sum,
        "expected_cgra_sum": expected_cgra_sum
    }

    np.savez(npz_filename, **npz_payload)

    print(f"Generated CGRA 3-Loop Kernel Data ({channels} ch, {kernel_h}x{kernel_w} kernel, {input_h}x{input_w} input):")
    print(f"  -> {h_filename}")
    print(f"  -> {npz_filename}")
    print(f"  -> Expected Global Sum: {expected_sum}")
    print(f"  -> Expected CGRA-only Sum: {expected_cgra_sum}")


def main():
    parser = argparse.ArgumentParser(description="Generate Data for CGRA 3-loop kernel")
    
    parser.add_argument("--channels", type=int, required=True, help="Number of channels per group (C)")
    parser.add_argument("--kh", type=int, required=True, help="Kernel Height (KH)")
    parser.add_argument("--kw", type=int, required=True, help="Kernel Width (KW)")
    parser.add_argument("--ih", type=int, required=True, help="Input Height (IH)")
    parser.add_argument("--iw", type=int, required=True, help="Input Width (IW)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

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