#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path

def parse_log(input_path, output_csv):

    # Función: acepta múltiples palabras, con o sin paréntesis, dimensiones opcionales
    func_pattern = re.compile(
        r'^([A-Za-z ]+?)\s*(?:\(\s*([0-9xX]+)\s*\)?|([0-9xX]+))?$'
    )

    # Métricas
    metrics_patterns = {
        "cc": re.compile(r'^Cc:\s*(\d+)'),
        "instr": re.compile(r'^Instr:\s*(\d+)'),
        "lds": re.compile(r'^Lds:\s*(\d+)'),
        "str": re.compile(r'^Str:\s*(\d+)'),
        "ld_stalls": re.compile(r'^Ld Stalls:\s*(\d+)'),
        "pipe_stalls": re.compile(r'^Pipe stalls:\s*(\d+)')
    }

    results = []
    current = None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # ───── Detectar nueva función ─────
            m = func_pattern.match(line)
            if m:
                # Guardar la función anterior
                if current:
                    results.append(current)

                func_name = m.group(1).strip()
                dimensions = m.group(2) if m.group(2) else m.group(3)
                dimensions = dimensions if dimensions else ""

                current = {
                    "function": func_name,
                    "size": dimensions,
                    "cc": "",
                    "instr": "",
                    "lds": "",
                    "str": "",
                    "ld_stalls": "",
                    "pipe_stalls": ""
                }
                continue

            if not current:
                continue

            # ───── Capturar métricas ─────
            for key, pattern in metrics_patterns.items():
                m = pattern.match(line)
                if m:
                    current[key] = m.group(1)
                    break  # una línea corresponde a una sola métrica

    # Guardar el último bloque
    if current:
        results.append(current)

    # ───── Escribir CSV ─────
    fieldnames = ["function", "size"] + list(metrics_patterns.keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"✔ Datos guardados en {output_csv}")
    print(f"  Bloques procesados: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Parsear logs de kernels y generar CSV")
    parser.add_argument("input_log", help="Ruta al fichero de log")
    parser.add_argument("output_csv", help="Nombre del CSV de salida")
    args = parser.parse_args()

    if not Path(args.input_log).exists():
        print(f"ERROR: No existe el fichero {args.input_log}")
        return

    parse_log(args.input_log, args.output_csv)

if __name__ == "__main__":
    main()
