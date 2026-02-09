#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path


def parse_log(input_path, output_csv):

    # Detecta inicio de kernel
    # Acepta:
    #   MatMul (121x4x121)
    #   MatMul (121x4x121
    #   MatMul 121x4x121
    func_pattern = re.compile(
        r'^([A-Za-z0-9_]+)\s*(?:\(\s*([0-9xX]+)\s*\)?|([0-9xX]+))'
    )

    cc_pattern = re.compile(r'^Cc:\s*(\d+)')
    instr_pattern = re.compile(r'^Instr:\s*(\d+)')
    lds_pattern = re.compile(r'^Lds:\s*(\d+)')

    results = []
    current = None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # ───── Detectar nueva función ─────
            m = func_pattern.match(line)
            if m:
                # Guardar anterior
                if current:
                    results.append(current)

                func_name = m.group(1)

                # dimensiones pueden estar en grupo2 o grupo3
                dimensions = m.group(2) if m.group(2) else m.group(3)
                dimensions = dimensions if dimensions else ""

                current = {
                    "function": func_name,
                    "size": dimensions,
                    "cc": "",
                    "instr": "",
                    "lds": ""
                }
                continue

            if not current:
                continue

            # ───── Métricas ─────
            m = cc_pattern.match(line)
            if m:
                current["cc"] = m.group(1)
                continue

            m = instr_pattern.match(line)
            if m:
                current["instr"] = m.group(1)
                continue

            m = lds_pattern.match(line)
            if m:
                current["lds"] = m.group(1)
                continue

    # Guardar último kernel
    if current:
        results.append(current)

    # ───── Escribir CSV ─────
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["function", "size", "cc", "instr", "lds"])

        for r in results:
            writer.writerow([r["function"], r["size"], r["cc"], r["instr"], r["lds"]])

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
