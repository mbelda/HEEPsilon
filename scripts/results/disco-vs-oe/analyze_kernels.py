#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


def classify_kernel(cpi, loads_per_instr):
    """
    Clasificación heurística:
    - Mucho acceso a memoria → memory bound
    - Poco acceso pero CPI alto → compute bound
    """

    if loads_per_instr > 1.2:
        return "memory-bound"

    if cpi > 2.0 and loads_per_instr < 0.6:
        return "compute-bound"

    return "balanced"


def analyze(input_csv, output_csv):

    results = []

    with open(input_csv, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            function = row["function"]
            size = row["size"]

            cc = safe_float(row["cc"])
            instr = safe_float(row["instr"])
            lds = safe_float(row["lds"])

            if instr == 0 or cc == 0:
                continue

            # ───── Métricas ─────
            cpi = cc / instr
            loads_per_instr = lds / instr
            loads_per_cycle = lds / cc

            # aproximación intensidad memoria
            # alto = mucho acceso memoria
            memory_intensity = loads_per_instr

            classification = classify_kernel(cpi, loads_per_instr)

            results.append({
                "function": function,
                "size": size,
                "cc": int(cc),
                "instr": int(instr),
                "lds": int(lds),
                "cpi": round(cpi, 3),
                "loads_per_instr": round(loads_per_instr, 3),
                "loads_per_cycle": round(loads_per_cycle, 3),
                "memory_intensity": round(memory_intensity, 3),
                "classification": classification
            })

    # ───── Guardar CSV ─────
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        fieldnames = [
            "function", "size", "cc", "instr", "lds",
            "cpi", "loads_per_instr", "loads_per_cycle",
            "memory_intensity", "classification"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(r)

    print(f"✔ Análisis guardado en {output_csv}")
    print(f"  Kernels analizados: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Analiza métricas de kernels desde CSV")
    parser.add_argument("input_csv", help="CSV generado por parse_logs.py")
    parser.add_argument("output_csv", help="CSV de salida con métricas")

    args = parser.parse_args()

    if not Path(args.input_csv).exists():
        print("ERROR: no existe el CSV de entrada")
        return

    analyze(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
