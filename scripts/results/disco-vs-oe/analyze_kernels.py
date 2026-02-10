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

            cc = safe_float(row.get("cc", 0))
            instr = safe_float(row.get("instr", 0))
            lds = safe_float(row.get("lds", 0))
            str_ = safe_float(row.get("str", 0))
            ld_stalls = safe_float(row.get("ld_stalls", 0))
            pipe_stalls = safe_float(row.get("pipe_stalls", 0))

            if instr == 0 or cc == 0:
                continue

            # ───── Métricas principales ─────
            cpi = cc / instr
            loads_per_instr = lds / instr
            loads_per_cycle = lds / cc
            stores_per_instr = str_ / instr
            stalls_per_instr = ld_stalls / instr
            pipe_stalls_ratio = pipe_stalls / cc

            # aproximación intensidad memoria
            memory_intensity = loads_per_instr

            classification = classify_kernel(cpi, loads_per_instr)

            results.append({
                "function": function,
                "size": size,
                "cc": int(cc),
                "instr": int(instr),
                "lds": int(lds),
                "str": int(str_),
                "ld_stalls": int(ld_stalls),
                "pipe_stalls": int(pipe_stalls),
                "cpi": round(cpi, 3),
                "loads_per_instr": round(loads_per_instr, 3),
                "loads_per_cycle": round(loads_per_cycle, 3),
                "stores_per_instr": round(stores_per_instr, 3),
                "stalls_per_instr": round(stalls_per_instr, 3),
                "pipe_stalls_ratio": round(pipe_stalls_ratio, 3),
                "memory_intensity": round(memory_intensity, 3),
                "classification": classification
            })

    # ───── Guardar CSV ─────
    fieldnames = [
        "function", "size", "cc", "instr", "lds", "str", "ld_stalls", "pipe_stalls",
        "cpi", "loads_per_instr", "loads_per_cycle",
        "stores_per_instr", "stalls_per_instr", "pipe_stalls_ratio",
        "memory_intensity", "classification"
    ]

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
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
