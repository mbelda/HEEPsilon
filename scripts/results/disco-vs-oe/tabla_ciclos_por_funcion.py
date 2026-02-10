import csv
from collections import defaultdict
from pathlib import Path

#!/usr/bin/env python3
import argparse

def sum_cycles_per_function(csv_file):
    """
    Suma todos los ciclos ('cc') por función y muestra tabla con porcentaje.
    """
    func_cycles = defaultdict(float)
    total_cycles = 0

    # Leer CSV
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            func = row.get("function", "Unknown")
            cc = float(row.get("cc", 0))
            func_cycles[func] += cc
            total_cycles += cc

    # Ordenar por ciclos descendente
    sorted_funcs = sorted(func_cycles.items(), key=lambda x: x[1], reverse=True)

    # Imprimir tabla
    print(f"{'Function':25s} | {'Cycles':>12s} | {'% Total':>8s}")
    print("-" * 50)
    for func, cycles in sorted_funcs:
        pct = (cycles / total_cycles * 100) if total_cycles > 0 else 0
        print(f"{func:25s} | {int(cycles):12d} | {pct:7.2f}%")

    return sorted_funcs

def main():
    parser = argparse.ArgumentParser(description="Generar gráficas de rendimiento de kernels")
    parser.add_argument("analysis_csv", help="CSV generado por analyze_kernels.py (ej: analyzed_v1.csv)")
    parser.add_argument("--outdir", default="plots", help="Directorio de salida (default: plots)")
    args = parser.parse_args()

    if not Path(args.analysis_csv).exists():
        print("ERROR: no existe el analysis.csv")
        return

    sum_cycles_per_function(args.analysis_csv)

# Ejemplo de uso
if __name__ == "__main__":
    main()


