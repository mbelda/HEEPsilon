#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def read_data(csv_file):
    data = []

    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                entry = {
                    "function": row["function"],
                    "size": row["size"],
                    "cc": float(row.get("cc", 0)),
                    "instr": float(row.get("instr", 0)),
                    "lds": float(row.get("lds", 0)),
                    "str": float(row.get("str", 0)),
                    "ld_stalls": float(row.get("ld_stalls", 0)),
                    "pipe_stalls": float(row.get("pipe_stalls", 0)),
                    "cpi": float(row.get("cpi", 0)),
                    "loads_per_instr": float(row.get("loads_per_instr", 0)),
                    "loads_per_cycle": float(row.get("loads_per_cycle", 0)),
                    "stores_per_instr": float(row.get("stores_per_instr", 0)),
                    "stalls_per_instr": float(row.get("stalls_per_instr", 0)),
                    "pipe_stalls_ratio": float(row.get("pipe_stalls_ratio", 0)),
                    "memory_intensity": float(row.get("memory_intensity", 0)),
                    "classification": row.get("classification", "")
                }
                data.append(entry)
            except:
                continue

    return data

# ─────────────────────────────────────────────
# Scatter Roofline-like
# ─────────────────────────────────────────────
def plot_scatter(data, outdir, suffix):
    plt.figure()
    for d in data:
        plt.scatter(d["loads_per_instr"], d["cpi"])
        plt.text(d["loads_per_instr"], d["cpi"], d["function"], fontsize=8)

    plt.xlabel("Loads per Instruction (Memory Intensity)")
    plt.ylabel("CPI")
    plt.title("Kernel Behaviour (Memory vs Compute)")
    plt.grid(True)

    plt.savefig(outdir / f"scatter_memory_vs_cpi_{suffix}.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────
# CPI Bar chart
# ─────────────────────────────────────────────
def plot_cpi_bars(data, outdir, suffix):
    labels = [f'{d["function"]}\n({d["size"]})' for d in data]
    cpis = [d["cpi"] for d in data]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, cpis)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("CPI")
    plt.title("CPI per Kernel Instance")
    plt.tight_layout()
    plt.savefig(outdir / f"cpi_per_kernel_{suffix}.png", dpi=150)
    plt.close()

def sum_cycles_per_function(csv_file):
    """
    Suma todos los ciclos ('cc') por función y devuelve lista ordenada por ciclos descendente.
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

# ─────────────────────────────────────────────
# PIE CHART — tiempo total por función
# ─────────────────────────────────────────────
def plot_time_pie(sorted_funcs, outdir, suffix, threshold=0.01):
    """
    Pie chart de tiempo total por función usando datos ya sumados.
    La tabla siempre muestra todas las funciones.
    """
    func_cycles = dict(sorted_funcs)
    total = sum(func_cycles.values())
    pie_labels = []
    pie_sizes = []
    others_cycles = 0
    others_names = []

    for func, cycles in func_cycles.items():
        frac = cycles / total
        if frac >= threshold:
            pie_labels.append(func)
            pie_sizes.append(cycles)
        else:
            others_cycles += cycles
            others_names.append(func)

    if others_cycles > 0:
        pie_labels.append("Otros")
        pie_sizes.append(others_cycles)

    plt.figure(figsize=(10, 10))
    plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    plt.title("Execution Time Distribution by Function")

    if others_names:
        legend_text = "\n".join(others_names)
        plt.gcf().text(0.95, 0.05, f"Otros incluye:\n{legend_text}",
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # ───── Tabla con todas las funciones (exacto como sum_cycles_per_function) ─────
    table_text = "\n".join([f"{f}: {v / total * 100:.2f}%" for f, v in sorted_funcs])
    plt.gcf().text(0.05, 0.05, table_text,
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / f"time_distribution_pie_{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()







def extract_suffix(input_csv):
    # Extrae lo que sigue a "analyzed_" en el nombre del CSV
    name = Path(input_csv).stem
    if "analyzed_" in name:
        return name.split("analyzed_", 1)[1]
    return "default"

def main():
    parser = argparse.ArgumentParser(description="Generar gráficas de rendimiento de kernels")
    parser.add_argument("analysis_csv", help="CSV generado por analyze_kernels.py (ej: analyzed_v1.csv)")
    parser.add_argument("--outdir", default="plots", help="Directorio de salida (default: plots)")
    args = parser.parse_args()

    if not Path(args.analysis_csv).exists():
        print("ERROR: no existe el analysis.csv")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    data = read_data(args.analysis_csv)
    if not data:
        print("No hay datos válidos")
        return

    suffix = extract_suffix(args.analysis_csv)

    # Scatter y CPI
    plot_scatter(data, outdir, suffix)
    plot_cpi_bars(data, outdir, suffix)

    # Pie chart: primero sumamos ciclos por función
    sorted_funcs = sum_cycles_per_function(args.analysis_csv)
    plot_time_pie(sorted_funcs, outdir, suffix, threshold=0.01)  # threshold solo afecta al gráfico, no a la tabla

    print("✔ Gráficas generadas en:", outdir.resolve())



if __name__ == "__main__":
    main()
