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
                    "cc": float(row["cc"]),
                    "instr": float(row["instr"]),
                    "lds": float(row["lds"]),
                    "cpi": float(row["cpi"]),
                    "loads_per_instr": float(row["loads_per_instr"]),
                    "loads_per_cycle": float(row["loads_per_cycle"]),
                    "classification": row["classification"]
                }
                data.append(entry)
            except:
                continue

    return data


# ─────────────────────────────────────────────
# Scatter Roofline-like
# ─────────────────────────────────────────────
def plot_scatter(data, outdir):
    plt.figure()

    for d in data:
        plt.scatter(d["loads_per_instr"], d["cpi"])
        plt.text(d["loads_per_instr"], d["cpi"], d["function"], fontsize=8)

    plt.xlabel("Loads per Instruction (Memory Intensity)")
    plt.ylabel("CPI")
    plt.title("Kernel Behaviour (Memory vs Compute)")
    plt.grid(True)

    plt.savefig(outdir / "scatter_memory_vs_cpi.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# CPI Bar chart
# ─────────────────────────────────────────────
def plot_cpi_bars(data, outdir):
    labels = [f'{d["function"]}\n({d["size"]})' for d in data]
    cpis = [d["cpi"] for d in data]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, cpis)
    plt.xticks(rotation=60, ha="right")

    plt.ylabel("CPI")
    plt.title("CPI per Kernel Instance")

    plt.tight_layout()
    plt.savefig(outdir / "cpi_per_kernel.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# PIE CHART — tiempo total por función
# ─────────────────────────────────────────────
def plot_time_pie(data, outdir):
    total_cycles = defaultdict(float)

    # Agrupar por nombre de función
    for d in data:
        total_cycles[d["function"]] += d["cc"]

    labels = []
    sizes = []

    for func, cycles in total_cycles.items():
        labels.append(func)
        sizes.append(cycles)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Execution Time Distribution by Function")

    plt.savefig(outdir / "time_distribution_pie.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generar gráficas de rendimiento de kernels")
    parser.add_argument("analysis_csv", help="CSV generado por analyze_kernels.py")
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

    plot_scatter(data, outdir)
    plot_cpi_bars(data, outdir)
    plot_time_pie(data, outdir)

    print("✔ Gráficas generadas en:", outdir.resolve())


if __name__ == "__main__":
    main()
