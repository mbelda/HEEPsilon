#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_auto(filename):
    try:
        df = pd.read_csv(filename)
        if df.shape[1] == 1:
            df = pd.read_csv(filename, sep='\t')
    except:
        df = pd.read_csv(filename, sep='\t')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_image")
    args = parser.parse_args()

    df = read_csv_auto(args.input_csv)
    df.columns = [c.strip() for c in df.columns]

    if len(df) != 1:
        raise ValueError("El CSV debe tener una única fila")

    row = df.iloc[0]

    baseline = float(row["CPU APROX"])

    # columnas a mostrar
    fields = ["CPU APROX", "CPU APROX 2.0", "OE", "DISCO"]

    cycles = np.array([float(row[f]) for f in fields])

    # métricas
    norm_cycles = cycles / baseline
    speedup = baseline / cycles

    x = np.arange(len(fields))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # -------------------------
    # BARRAS (eje izquierdo)
    # -------------------------
    bars = ax1.bar(
        x,
        norm_cycles,
        width=0.6,
        color="#6baed6",
        edgecolor="black",
        zorder=1   # <- detrás
    )

    ax1.set_ylabel("Normalized cycles (vs CPU APROX)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fields, rotation=25)
    ax1.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # -------------------------
    # SPEEDUP (eje derecho)
    # -------------------------
    ax2 = ax1.twinx()

    line = ax2.plot(
        x,
        speedup,
        color="#ff7f0e",        # naranja visible
        marker="o",
        markersize=8,
        linewidth=3,
        label="Speedup",
        zorder=5                # <- delante de todo
    )

    ax2.set_ylabel("Speedup (vs CPU APROX)")
    ax2.axhline(1.0, linestyle="--", linewidth=1, color="gray")

    # Escala razonable
    ax2.set_ylim(0, max(speedup) * 1.25)

    # pequeña mejora visual: valores encima de cada punto
    for i, val in enumerate(speedup):
        ax2.text(
            x[i],
            val,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#ff7f0e",
            zorder=6
        )

    plt.title("Transformer Performance (Normalized to CPU APROX)")
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300)
    plt.close()

    print("Figura guardada en:", args.output_image)


if __name__ == "__main__":
    main()
