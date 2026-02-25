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


def sanitize(name):
    return name.replace(" ", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_prefix")
    args = parser.parse_args()

    df = read_csv_auto(args.input_csv)
    df.columns = [c.strip() for c in df.columns]

    show_legend = True

    for _, row in df.iterrows():

        size = str(row["kernel size"])

        cpu = float(row["CPU"])
        oe = float(row["OE"])
        disco_mem = float(row["DISCO mem"])
        disco_rcs = float(row["DISCO rcs"])
        disco_total = disco_mem + disco_rcs

        # Speedups
        speedup_oe = cpu / oe
        speedup_disco = cpu / disco_total

        fig, ax1 = plt.subplots(figsize=(6, 5))

        labels = ["CPU", "OE", "DISCO"]
        x = np.arange(len(labels))
        width = 0.6

        # -------------------------
        # BARRAS
        # -------------------------
        ax1.bar(x[0], cpu, width, color="#d62728", label="CPU")
        ax1.bar(x[1], oe, width, color="#2ca02c", label="OE")

        ax1.bar(
            x[2],
            disco_mem,
            width,
            color="#8ecae6",   # azul claro
            label="DISCO mem"
        )

        ax1.bar(
            x[2],
            disco_rcs,
            width,
            bottom=disco_mem,
            color="#023047",   # azul oscuro
            label="DISCO rcs"
        )

        ax1.set_ylabel("Cycles")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title(f"Matrix size {size}")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # -------------------------
        # SPEEDUP (EJE DERECHO)
        # -------------------------
        ax2 = ax1.twinx()
        ax2.set_ylabel("Speedup vs CPU")

        # Solo OE y DISCO
        x_speedup = [x[1], x[2]]
        y_speedup = [speedup_oe, speedup_disco]

        ax2.plot(
            x_speedup,
            y_speedup,
            marker="o",
            linestyle="-",
            linewidth=2,
            color="black",
            label="Speedup"
        )

        ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1)

        # Escala automática con margen
        max_speedup = max(speedup_oe, speedup_disco)
        ax2.set_ylim(0, max_speedup * 1.2)

        # -------------------------
        # LEYENDA (solo primera figura)
        # -------------------------
        if show_legend:
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
            show_legend = False

        plt.tight_layout()

        filename = f"{args.output_prefix}_{sanitize(size)}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print("Generado:", filename)


if __name__ == "__main__":
    main()
