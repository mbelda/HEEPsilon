#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv_auto(filename):
    try:
        df = pd.read_csv(filename)
        if df.shape[1] == 1:
            df = pd.read_csv(filename, sep="\t")
    except:
        df = pd.read_csv(filename, sep="\t")
    return df


# normaliza texto: ignora espacios, mayúsculas y _
def normalize(s):
    return s.strip().lower().replace("_", "").replace(" ", "")


def map_columns(df):
    mapping = {}
    for c in df.columns:
        n = normalize(c)
        if n == "sizes":
            mapping["sizes"] = c
        elif n == "cpu":
            mapping["cpu"] = c
        elif n == "oe":
            mapping["oe"] = c
        elif n == "discomem":
            mapping["disco_mem"] = c
        elif n == "discorcs":
            mapping["disco_rcs"] = c

    missing = {"sizes", "cpu", "oe", "disco_mem", "disco_rcs"} - set(mapping.keys())
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {missing}")

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_file")
    args = parser.parse_args()

    df = read_csv_auto(args.input_csv)
    cols = map_columns(df)

    # --- extraer columnas SIN reordenar ---
    sizes = df[cols["sizes"]].astype(str)
    cpu = df[cols["cpu"]].astype(float)
    oe = df[cols["oe"]].astype(float)
    disco_mem = df[cols["disco_mem"]].astype(float)
    disco_rcs = df[cols["disco_rcs"]].astype(float)
    disco_total = disco_mem + disco_rcs

    # --- speedups ---
    speedup_oe = cpu / oe
    speedup_disco = cpu / disco_total

    x = np.arange(len(sizes))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # -------------------- BARRAS (CICLOS)
    ax1.bar(x - width, cpu, width,
            label="CPU",
            color="#d62728",
            zorder=2)

    ax1.bar(x, oe, width,
            label="OE",
            color="#2ca02c",
            zorder=2)

    # DISCO apilado
    ax1.bar(x + width, disco_mem, width,
            label="DISCO mem",
            color="#8ecae6",
            zorder=2)

    ax1.bar(x + width, disco_rcs, width,
            bottom=disco_mem,
            label="DISCO rcs",
            color="#219ebc",
            zorder=2)

    # eje izquierdo
    ax1.set_ylabel("Cycles")
    ax1.set_xlabel("Configuration (sizes)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)

    # etiquetas largas
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    plt.subplots_adjust(bottom=0.18)

    # IMPORTANTÍSIMO para arquitectura
    ax1.set_yscale("log")

    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # -------------------- SPEEDUP
    ax2 = ax1.twinx()
    ax2.set_ylabel("Speedup vs CPU")

    ax2.plot(x, speedup_oe,
             marker="o",
             linewidth=2.5,
             color="black",
             label="Speedup OE",
             zorder=5)

    ax2.plot(x, speedup_disco,
             marker="s",
             linewidth=2.5,
             color="purple",
             label="Speedup DISCO",
             zorder=5)

    ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1)

    max_speedup = max(speedup_oe.max(), speedup_disco.max())
    ax2.set_ylim(0, max_speedup * 1.25)

    # -------------------- LEYENDA
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    plt.close()

    print("Generated:", args.output_file)


if __name__ == "__main__":
    main()