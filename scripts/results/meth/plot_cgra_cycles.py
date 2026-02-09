import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_speedup_vs_colsB(csv_file, output_image):
    # Cargar el CSV en un DataFrame
    df = pd.read_csv(csv_file)

    # Ordenar por ColsA y ColsB para una mejor visualización
    df = df.sort_values(by=["ColsA", "ColsB"])

    # Obtener los valores únicos de ColsA
    unique_colsA = df["ColsA"].unique()

    # Crear la gráfica
    plt.figure(figsize=(8, 6))

    # Iterar sobre cada grupo de ColsA y graficarlo sin marcadores
    for colsA in unique_colsA:
        subset = df[df["ColsA"] == colsA]
        plt.plot(subset["ColsB"], subset["CGRA"], linestyle='-', label=f"ColsA = {colsA}")

    # Configuración del gráfico
    plt.xlabel("ColsB")
    plt.ylabel("CGRA (Ciclos de Ejecución)")
    plt.title("CGRA vs ColsB para distintos ColsA")
    plt.grid(True)
    plt.legend(title="ColsA")

    # Guardar la imagen
    plt.savefig(output_image)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Genera una gráfica de CGRA vs ColsB diferenciada por ColsA.")
    parser.add_argument("csv_file", help="Ruta del archivo CSV de entrada.")
    parser.add_argument("output_image", help="Ruta de la imagen de salida (.png, .jpg, etc.)")

    args = parser.parse_args()
    
    plot_speedup_vs_colsB(args.csv_file, args.output_image)

if __name__ == "__main__":
    main()
