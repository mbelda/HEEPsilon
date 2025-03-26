import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_speedup(csv_file, output_file):
    # Leer el CSV
    df = pd.read_csv(csv_file)
    
    # Convertir Speedup de string con coma decimal a float
    df['Speedup'] = df['Speedup'].str.replace(',', '.').astype(float)
    
    # Agrupar por (RowsA, ColsA)
    grouped = df.groupby(['RowsA', 'ColsA'])
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    
    for (rowsA, colsA), group in grouped:
        plt.plot(group['ColsB'], group['Speedup'], marker='o', label=f'RowsA={int(rowsA)}, ColsA={int(colsA)}')
    
    plt.xlabel('ColsB')
    plt.ylabel('Speedup (CPU/CGRA)')
    plt.title('Speedup por configuración de RowsA y ColsA')
    plt.legend()
    plt.grid()
    
    # Guardar la imagen
    plt.savefig(output_file, dpi=300)
    
    # Mostrar la imagen
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera una gráfica de speedup desde un archivo CSV.")
    parser.add_argument("csv_file", help="Ruta al archivo CSV de entrada")
    parser.add_argument("output_file", help="Ruta al archivo de imagen de salida")
    args = parser.parse_args()
    
    plot_speedup(args.csv_file, args.output_file)
