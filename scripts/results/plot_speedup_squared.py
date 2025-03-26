import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_speedup(csv_file, output_file):
    # Leer el CSV
    df = pd.read_csv(csv_file)
    
    # Rellenar valores faltantes en RowsA, ColsA y ColsB
    df[['RowsA', 'ColsA', 'ColsB']] = df[['RowsA', 'ColsA', 'ColsB']].ffill()
    
    # Eliminar posibles espacios en blanco y convertir a enteros
    df['RowsA'] = df['RowsA'].astype(str).str.strip().astype(int)
    df['ColsA'] = df['ColsA'].astype(str).str.strip().astype(int)
    df['ColsB'] = df['ColsB'].astype(str).str.strip().astype(int)
    
    # Convertir Speedup de string con coma decimal a float
    df['Speedup'] = df['Speedup'].str.replace(',', '.').astype(float)
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    
    # Suponiendo que ya no necesitamos agrupar, solo graficar directamente
    plt.plot(df['ColsB'], df['Speedup'], marker='o', label='Speedup')
    
    plt.xlabel('RowsA=ColsA=ColsB')
    plt.ylabel('Speedup (CPU/CGRA)')
    plt.title('Speedup para matrices cuadradas')
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
