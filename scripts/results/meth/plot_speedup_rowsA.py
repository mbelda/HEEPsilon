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
    
    # Calcular media y desviación estándar del Speedup para cada ColsB y RowsA
    grouped = df.groupby(['ColsB', 'RowsA']).agg({'Speedup': ['mean', 'std']}).reset_index()
    grouped.columns = ['ColsB', 'RowsA', 'Speedup_Mean', 'Speedup_Std']
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    
    for colsB, group in grouped.groupby('ColsB'):
        plt.plot(group['RowsA'], group['Speedup_Mean'], marker='o', label=f'ColsB={colsB}')
        plt.fill_between(group['RowsA'], group['Speedup_Mean'] - group['Speedup_Std'], 
                         group['Speedup_Mean'] + group['Speedup_Std'], alpha=0.2)
    
    plt.xlabel('RowsA')
    plt.ylabel('Speedup (CPU/CGRA)')
    plt.title('Speedup por (ColsA,ColsB) con Media y Desviación Estándar')
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