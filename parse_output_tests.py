import csv
import re
import argparse

def parse_txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("--- Ejemplo"):
            # Extraer dimensiones
            dimensions = re.search(r"(\d+)x(\d+)x(\d+)", line)
            if dimensions:
                rows_a, cols_a, cols_b = map(int, dimensions.groups())
            
            # Saltar la línea siguiente
            i += 1

            # Leer datos de Sw y CGRA
            sw_time = cgra_time = None
            while i < len(lines) and lines[i].strip():
                match_sw = re.search(r"Sw:\s*(\d+)", lines[i])
                match_cgra = re.search(r"Cgra:\s*(\d+)", lines[i])
                
                if match_sw:
                    sw_time = int(match_sw.group(1))
                if match_cgra:
                    cgra_time = int(match_cgra.group(1))
                
                i += 1

            if sw_time is not None and cgra_time is not None:
                speedup = sw_time / cgra_time
                data.append([rows_a, cols_a, cols_b, cgra_time, sw_time, speedup])
        
        i += 1

    # Escribir en CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["RowsA", "ColsA", "ColsB", "CGRA", "CPU", "Speedup"])
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description="Parsea un fichero de texto y genera un CSV con los resultados.")
    parser.add_argument("input_file", help="Ruta del fichero de entrada (.txt)")
    parser.add_argument("output_file", help="Ruta del fichero de salida (.csv)")
    
    args = parser.parse_args()
    
    parse_txt_to_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
