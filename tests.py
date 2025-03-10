import os
import time
import re

# Lista de dimensiones a probar
graph_cases = [
    (4,4,4), (16,4,4), (4,16,4), (4,4,16), (16,16,16), (32,32,32), (33,32,32), (34,32,32), (35,32,32), (32,32,33), (32,32,34), (32,32,35), (35,35,35)
]

#test_cases_rowsA = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
#test_cases_colsA = [9]
#test_cases_colsB = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

# Rutas de archivos y directorios
PROJECT_NAME = "mmul_os_opt2"
TRANSFORMER_PATH = f"./sw/applications/{PROJECT_NAME}/transformer.h"
BUILD_DIR = "./build/eslepfl_systems_cgra-x-heep_0/sim-verilator"
OUTPUT_FILE = "output_tests_mmul_opt.txt"


def modify_file(filename, rowsA, colsA, colsB):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Expresiones regulares para buscar y reemplazar
    patterns = {
        r"#define ROWS_A \d+": f"#define ROWS_A {rowsA}",
        r"#define COLS_A \d+": f"#define COLS_A {colsA}",
        r"#define COLS_B \d+": f"#define COLS_B {colsB}"
    }
    
    new_lines = []
    for line in lines:
        for pattern, replacement in patterns.items():
            if re.match(pattern, line):
                line = replacement + '\n'
                break  # Evita múltiples reemplazos en la misma línea
        new_lines.append(line)
    
    with open(filename, 'w') as file:
        file.writelines(new_lines)

def execute_command(command):
    """Ejecuta un comando en la terminal sin mostrar la salida."""
    os.system(f"{command} > /dev/null 2>&1")

def main():
    i = 0
    #for rows_a in test_cases_rowsA:
    #    for cols_a in test_cases_colsA:
    #        for cols_b in test_cases_colsB:
    for i, (rows_a, cols_a, cols_b) in enumerate(graph_cases, 1):
        print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} ({i}/{len(graph_cases)})")
        i+=1

        # Modificar el archivo transformer.h
        modify_file(TRANSFORMER_PATH, rows_a, cols_a, cols_b)

        # Compilar el proyecto
        execute_command(f"make app PROJECT={PROJECT_NAME}")

        # Moverse a la carpeta de simulación y ejecutar la simulación
        os.chdir(BUILD_DIR)
        execute_command("./Vtestharness +firmware=../../../sw/build/main.hex")

        # Leer y guardar salida del log
        if os.path.exists("uart0.log"):
            with open("uart0.log", "r") as log_file:
                content = log_file.read()
            
            # Abrir en modo append para agregar contenido al final
            with open(os.path.join("../../..", OUTPUT_FILE), "a") as output_file:
                output_file.write(f"\n--- Ejemplo {rows_a}x{cols_a}x{cols_b} ---\n")
                output_file.write(content)
        
        #execute_command(f"cp uart0.log ../../../test_logs/uart0_{rows_a}x{cols_a}x{cols_b}.log")

        # Volver al directorio original
        os.chdir("../../..")

        # Pequeña pausa entre pruebas
        time.sleep(2)

    print("Todas las pruebas han finalizado.")

if __name__ == "__main__":
    main()
