import os
import time
import re

# Lista de dimensiones a probar
graph_cases = [
    (4,4,4), (16,4,4), (4,16,4), (4,4,16), (16,16,16), (32,32,32), (33,32,32), (34,32,32), (35,32,32), (32,32,33), (32,32,34), (32,32,35), (35,35,35)
]

graph_big_cases = [
    (64,64,64), (65,65,65), (66,66,66), (67,67,67), (68,68,68)
]

test_cases_rowsA_1 = [32]
test_cases_colsA_1 = [32,33,34]
test_cases_colsB_1 = [range(32,72)]

test_cases_rowsA_2 = [32]
test_cases_colsA_2 = [range(32,72)]
test_cases_colsB_2 = [32]

test_cases_rowsA_3 = [range(32,72)]
test_cases_colsA_3 = [32]
test_cases_colsB_3 = [32]

test_cases_dims = [range(32,72)]

# Rutas de archivos y directorios
PROJECT_NAME = "mmul_os_opt2"
TRANSFORMER_PATH = f"./sw/applications/{PROJECT_NAME}/transformer.h"
BUILD_DIR = "./build/eslepfl_systems_cgra-x-heep_0/sim-verilator"
OUTPUT_FILE = "output_tests_2_mmul_opt2.txt"


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
    for rows_a in test_cases_rowsA_2:
        for cols_a in test_cases_colsA_2:
            for cols_b in test_cases_colsB_2:
    #for i, (rows_a, cols_a, cols_b) in enumerate(graph_big_cases, 1):
                print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} ({i}/{len(test_cases_colsB_2)*len(test_cases_colsA_2)*len(test_cases_rowsA_2)})")
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
