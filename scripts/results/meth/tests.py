import os
import time
import re

# Lista de dimensiones a probar
test_cases_rowsA_1 = [32]
test_cases_colsA_1 = [32,33,34]
test_cases_colsB_1 = [i for i in range(32,72)]

test_cases_rowsA_2 = [32]
test_cases_colsA_2 = [i for i in range(32,72)]
test_cases_colsB_2 = [32]

test_cases_rowsA_3 = [i for i in range(32,72)]
test_cases_colsA_3 = [32]
test_cases_colsB_3 = [32]

test_cases_dims = [i for i in range(32,72)]

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

def extract_times_from_log(log_file):
    """Extrae los tiempos Sw y CGRA del archivo uart0.log."""
    sw_time = None
    cgra_time = None
    
    with open(log_file, "r") as file:
        for line in file:
            if line.startswith("Sw:"):
                sw_time = int(line.split(":")[1].strip())
            elif line.startswith("CGRA:"):
                cgra_time = int(line.split(":")[1].strip())
    
    return sw_time, cgra_time

def main():
    
    i = 0
    # Ejecutar casos para test_cases_*_1
    for rows_a in test_cases_rowsA_1:
        for cols_a in test_cases_colsA_1:
            for cols_b in test_cases_colsB_1:
                # Modificar el archivo transformer.h una sola vez
                modify_file(TRANSFORMER_PATH, rows_a, cols_a, cols_b)

                # Compilar el proyecto una sola vez
                execute_command(f"make app PROJECT={PROJECT_NAME}")

                # Moverse a la carpeta de simulación
                os.chdir(BUILD_DIR)
                
                sw_times = []
                cgra_times = []
                
                for run in range(5):  # Ejecutar cada caso de prueba 5 veces
                    print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} - Intento {run+1}/5 ({i+1}/{len(test_cases_colsB_1)*len(test_cases_colsA_1)*len(test_cases_rowsA_1)*5})")
                    i += 1

                    # Ejecutar la simulación
                    execute_command("./Vtestharness +firmware=../../../sw/build/main.hex")

                    # Leer y guardar salida del log
                    if os.path.exists("uart0.log"):
                        sw_time, cgra_time = extract_times_from_log("uart0.log")
                        
                        # Almacenar los tiempos para calcular la media
                        if sw_time is not None and cgra_time is not None:
                            sw_times.append(sw_time)
                            cgra_times.append(cgra_time)

                    # Pequeña pausa entre pruebas
                    time.sleep(2)
                
                # Volver al directorio original
                os.chdir("../../..")

                # Calcular la media de los tiempos
                if sw_times and cgra_times:
                    sw_avg = sum(sw_times) / len(sw_times)
                    cgra_avg = sum(cgra_times) / len(cgra_times)

                # Guardar en el archivo de salida
                with open(os.path.join("../../..", OUTPUT_FILE), "a") as output_file:
                    output_file.write(f"\n--- Ejemplo {rows_a}x{cols_a}x{cols_b} ---\n")
                    output_file.write(f"Tiempo promedio Sw: {sw_avg}\n")
                    output_file.write(f"Tiempo promedio CGRA: {cgra_avg}\n")

    i=0
    # Ejecutar casos para test_cases_*_2
    for rows_a in test_cases_rowsA_2:
        for cols_a in test_cases_colsA_2:
            for cols_b in test_cases_colsB_2:
                # Modificar el archivo transformer.h una sola vez
                modify_file(TRANSFORMER_PATH, rows_a, cols_a, cols_b)

                # Compilar el proyecto una sola vez
                execute_command(f"make app PROJECT={PROJECT_NAME}")

                # Moverse a la carpeta de simulación
                os.chdir(BUILD_DIR)

                sw_times = []
                cgra_times = []
                
                for run in range(5):  # Ejecutar cada caso de prueba 5 veces
                    print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} - Intento {run+1}/5 ({i+1}/{len(test_cases_colsB_2)*len(test_cases_colsA_2)*len(test_cases_rowsA_2)*5})")
                    i += 1

                    # Ejecutar la simulación
                    execute_command("./Vtestharness +firmware=../../../sw/build/main.hex")

                    # Leer y guardar salida del log
                    if os.path.exists("uart0.log"):
                        sw_time, cgra_time = extract_times_from_log("uart0.log")
                        
                        # Almacenar los tiempos para calcular la media
                        if sw_time is not None and cgra_time is not None:
                            sw_times.append(sw_time)
                            cgra_times.append(cgra_time)

                    # Pequeña pausa entre pruebas
                    time.sleep(2)
                
                # Volver al directorio original
                os.chdir("../../..")

                # Calcular la media de los tiempos
                if sw_times and cgra_times:
                    sw_avg = sum(sw_times) / len(sw_times)
                    cgra_avg = sum(cgra_times) / len(cgra_times)

                # Guardar en el archivo de salida
                with open(os.path.join("../../..", OUTPUT_FILE), "a") as output_file:
                    output_file.write(f"\n--- Ejemplo {rows_a}x{cols_a}x{cols_b} ---\n")
                    output_file.write(f"Tiempo promedio Sw: {sw_avg}\n")
                    output_file.write(f"Tiempo promedio CGRA: {cgra_avg}\n")

    i=0
    # Ejecutar casos para test_cases_*_3
    for rows_a in test_cases_rowsA_3:
        for cols_a in test_cases_colsA_3:
            for cols_b in test_cases_colsB_3:
                # Modificar el archivo transformer.h una sola vez
                modify_file(TRANSFORMER_PATH, rows_a, cols_a, cols_b)

                # Compilar el proyecto una sola vez
                execute_command(f"make app PROJECT={PROJECT_NAME}")

                # Moverse a la carpeta de simulación
                os.chdir(BUILD_DIR)

                sw_times = []
                cgra_times = []
                
                for run in range(5):  # Ejecutar cada caso de prueba 5 veces
                    print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} - Intento {run+1}/5 ({i+1}/{len(test_cases_colsB_3)*len(test_cases_colsA_3)*len(test_cases_rowsA_3)*5})")
                    i += 1

                    # Ejecutar la simulación
                    execute_command("./Vtestharness +firmware=../../../sw/build/main.hex")

                    # Leer y guardar salida del log
                    if os.path.exists("uart0.log"):
                        sw_time, cgra_time = extract_times_from_log("uart0.log")
                        
                        # Almacenar los tiempos para calcular la media
                        if sw_time is not None and cgra_time is not None:
                            sw_times.append(sw_time)
                            cgra_times.append(cgra_time)

                    # Pequeña pausa entre pruebas
                    time.sleep(2)

                # Volver al directorio original
                os.chdir("../../..")

                # Calcular la media de los tiempos
                if sw_times and cgra_times:
                    sw_avg = sum(sw_times) / len(sw_times)
                    cgra_avg = sum(cgra_times) / len(cgra_times)

                # Guardar en el archivo de salida
                with open(os.path.join("../../..", OUTPUT_FILE), "a") as output_file:
                    output_file.write(f"\n--- Ejemplo {rows_a}x{cols_a}x{cols_b} ---\n")
                    output_file.write(f"Tiempo promedio Sw: {sw_avg}\n")
                    output_file.write(f"Tiempo promedio CGRA: {cgra_avg}\n")

    print("Todas las pruebas han finalizado.")

if __name__ == "__main__":
    main()
