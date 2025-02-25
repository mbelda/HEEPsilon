import os
import time

# Lista de dimensiones a probar
test_cases = [
    (16,32,16), (16,16,32), (35,32,32)
]

# Rutas de archivos y directorios
TRANSFORMER_PATH = "./sw/applications/mmul_os/transformer.h"
BUILD_DIR = "./build/eslepfl_systems_cgra-x-heep_0/sim-verilator"
OUTPUT_FILE = "tests_output.txt"

def modify_transformer_file(rows_a, cols_a, cols_b):
    """Modifica el archivo transformer.h con las nuevas dimensiones."""
    with open(TRANSFORMER_PATH, "w") as f:
        f.write(f"#define ROWS_A {rows_a}\n")
        f.write(f"#define COLS_A {cols_a}\n")
        f.write(f"#define COLS_B {cols_b}\n")

def execute_command(command):
    """Ejecuta un comando en la terminal sin mostrar la salida."""
    os.system(f"{command} > /dev/null 2>&1")

def main():
    for i, (rows_a, cols_a, cols_b) in enumerate(test_cases, 1):
        print(f"Ejecutando ejemplo {rows_a}x{cols_a}x{cols_b} ({i}/{len(test_cases)})")

        # Modificar el archivo transformer.h
        modify_transformer_file(rows_a, cols_a, cols_b)

        # Compilar el proyecto
        execute_command("make app PROJECT=mmul_os")

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

        # Volver al directorio original
        os.chdir("../../..")

        # Pequeña pausa entre pruebas
        time.sleep(2)

    print("Todas las pruebas han finalizado.")

if __name__ == "__main__":
    main()
