import os
import subprocess
import shutil

sizes = [8, 16, 32, 64, 128, 256]

initial_dir = os.getcwd()
relu_cpu_dir = os.path.join(initial_dir, "sw/applications/relu_cpu")
build_dir = os.path.join(initial_dir, "build/eslepfl_systems_cgra-x-heep_0/sim-verilator")
uart_log = os.path.join(build_dir, "uart0.log")
output_log = os.path.join(initial_dir, "results_relu_cpu.log")

dataset_script = "gen_data.py"
dataset_file = os.path.join(relu_cpu_dir, "dataset.h")
datasets_dir = os.path.join(relu_cpu_dir, "datasets")

os.makedirs(datasets_dir, exist_ok=True)

with open(output_log, "w") as outfile:
    for size in sizes:
        print(f"\n=== Procesando tamaño {size}x{size} ===")
        os.chdir(relu_cpu_dir)

        # Generar dataset.h
        with open(dataset_file, 'w') as f:
            subprocess.run(
                ["python3", dataset_script, str(size)],
                check=True, stdout=f, stderr=subprocess.DEVNULL
            )

        dataset_copy = os.path.join(datasets_dir, f"dataset_{size}.h")
        shutil.copyfile(dataset_file, dataset_copy)
        print(f"Dataset para tamaño {size} guardado en: {dataset_copy}")

        os.chdir(initial_dir)

        # Compilar
        make_command = [
            "make", "app",
            "PROJECT=relu_cpu",
            "COMPILER_FLAGS=-O3",
            "COMPILER_PREFIX=riscv32-corev-",
            "ARCH=rv32imc_zicsr_zifencei_xcvhwlp0p1_xcvmem0p1_xcvmac0p1_xcvbi0p1_xcvalu0p1_xcvsimd0p1_xcvbitmanip0p1"
        ]
        subprocess.run(
            make_command,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Ejecutar simulador
        os.chdir(build_dir)
        subprocess.run(
            ["./Vtestharness", "+firmware=../../../sw/build/main.hex"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Guardar contenido uart0.log
        outfile.write(f"\n===== Resultado para tamaño {size}x{size} =====\n")
        if os.path.exists(uart_log):
            with open(uart_log, "r") as uart_file:
                outfile.write(uart_file.read())
        else:
            outfile.write("uart0.log no encontrado\n")

        os.chdir(initial_dir)

print(f"\nTodos los tamaños procesados. Resultados en {output_log}")
