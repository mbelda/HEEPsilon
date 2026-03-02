#!/bin/bash

# Parámetros fijos
COLS_A=3
COLS_B=12

# Guardamos el directorio inicial
ROOT_DIR=$(pwd)

# Ruta al generador de datos
DATA_DIR="sw/applications/mmul_ws_il_rv_summit/data"

# Ruta destino de resultados
RESULTS_DIR="scripts/results/rv-summit/mmul_ws/rows_a"

# Crear carpeta de resultados si no existe
mkdir -p "$RESULTS_DIR"

# Loop para ROWS_A múltiplos de 4 desde 4 hasta 120
for ROWS_A in $(seq 4 4 120)
do
    echo "========================================"
    echo "Running experiment for ${ROWS_A}x${COLS_A}x${COLS_B}"
    echo "========================================"

    # 1️⃣ Generar datos
    cd "$DATA_DIR" || exit 1

    DATASET_NAME="dataset_${ROWS_A}x${COLS_A}x${COLS_B}.h"

    python3 gen_data.py $ROWS_A $COLS_A $COLS_B > "$DATASET_NAME"

    cp "$DATASET_NAME" ../dataset.h

    # 2️⃣ Compilar y ejecutar
    cd "$ROOT_DIR" || exit 1

    ./compileAndRun.sh mmul_ws_il_rv_summit > compile.log 2>&1

    # 3️⃣ Guardar salida
    cp build/eslepfl_systems_cgra-x-heep_0/sim-verilator/uart0.log \
       "$RESULTS_DIR/${ROWS_A}x${COLS_A}x${COLS_B}.log"

    echo "Finished ${ROWS_A}x${COLS_A}x${COLS_B}"
    echo ""
done

echo "Todos los experimentos han terminado."
