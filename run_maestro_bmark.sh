#!/bin/bash

# 1. Validar que se ha pasado el nombre del benchmark como parámetro
if [ -z "$1" ]; then
    echo "Error: Debes proporcionar el nombre del benchmark."
    echo "Uso: $0 <NOMBRE_BENCHMARK>"
    exit 1
fi

BENCHMARK=$1

# --- RUTAS ---
BENCHMARK_DIR="sw/applications/MAESTRO/$BENCHMARK"
DATA_DIR="$BENCHMARK_DIR/data"
LOGS_DIR="$BENCHMARK_DIR/logs"
LOG_ORIGEN="build/eslepfl_systems_cgra-x-heep_0/sim-verilator/uart0.log"

# Argumento para compileAndRun.sh
ARG_COMPILE_RUN="MAESTRO/$BENCHMARK"

# 2. Validar que la carpeta de datos existe
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: La carpeta de datos '$DATA_DIR' no existe."
    exit 1
fi

# 3. Crear la carpeta de logs si no existe
mkdir -p "$LOGS_DIR"

# 4. Iterar sobre todos los ficheros data_*.h
shopt -s nullglob
FICHEROS_DATA=("$DATA_DIR"/data_*.h)

if [ ${#FICHEROS_DATA[@]} -eq 0 ]; then
    echo "No se encontraron ficheros 'data_*.h' en $DATA_DIR"
    exit 1
fi

for filepath in "${FICHEROS_DATA[@]}"; do
    filename=$(basename "$filepath")
    
    if [[ $filename =~ data_(.*)\.h ]]; then
        N="${BASH_REMATCH[1]}"
    else
        continue
    fi

    echo "--------------------------------------------------"
    echo "Procesando tamaño N = $N"
    echo "--------------------------------------------------"

    # A. Copiar el archivo actual a dataset.h
    echo "Copiando $filename a $BENCHMARK_DIR/dataset.h..."
    cp "$filepath" "$BENCHMARK_DIR/dataset.h"

    # B. Ejecutar el script ocultando la salida y guardándola en log_N.txt
    CONSOLE_LOG="$LOGS_DIR/log_$N.txt"
    echo "Ejecutando ./compileAndRun.sh (Salida redirigida a $CONSOLE_LOG)..."
    
    # El operador > redirige la salida estándar y 2>&1 redirige los errores al mismo sitio
    ./compileAndRun.sh "$ARG_COMPILE_RUN" > "$CONSOLE_LOG" 2>&1

    # C. Guardar el log de la UART resultante
    if [ -f "$LOG_ORIGEN" ]; then
        LOG_DESTINO="$LOGS_DIR/uart0_$N.log"
        echo "Guardando log de la UART en $LOG_DESTINO..."
        cp "$LOG_ORIGEN" "$LOG_DESTINO"
    else
        echo "¡Advertencia! No se encontró el archivo de log uart0.log en la ruta esperada."
    fi

done

echo "--------------------------------------------------"
echo "¡Proceso completado! Revisa la carpeta '$LOGS_DIR' para ver los resultados."