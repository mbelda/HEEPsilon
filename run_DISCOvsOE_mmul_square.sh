#!/bin/bash

# --- Configuración de rutas ---
RUTA_BASE=$(pwd)
RUTA_MMUL="./sw/applications/DISCOvsOE/mmul"
RUTA_DATA="${RUTA_MMUL}/data"
RUTA_UART="./build/eslepfl_systems_cgra-x-heep_0/sim-verilator/uart0.log"
RUTA_DESTINO_DATASET="${RUTA_MMUL}/dataset.h"

# Crear la carpeta de logs (mantiene la misma para unificar resultados)
CARPETA_LOGS="${RUTA_BASE}/logs_resultados"
mkdir -p "${CARPETA_LOGS}"

# --- LISTA DE DATASETS PENDIENTES ---
# Definimos exactamente qué tamaños queremos procesar ahora
DATASETS_PENDIENTES=("32x32x32" "64x64x64" "128x128x128")

echo "=== Reanudando batería de tests ==="
echo "Ejecutando tamaños específicos: ${DATASETS_PENDIENTES[*]}"
echo "Logs guardados en: ${CARPETA_LOGS}"
echo "----------------------------------"

# Verificar si la carpeta de datos existe
if [ ! -d "$RUTA_DATA" ]; then
    echo "Error: La carpeta de datos no existe en ${RUTA_DATA}"
    exit 1
fi

# --- Función auxiliar para procesar un dataset ---
procesar_dataset() {
    local dimensiones="$1"
    local fichero_completo="${RUTA_DATA}/dataset_${dimensiones}.h"

    # Verificar si el archivo específico existe antes de intentar copiarlo
    if [ ! -f "$fichero_completo" ]; then
        echo "[ERROR] No se encontró el archivo: dataset_${dimensiones}.h en ${RUTA_DATA}"
        echo "----------------------------------"
        return 1
    fi

    echo "Procesando dataset con dimensiones: ${dimensiones}"

    # 1. Copiar el contenido del dataset al fichero destino correcto
    cp "$fichero_completo" "$RUTA_DESTINO_DATASET"

    # 2. Ejecutar ./compileAndRun.sh guardando el log
    "${RUTA_BASE}/compileAndRun.sh" "DISCOvsOE/mmul" > "${CARPETA_LOGS}/scriptOut_${dimensiones}.log" 2>&1
    
    # 3. Comprobar y copiar el fichero uart0.log
    if [ -f "$RUTA_UART" ]; then
        cp "$RUTA_UART" "${CARPETA_LOGS}/uart_${dimensiones}.log"
        
        # 4. Leer el contenido buscando la palabra "OK"
        if grep -q "OK" "$RUTA_UART"; then
            echo "  [RESULTADO] Test (${dimensiones}): finalizado OK"
        else
            echo "  [ERROR] Test (${dimensiones}): Falló (No se encontró 'OK' en uart0.log)"
        fi
    else
        echo "  [ERROR] Test (${dimensiones}): No se generó el fichero uart0.log."
    fi
    echo "----------------------------------"
}

# --- Bucle sobre la lista de pendientes ---
for dim in "${DATASETS_PENDIENTES[@]}"; do
    procesar_dataset "$dim"
done

echo "=== Ejecución de pendientes completada ==="
