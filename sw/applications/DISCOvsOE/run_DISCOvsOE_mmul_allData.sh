#!/bin/bash

# --- Configuración de rutas ---
RUTA_BASE=$(pwd)
KERNEL_NAME="gemm"
RUTA_KERNEL="./sw/applications/DISCOvsOE/${KERNEL_NAME}"
RUTA_DATA="${RUTA_KERNEL}/datasets"
RUTA_UART="./build/eslepfl_systems_cgra-x-heep_0/sim-verilator/uart0.log"

# NUEVA RUTA: El destino correcto del dataset.h dentro de la app
RUTA_DESTINO_DATASET="${RUTA_KERNEL}/dataset.h"

# Archivo prioritario que queremos ejecutar primero
DATASET_PRIORITARIO="dataset_16x16x16.h"

# Crear la carpeta de logs
CARPETA_LOGS="${RUTA_BASE}/logs_resultados"
mkdir -p "${CARPETA_LOGS}"

echo "=== Iniciando batería de tests ==="
echo "Logs guardados en: ${CARPETA_LOGS}"
echo "----------------------------------"

# Verificar si la carpeta de datos existe
if [ ! -d "$RUTA_DATA" ]; then
    echo "Error: La carpeta de datos no existe en ${RUTA_DATA}"
    exit 1
fi

# --- Función auxiliar para procesar un dataset ---
procesar_dataset() {
    local fichero_completo="$1"
    local fichero_nombre=$(basename "$fichero_completo")
    local dimensiones=$(echo "$fichero_nombre" | sed -e 's/^dataset_//' -e 's/\.h$//')

    echo "Procesando dataset con dimensiones: ${dimensiones}"

    # 1. Copiar el contenido del dataset al fichero destino correcto
    cp "$fichero_completo" "$RUTA_DESTINO_DATASET"

    # 2. Ejecutar ./compileAndRun.sh guardando el log
    "${RUTA_BASE}/compileAndRun.sh" "DISCOvsOE/${KERNEL_NAME}" > "${CARPETA_LOGS}/scriptOut_${dimensiones}.log" 2>&1
    
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

# --- PASO 1: Forzar la ejecución del dataset prioritario (16x16x16) ---
FICHERO_16="${RUTA_DATA}/${DATASET_PRIORITARIO}"

if [ -f "$FICHERO_16" ]; then
    echo "[INFO] Forzando ejecución prioritaria de prueba rápida: ${DATASET_PRIORITARIO}"
    echo "----------------------------------"
    procesar_dataset "$FICHERO_16"
else
    echo "[AVISO] No se encontró el dataset prioritario (${DATASET_PRIORITARIO}) en ${RUTA_DATA}. Procediendo con el orden normal."
fi

# --- PASO 2: Iteración por el resto de ficheros de la carpeta data ---
for fichero_completo in "${RUTA_DATA}"/dataset_*.h; do
    
    # Validar si existen archivos
    [ -e "$fichero_completo" ] || { echo "No se encontraron ficheros dataset_*.h"; exit 1; }

    fichero_nombre=$(basename "$fichero_completo")

    # Si es el archivo que ya ejecutamos al principio, lo saltamos
    if [ "$fichero_nombre" = "$DATASET_PRIORITARIO" ]; then
        continue
    fi

    # Procesar el resto de datasets normalmente
    procesar_dataset "$fichero_completo"
done

echo "=== Batería de tests completada ==="