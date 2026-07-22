
#include <stdint.h>
/*
 * cgra_kernel_execution -- SIMULACIÓN DEL HARDWARE DEL CGRA
 *
 * Esta función ejecuta exactamente la tarea que realiza el CGRA cuando 
 * la CPU llama a cgra_start() y espera en cgra_wait_done().
 *
 * Entradas: Configuración cargada previamente en los registros del CGRA.
 * Salida:   Suma acumulada del producto punto (MAC sum).
 */
int32_t cgra_kernel_execution(
    const int32_t *input_ptr,   // Puntero base al primer canal de entrada
    const int32_t *weights_ptr, // Puntero base a los pesos del filtro (OC actual)
    int in_per_group,           // Número de canales de entrada por grupo (IC)
    int kernel_w,               // Ancho del filtro (KW)
    int ow,                     // Índice de salida actual
    int stride_w,               // Stride horizontal
    int pad_w,                  // Padding horizontal
    int iw_total                // Ancho total de la imagen de entrada (IW)
) {
    int32_t mac_accumulator = 0;
    int icg, kw;

    // Punteros locales de trabajo dentro del CGRA
    const int32_t *curr_weight = weights_ptr;

    // -----------------------------------------------------------------
    // BUCLE INTERNO 1: Recorrer Canales de Entrada (icg)
    // -----------------------------------------------------------------
    for (icg = 0; icg < in_per_group; ++icg) {

        // Puntero al inicio del canal de entrada actual
        // En memoria, cada canal de entrada se encuentra desplazado por 'iw_total' elementos
        const int32_t *curr_input_channel = input_ptr + (icg * iw_total);

        // -------------------------------------------------------------
        // BUCLE INTERNO 2: Recorrer Ventana Espacial del Filtro (kw)
        // -------------------------------------------------------------
        for (kw = 0; kw < kernel_w; ++kw) {

            // 1. Calcular la posición espacial real en el vector de entrada (iw)
            int iw = ow * stride_w + kw - pad_w;

            // 2. Control de Límites (Zero-Padding en Hardware)
            // Si iw está fuera de límites (< 0 o >= IW), equivale a multiplicar por 0,
            // por lo que el acumulador no cambia.
            if (iw < 0 || iw >= iw_total) continue;
                
            // Lectura de Entrada y Peso
            int32_t in_val  = curr_input_channel[iw];
            int32_t w_val   = *curr_weight;

            // *** OPERACIÓN MAC PRINCIPAL ***
            mac_accumulator += in_val * w_val;
            
            // 3. Salto en Memoria de Pesos:
            // Los pesos están contiguos, el puntero siempre avanza +1 elemento
            curr_weight++;
        }
    }

    // Devuelve el resultado del producto punto acumulado
    return mac_accumulator;
}





/*
 * cgra_kernel_execution -- SIMULACIÓN DEL HARDWARE DEL CGRA (Con bucle OW)
 *
 * Entradas: Configuración cargada en registros.
 * Tarea:    Calcula la LÍNEA COMPLETA de salida (out_w elementos) 
 *           para UN canal de salida (OC) concreto.
 */
void cgra_kernel_execution_ow(
    const int32_t *input_ptr,   // Puntero base al primer canal de entrada
    const int32_t *weights_ptr, // Puntero base a los pesos del filtro (OC actual)
    int32_t *output_channel_ptr,// Puntero al inicio del buffer de salida para este OC
    int in_per_group,           // Número de canales de entrada (IC)
    int kernel_w,               // Ancho del filtro (KW)
    int out_w,                  // Ancho total de la salida (OW)
    int stride_w,               // Stride horizontal
    int pad_w,                  // Padding horizontal
    int iw_total                // Ancho total de la imagen de entrada (IW)
) {
    int ow, icg, kw;

    // -----------------------------------------------------------------
    // NUEVO BUCLE EN CGRA: Recorrer las posiciones de salida (ow)
    // -----------------------------------------------------------------
    for (ow = 0; ow < out_w; ++ow) {
        int32_t mac_accumulator = 0;

        // ¡OJO! Al reiniciar cada 'ow', los pesos se vuelven a leer desde el inicio
        const int32_t *curr_weight = weights_ptr;

        // -------------------------------------------------------------
        // BUCLE INTERNO 1: Recorrer Canales de Entrada (icg)
        // -------------------------------------------------------------
        for (icg = 0; icg < in_per_group; ++icg) {

            // Puntero al canal de entrada actual
            const int32_t *curr_input_channel = input_ptr + (icg * iw_total);

            // ---------------------------------------------------------
            // BUCLE INTERNO 2: Recorrer Kernel (kw)
            // ---------------------------------------------------------
            for (kw = 0; kw < kernel_w; ++kw) {

                int iw = ow * stride_w + kw - pad_w;

                if (iw >= 0 && iw < iw_total) {
                    int32_t in_val = curr_input_channel[iw];
                    int32_t w_val  = *curr_weight;

                    mac_accumulator += in_val * w_val;
                }

                // Avanzar al siguiente peso
                curr_weight++;
            }
        }

        // Guardamos el resultado del punto espacial 'ow' en la memoria de salida
        output_channel_ptr[ow] = mac_accumulator;
    }
}