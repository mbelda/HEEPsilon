//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>


void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size ) {

    for (size_t i = 0; i < seq_len; i++) {
        // Iterar sobre las columnas de la matriz B
        for (size_t j = 0; j < output_size; j++) {
            // Inicializar el valor de salida en cero
            output[i * output_size + j] = 0;

            // Calcular el producto punto de la fila i de A y la columna j de B
            for (size_t k = 0; k < input_size; k++) {
                // Multiplicar los elementos correspondientes y agregar al resultado
                output[i * output_size + j] += input[i * input_size + k] * weight[k * output_size + j];
            }
        }
    }
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {

    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
}

