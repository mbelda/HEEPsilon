//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>

#define PRINTS_DEBUG_MATMUL 1

void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size ) {
    uint64_t begin, end;
    if (PRINTS_DEBUG_MATMUL){
        begin = getTime_cy();
    }                       
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

    if (PRINTS_DEBUG_MATMUL){
        end = getTime_cy();
        uint64_t total = end - begin;
        printf("Matmul %dx%dx%d: begin: 0x%08lx%08lx, end: 0x%08lx%08lx, total: 0x%08lx%08lx\n", 
            seq_len, input_size, output_size,
            (unsigned long)(begin >> 32), (unsigned long)(begin & 0xFFFFFFFF), 
            (unsigned long)(end >> 32), (unsigned long)(end & 0xFFFFFFFF),
            (unsigned long)(total >> 32), (unsigned long)(total & 0xFFFFFFFF));
    }
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {

    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
}

