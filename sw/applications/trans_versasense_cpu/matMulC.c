//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>

void multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size){
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < output_size; j++) {
            output[i*output_size + j] = 0; // Inicializar cada elemento de la matriz de resultado en 0
            for (int k = 0; k < input_size; k++) {
                output[i*output_size + j] += input[i*input_size + k] * weight[k*output_size + j];
            }
        }
    }
}


void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size) {
    //printf("\rMul %dx%dx%d\n", seq_len, input_size, output_size);
    //multiply_cgra(input, seq_len, input_size, weight, output_size, output);
    multiply(input, seq_len, input_size, weight, output_size, output);
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {

    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
}

