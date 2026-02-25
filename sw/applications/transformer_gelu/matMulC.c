//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>
#include <stdint.h>
#include "performance.h"


void MatMul_multiply_cpu(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size ) {
    printf("-----------------------\n");
    printf("MatMul (%dx%dx%d)\n", seq_len, input_size, output_size);
    printf("-----------------------\n");
    reset_csr_counters();
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < output_size; j++) {
            output[i * output_size + j] = 0;
            for (size_t k = 0; k < input_size; k++) {
                output[i * output_size + j] += input[i * input_size + k] * weight[k * output_size + j];
            }
        }
    }
    read_csr_counters();
}

void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size) {
    
    MatMul_multiply_cpu(seq_len, input, weight, output, input_size, output_size);
    //multiply_cgra(input, seq_len, input_size, weight, output_size, output);
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {
    printf("-----------------------\n");
    printf("MatMul scale (%d)\n", mat_size);
    printf("-----------------------\n");
    reset_csr_counters();
    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
    read_csr_counters();
}

