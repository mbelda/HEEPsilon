//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>


void MatMul_multiply_blocks(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size ) {
    
    for (size_t i  = 0; i < seq_len; i+=BLOCK_SIZE){
        size_t block_size_i = i + BLOCK_SIZE <= seq_len ? BLOCK_SIZE : seq_len - i;
        for (size_t j = 0; j < output_size; j+=BLOCK_SIZE){
            size_t block_size_j = j + BLOCK_SIZE <= output_size ? BLOCK_SIZE : output_size - j;
            // Multiply the block --> CGRA
            for (size_t ii = i; (ii < i + block_size_i) && (ii < seq_len); ii++){
                for (size_t jj = j; (jj < j + block_size_j) && (jj < output_size); jj++){
                    int32_t sum = 0;
                    for(size_t k = 0; k < input_size; k++){
                        sum += input[ii*input_size + k] * weight[k*output_size + jj];
                    }
                    output[ii*output_size + jj] = (quant_bit_width)(sum >> NUM_FRACTION_BITS);
                }
            }
        }
    }
}

void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size, cgra_t * cgra, uint8_t cgra_slot ) {
    printf("\rMul %dx%dx%d\n", seq_len, input_size, output_size);
    multiply_cgra(output, input, seq_len, input_size, weight, output_size, cgra, cgra_slot);
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {

    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
}

