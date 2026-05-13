//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>


void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size ) {
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < output_size; j++) {
            double *out_ptr = output + (i * output_size + j);
            *out_ptr = 0;
            for (size_t k = 0; k < input_size; k++) {
                double *in_ptr = input + (i * input_size + k);
                double *weight_ptr = weight + (k * output_size + j);
                *out_ptr += (*in_ptr) * (*weight_ptr);
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

