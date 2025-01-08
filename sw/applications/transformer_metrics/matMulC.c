//
// Created by alireza on 10/6/23.
//

#include "matMulC.h"
#include <stdio.h>
#include <stdint.h>


void MatMul_multiply(size_t seq_len, quant_bit_width* input, quant_bit_width* weight,
                           quant_bit_width* output, size_t input_size, size_t output_size) {
    
    //uint64_t begin, end;
    //begin = getTime_cy();
    multiply_cgra(input, seq_len, input_size, weight, output_size, output);
    /*end = getTime_cy();

    if (end < begin) {
        end += (1ULL << 64); // Adjust for 64-bit overflow
    }
    uint64_t total = end - begin;
    printf("\rMatmul %dx%dx%d: %llu\n", seq_len, input_size, output_size,  (unsigned long long)(end - begin));
    printf("begin: 0x%08lx%08lx, end: 0x%08lx%08lx, total: 0x%08lx%08lx\n", 
    (unsigned long)(begin >> 32), (unsigned long)(begin & 0xFFFFFFFF), 
    (unsigned long)(end >> 32), (unsigned long)(end & 0xFFFFFFFF),
    (unsigned long)(total >> 32), (unsigned long)(total & 0xFFFFFFFF));*/
}

void MatMul_scale(quant_bit_width* input, int shift_scale, size_t mat_size) {

    for (size_t i = 0; i < mat_size; i++) {
        *input = (*input) >> shift_scale;
        input++;
    }
}

