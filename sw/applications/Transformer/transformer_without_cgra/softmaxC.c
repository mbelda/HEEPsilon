//
// Created by alireza on 10/6/23.
//

#include "softmaxC.h"

#define PRINTS_DEBUG_MATMUL 1

void computeSoftmax(int32_t* input, size_t seq_len) {
    uint64_t begin, end;
    if (PRINTS_DEBUG_MATMUL){
        begin = getTime_cy();
    }  
    size_t width = seq_len;
    float input_float = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        // Look for the biggest value of the row
        int32_t max_val = input[i * seq_len];
        for (int j = 1; j < width; j++) { // Assuming its squared (width = seq_len)
            if (input[i * seq_len + j] > max_val) {
                max_val = input[i * seq_len + j];
            }
        }
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = (int32_t) fmax(input[i * seq_len + j] - max_val, -32767);
        }
        // Sum all values on the row
        int32_t sum = 0;
        for (int j = 0; j < width; j++) {
            input_float = (float) input[i * seq_len + j] / (float) (1 << NUM_FRACTION_BITS);
            input_float = expf(input_float);
            input[i * seq_len + j] = (int32_t) (input_float * (1 << NUM_FRACTION_BITS));
            sum += input[i * seq_len + j];
        }
        float sum_float = (float) sum / (float) (1 << NUM_FRACTION_BITS);
        float sum_inv = (float) (1 / (sum_float + 0.00001)); // prevent zero divide!
        int32_t sum_inv_int = (int32_t) (sum_inv * (1 << NUM_FRACTION_BITS));
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = (int32_t) MUL(input[i * seq_len + j], sum_inv_int);
        }
    }

    if (PRINTS_DEBUG_MATMUL){
        end = getTime_cy();
        uint64_t total = end - begin;
        printf("S: begin: 0x%08lx%08lx, end: 0x%08lx%08lx, total: 0x%08lx%08lx\n", 
            (unsigned long)(begin >> 32), (unsigned long)(begin & 0xFFFFFFFF), 
            (unsigned long)(end >> 32), (unsigned long)(end & 0xFFFFFFFF),
            (unsigned long)(total >> 32), (unsigned long)(total & 0xFFFFFFFF));
    }
}

