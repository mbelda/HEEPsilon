//
// Created by alireza on 10/6/23.
//

#include "dense_layerC.h"
#include <stdio.h>
#include "performance.h"
#include <stdint.h>

#include "gelu_aprox.h"

void createDense(Dense* dense, size_t input_dim, size_t output_dim, quant_bit_width *weight, quant_bit_width* bias) {
    dense->input_size_ = input_dim;
    dense->output_size_ = output_dim;
    dense->weight = weight;
    dense->bias = bias;
}

void destroyDense(Dense* dense) {
    // Free the memory allocated for the Dense struct
    free(dense);
}

void multiplyweight(Dense* dense, size_t seq_len, int32_t* input, int32_t* output) {
    printf("Multiply\n");
    reset_csr_counters();
    for (int length = 0; length < seq_len; length++) {
        for (int out_idx = 0; out_idx < dense->output_size_; out_idx++) {
            int32_t* weight_ptr = dense->weight + out_idx;
            int32_t* output_ptr = output + (length * dense->output_size_) + out_idx;
            int32_t* input_ptr = input + (length * dense->input_size_);
            int32_t sum = 0;
            for (int i = 0; i < dense->input_size_; i++) {
                sum += MUL_HQ(*weight_ptr, *input_ptr); // MUL_HQ macro
                input_ptr++;
                weight_ptr += dense->output_size_;
            }
            *(output_ptr) = (int32_t) (sum >> NUM_FRACTION_BITS); // NUM_FRACTION_BITS macro
        }
    }
    read_csr_counters();
}

void addbias(Dense* dense, size_t seq_len, int32_t* output) {
    printf("Add bias\n");
    reset_csr_counters();
    for (size_t idx = 0; idx < seq_len; idx++) {
        for (size_t feature_idx = 0; feature_idx < dense->output_size_; feature_idx++) {
            output[idx * dense->output_size_ + feature_idx] += dense->bias[feature_idx];
        }
    }
    read_csr_counters();
}

void computeDense(Dense* dense, size_t seq_len, int32_t* input, int32_t* output) {
    

    //uint64_t begin, end;
    //begin = getTime_cy();
    //kcom_perfRecordStart(&(kperf->time.mul));
    multiplyweight(dense, seq_len, input, output);
    //multiply_cgra(input, seq_len, dense->input_size_, dense->weight, dense->output_size_, output);
    //kcom_perfRecordStop(&(kperf->time.mul));
    //end = getTime_cy();
    //uint64_t total = end - begin;
    /*printf("\rMul %dx%dx%d: %llu\n", seq_len, dense->input_size_, dense->output_size_, (unsigned long long)(end - begin));
    printf("begin: 0x%08lx%08lx, end: 0x%08lx%08lx, total: 0x%08lx%08lx\n", 
        (unsigned long)(begin >> 32), (unsigned long)(begin & 0xFFFFFFFF), 
        (unsigned long)(end >> 32), (unsigned long)(end & 0xFFFFFFFF),
        (unsigned long)(total >> 32), (unsigned long)(total & 0xFFFFFFFF));
    */
    if (dense->bias != NULL) {
        addbias(dense, seq_len, output);
    }
}

void activation(Dense* dense, size_t length, int32_t* input, int32_t* output) {

    gelu(dense, length, input, output);
    
}

