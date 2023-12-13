//
// Created by alireza on 10/6/23.
//

#include "dense_layerC.h"
#include <stdio.h>

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

void addbias(Dense* dense, size_t seq_len, int32_t* output) {
    for (size_t idx = 0; idx < seq_len; idx++) {
        for (size_t feature_idx = 0; feature_idx < dense->output_size_; feature_idx++) {
            output[idx * dense->output_size_ + feature_idx] += dense->bias[feature_idx];
        }
    }
}

void computeDense(Dense* dense, size_t seq_len, int32_t* input, int32_t* output) {
    multiply_cgra(output, input, dense->input_size_, seq_len, dense->weight, dense->output_size_, 0);
    if (dense->bias != NULL) {
        addbias(dense, seq_len, output);
    }
}

void activation(Dense* dense, size_t length, int32_t* input, int32_t* output) {
    float in_float, in_tanh;
    int32_t x3, in_tanh_fxp;
    for (int i = 0; i < length; i++) {
        x3 = MUL(MUL(input[i], input[i]), input[i]);
        x3 = MUL(x3, 183); // 183 = 0.044715 in fixed-point 12 bit
        x3 += input[i];
        x3 = MUL(x3, 3268); // 3268 = sqrt(2/PI) in fixed-point 12 bit
        in_float = (float) x3 / (float) (1 << NUM_FRACTION_BITS);
        in_tanh = tanhf(in_float);
        in_tanh_fxp = (int16_t) (in_tanh * (1 << NUM_FRACTION_BITS));
        in_tanh_fxp += (1 << NUM_FRACTION_BITS);
        output[i] = MUL(in_tanh_fxp, input[i] >> 1);
    }
}

