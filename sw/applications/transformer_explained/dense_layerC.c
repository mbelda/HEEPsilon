//
// Created by alireza on 10/6/23.
//

#include "dense_layerC.h"
#include <stdio.h>
#include "performance.h"
#include <stdint.h>

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

/**
 * @brief Multiplies the input with the weight matrix of the dense layer.
 * @param dense Pointer to the Dense layer structure.
 * @param seq_len Length of the sequence (number of rows on the input).
 * @param input Pointer to the input array. Type: Q12.
 * @param output Pointer to the output array where the result will be stored. Type: Q12.
 */
void multiplyweight(Dense* dense, size_t seq_len, int32_t* input, int32_t* output) {
    // Perfect for CGRA
    for (int length = 0; length < seq_len; length++) { // Rows input
        for (int out_idx = 0; out_idx < dense->output_size_; out_idx++) { // Cols output
            int32_t* weight_ptr = dense->weight + out_idx;
            int32_t* output_ptr = output + (length * dense->output_size_) + out_idx;
            int32_t* input_ptr = input + (length * dense->input_size_);
            int32_t sum = 0; // Sum int32_t to handle overflow
            for (int i = 0; i < dense->input_size_; i++) { // Cols input
                sum += MUL_HQ(*weight_ptr, *input_ptr); 
                input_ptr++;
                weight_ptr += dense->output_size_;
            }
            *(output_ptr) = (int32_t) (sum >> NUM_FRACTION_BITS); // int32_t to Q12
        }
    }
}

/**
 * @brief Adds bias to the output of the dense layer.
 * @param dense Pointer to the Dense layer structure.
 * @param seq_len Length of the sequence (number of input vectors).
 * @param output Pointer to the output array where the result will be stored. The same as input. Type: Q12.
 * Notes: It does not take care of overflow.
 */
void addbias(Dense* dense, size_t seq_len, int32_t* output) {
    // Perfect for CGRA
    for (size_t idx = 0; idx < seq_len; idx++) {
        for (size_t feature_idx = 0; feature_idx < dense->output_size_; feature_idx++) {
            output[idx * dense->output_size_ + feature_idx] += dense->bias[feature_idx];
        }
    }
}


/**
 * @brief Computes the output of the dense layer given the input.
 * @param dense Pointer to the Dense layer structure.
 * @param seq_len Length of the sequence (number of input vectors).
 * @param input Pointer to the input array. Type: Q12.
 * @param output Pointer to the output array where the result will be stored. Type: Q12.
 * Calls the multiplyweight and addbias functions.
 */
void computeDense(Dense* dense, size_t seq_len, int32_t* input, int32_t* output) {
    multiplyweight(dense, seq_len, input, output);
    if (dense->bias != NULL) {
        addbias(dense, seq_len, output);
    }
}

/**
 * @brief Applies the GELU activation function to the input array.
 * @param dense Pointer to the Dense layer structure.
 * @param length Length of the input array.
 * @param input Pointer to the input array. Type: Q12.
 * @param output Pointer to the output array where the result will be stored. Type: Q12.
 */
void activation(Dense* dense, size_t length, int32_t* input, int32_t* output) {
    float in_float, in_tanh;
    int32_t x3, in_tanh_fxp; // To handle overflow
    for (int i = 0; i < length; i++) {
        x3 = MUL(MUL(input[i], input[i]), input[i]);
        x3 = MUL(x3, 183); // 183 = 0.044715 in fixed-point 12 bit
        x3 += input[i];
        x3 = MUL(x3, 3268); // 3268 = sqrt(2/PI) in fixed-point 12 bit
        in_float = (float) x3 / (float) (1 << NUM_FRACTION_BITS); // Q12 to float
        in_tanh = tanhf(in_float); // tanh in float // Problem for CGRA
        in_tanh_fxp = (int32_t) (in_tanh * (1 << NUM_FRACTION_BITS)); // float to Q12
        in_tanh_fxp += (1 << NUM_FRACTION_BITS); // Add 1 in Q12
        output[i] = MUL(in_tanh_fxp, input[i] >> 1); // Multiply result by x/2
    }
}

