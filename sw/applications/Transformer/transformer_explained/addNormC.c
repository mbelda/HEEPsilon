//
// Created by alireza on 10/6/23.
//

#include "addNormC.h"



// Function implementations
AddNormalize createAddNormalize(int seq_len, int input_dim, quant_bit_width *weight, quant_bit_width *bias) {
    AddNormalize addNorm;
    addNorm.seq_len_ = seq_len;
    addNorm.input_dim_ = input_dim;
    addNorm.weight_ = weight;
    addNorm.bias_ = bias;
    // Initialize other fields as needed
    return addNorm;
}


/**
 * @brief normalize: Normalizes the input using mean and standard deviation, then applies scale and shift.
 * @param addNorm Pointer to AddNormalize structure containing parameters.
 * @param input Pointer to the input data array (size: seq_len_ x input_dim_)(data type: Q12).
 * @param input_normalized Pointer to the output normalized data array (size: seq_len_ x input_dim_)(data type: Q12).
 */
void normalize(AddNormalize *addNorm, quant_bit_width *input, quant_bit_width *input_normalized) {

    for (int i = 0; i < addNorm->seq_len_; i++) { // Go row by row
        quant_bit_width *input_ptr = input + i * (addNorm->input_dim_); // move pointer to the start of the row
        quant_bit_width *input_normalized_ptr = input_normalized + i * (addNorm->input_dim_); // move pointer to the start of the row

        // Compute sum of the row. Sum is int32_t and stores additions of Q12.
        //It does not take care of overflow but data should be well behaved because is int16_t and the sum buffer is int32_t.
        int sum = 0;
        for (int j = 0; j < addNorm->input_dim_; j++) {
            sum += *input_ptr;
            input_ptr++;
        }

        input_ptr = input + i * (addNorm->input_dim_); // Restart input_ptr to the beginning of the row
        // Compute mean of the row. The division is float! The result os Q12 without rescaling. Se pierde la parte fraccionaria.
        quant_bit_width mean = (quant_bit_width)((float)sum / (float)addNorm->input_dim_);

        // Compute variance. It is int64_t as Q24, to manage overflow
        int64_t variance = 0;
        for (int j = 0; j < addNorm->input_dim_; j++) {
            variance += MUL_HQ((*input_ptr - mean), (*input_ptr - mean));
            input_ptr++;
        }

        variance = SHIFT(variance); // Q24 to Q12
        float variance_float = (float)variance / (float)(addNorm->input_dim_); // Divide por N en float
        variance_float = variance_float / (float)(1 << NUM_FRACTION_BITS); // Q12 to float

        // Compute standard deviation and its inverse 
        float sd = sqrtf(variance_float); // Float
        float sd_inv = (float)(1 / (sd + 0.00001)); // prevent zero divide!
        quant_bit_width sd_inv_int = (quant_bit_width)(sd_inv * (1 << NUM_FRACTION_BITS)); // float to Q12

        input_ptr = input + i * (addNorm->input_dim_); // Restart input_ptr to the beginning of the row
        input_normalized_ptr = input_normalized + i * (addNorm->input_dim_);

        // Normalize the row and apply scale and shift // Only this part can be done on CGRA
        for (int j = 0; j < addNorm->input_dim_; j++) {
            // Normalize: (x - mu) * (1/sigma)
            *input_normalized_ptr = (quant_bit_width)MUL((*input_ptr - mean), sd_inv_int); // Q12
            // Scale and shift: y = x * weight + bias
            *input_normalized_ptr = (quant_bit_width)(MUL((*input_normalized_ptr), addNorm->weight_[j]) + addNorm->bias_[j]); // Q12
            input_ptr++;
            input_normalized_ptr++;
        }
    }
}

/**
 * @brief add: Element-wise addition of two arrays with overflow handling.
 * @param input Pointer to the first input array (size: seq_len x input_dim)(data type: Q12). This array will be updated with the result.
 * @param to_be_added Pointer to the second input array (size: seq_len x input_dim)(data type: Q12).
 * @param seq_len Number of rows in the arrays.
 * @param input_dim Number of columns in the arrays.
 */
void add(quant_bit_width *input, quant_bit_width *to_be_added, int seq_len, int input_dim) {
    int32_t sum;
    for (int i = 0; i < seq_len * input_dim; i++) {
        sum = input[i] + to_be_added[i];
        if ((quant_bit_width)sum != sum) // In case of overflow in 16 bits // To be studied for CGRA
            input[i] = (sum > 0) ? INT16_MAX : INT16_MIN;
        else
            input[i] = (quant_bit_width)sum;
    }
}
