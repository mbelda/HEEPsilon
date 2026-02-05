//
// Created by alireza on 10/6/23.
// Approx version by Hossein Taji and Francesco Poluzzi on 25/2/25.
//

#include "softmaxC.h"
#include "defines.h"

#define USE_RCF

#define N_TAYLOR_COEFF 3
int32_t factorials[N_TAYLOR_COEFF-1] = {1<<NUM_FRACTION_BITS, 2<<NUM_FRACTION_BITS, 6<<NUM_FRACTION_BITS, 24<<NUM_FRACTION_BITS, 120<<NUM_FRACTION_BITS};

#define FP_ONE (1 << NUM_FRACTION_BITS)
#define EPSILON 1


//  === Below values are used for ConSmax ===
int16_t gamma_inv_fxp = 41;  //in Q4.12, 40.96 which is equal to 0.01 (or gamma = 100.0) in float -> 0.01 * 4096 = 41
int16_t beta_fxp = 6144;  //in Q4.12, 1.5 in float -> 1.5 * 4096 = 6144
// =========================================

/* 
    Pre-calculated reciprocals of factorials in Q4.12 format.
    Calculated as: (int32_t)(((int64_t)FP_ONE * FP_ONE) / factorial)
    where FP_ONE = (1 << NUM_FRACTION_BITS) = 4096
    and factorial = 1, 2, 6, 24, 120

    1/1! * 2^24 = 16777216
    1/2! * 2^24 = 8388608
    1/3! * 2^24 = 2796203 (rounded)
    1/4! * 2^24 = 699051 (rounded)
    1/5! * 2^24 = 139810 (rounded)

    To recalculate:
    1. Change NUM_FRACTION_BITS if your Q format changes (e.g., Q4.8 would be 8).
    2. Use a calculator (programming calculator or Python is good) to do 64-bit integer math.
    3. For each factorial (1!, 2!, 3!, 4!, 5!):
    a. Calculate (1 << (2 * NUM_FRACTION_BITS))  // This is FP_ONE * FP_ONE
    b. Divide the result of (a) by the factorial.
    c. Round to the nearest integer.  This is your reciprocal_factorials value.

    int32_t reciprocal_factorials[N_TAYLOR_COEFF - 1] = {
        16777216, 8388608, 2796203, 699051, 139810
    };
*/
int32_t reciprocal_factorials[N_TAYLOR_COEFF - 1] = {16777216, 8388608, 2796203, 699051, 139810};

void computeSoftmax(int16_t* input, size_t seq_len) {
    #if SM_IMPL == SM_FP
    computeSoftmax_fp(input, seq_len);
    #elif SM_IMPL == SM_SOFTERMAX
    softermax(input, seq_len, seq_len);
    #elif SM_IMPL == SM_FIXED
    computeSoftmax_nonsquare_fixed(input, seq_len, seq_len);
    #elif SM_IMPL == SM_ConSmax
    consmax(input, seq_len, seq_len, beta_fxp, gamma_inv_fxp);
    #endif
}

// softmax scales a matrix into values between 0 and 1 => turns into a probability distribution
// consider for square matrices
void computeSoftmax_fp(int16_t* input, size_t seq_len) {
    size_t width = seq_len;
    float input_float = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        int16_t max_val = input[i * seq_len];
        for (int j = 1; j < width; j++) {
            if (input[i * seq_len + j] > max_val) {
                max_val = input[i * seq_len + j];
            }
        }
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = FIXED_MAX(input[i * seq_len + j] - max_val, -32767);
        }
        int32_t sum = 0;
        for (int j = 0; j < width; j++) {
            input_float = (float) input[i * seq_len + j] / (float) (1 << NUM_FRACTION_BITS);
            input_float = expf(input_float);
            input[i * seq_len + j] = (int16_t) (input_float * (1 << NUM_FRACTION_BITS));
            sum += input[i * seq_len + j];
        }
        float sum_float = (float) sum / (float) (1 << NUM_FRACTION_BITS);
        float sum_inv = (float) (1 / (sum_float + 0.00001)); // prevent zero divide!
        int16_t sum_inv_int = (int16_t) (sum_inv * (1 << NUM_FRACTION_BITS));
        for (int j = 0; j < width; j++) {
            input[i * seq_len + j] = (int16_t) MUL(input[i * seq_len + j], sum_inv_int);
        }
    }
}

int32_t fixed_div(int32_t numerator, int32_t denominator) {
    int64_t scaled_numerator = (int64_t)numerator << NUM_FRACTION_BITS;
    int32_t result = (int32_t)(scaled_numerator / denominator); //TODO: simply put multipliication to see difference in timing
    return result;
}

int32_t exp_fixed_point_taylor(int32_t input){
    // int32_t result = (1 << NUM_FRACTION_BITS);
    int32_t result = FP_ONE; // 1.0 in Q4.12
    int32_t x_pow_n = input;
    for (int i = 0; i < N_TAYLOR_COEFF-1; i++) {
        #ifdef USE_RCF
        result += MUL(x_pow_n, reciprocal_factorials[i]);
        #else
        result += fixed_div(x_pow_n, factorials[i]); //TODO: use LUT and MUL
        #endif    
        x_pow_n = MUL(x_pow_n, input);
    }
    return result;
}


// alternative implementation of softmax that works with non-square matrices
void computeSoftmax_nonsquare(int16_t* input, size_t num_rows, size_t num_cols) {
    float input_float = 0.0f;
    for (int i = 0; i < num_rows; i++) {
        int16_t max_val = input[i * num_cols];
        for (int j = 1; j < num_cols; j++) {
            if (input[i * num_cols + j] > max_val) {
                max_val = input[i * num_cols + j];
            }
        }
        for (int j = 0; j < num_cols; j++) {
            input[i * num_cols + j] = (int16_t) FIXED_MAX(input[i * num_cols + j] - max_val, -32767);
        }
        int32_t sum = 0;
        for (int j = 0; j < num_cols; j++) {
            input_float = (float) input[i * num_cols + j] / (float) (1 << NUM_FRACTION_BITS);
            input_float = expf(input_float);
            input[i * num_cols + j] = (int16_t) (input_float * (1 << NUM_FRACTION_BITS));
            sum += input[i * num_cols + j];
        }
        float sum_float = (float) sum / (float) (1 << NUM_FRACTION_BITS);
        float sum_inv = (float) (1 / (sum_float + 0.00001f)); // prevent zero divide!
        int16_t sum_inv_int = (int16_t) (sum_inv * (1 << NUM_FRACTION_BITS));
        for (int j = 0; j < num_cols; j++) {
            input[i * num_cols + j] = (int16_t) MUL(input[i * num_cols + j], sum_inv_int);
        }
    }
}


void computeSoftmax_nonsquare_fixed(int16_t* input, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; i++) {
        int16_t max_val = input[i * num_cols];
        for (size_t j = 1; j < num_cols; j++) {
            int16_t val = input[i * num_cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        for (size_t j = 0; j < num_cols; j++) {
            int32_t diff = (int32_t)input[i * num_cols + j] - (int32_t)max_val;
            if (diff > 32767) diff = 32767;
            if (diff < -32768) diff = -32768;
            input[i * num_cols + j] = (int16_t)diff;
        }
        // Compute exponent and sum
        int32_t sum = 0;
        for (size_t j = 0; j < num_cols; j++) {
            int32_t val = (int32_t)input[i * num_cols + j];
            int32_t exp_val = exp_fixed_point_taylor(val); 
            if (exp_val > 32767) exp_val = 32767;
            if (exp_val < -32768) exp_val = -32768;
            input[i * num_cols + j] = (int16_t)exp_val;
            sum += exp_val; 
        }
        int32_t sum_inv = fixed_div(FP_ONE, sum + EPSILON);
        // Multiply each element by sum_inv to normalize
        for (size_t j = 0; j < num_cols; j++) {
            int32_t val = (int32_t)input[i * num_cols + j];
            int32_t normalized = MUL(val, sum_inv); 
            if (normalized > 32767) normalized = 32767;
            if (normalized < -32768) normalized = -32768;
            input[i * num_cols + j] = (int16_t)normalized;
        }
    }
}

// alternative implementation of softmax that works with non-square matrices
void softermax(int16_t* input, size_t num_rows, size_t num_cols) {
    int16_t max_values[num_cols];
    max_values[0] = -32767;
    for (int i = 0; i < num_rows; i++) {
        int32_t d = 0;
        for (int j = 1; j < num_cols; j++) {
            int16_t current_val = input[i * num_cols + j];
            int16_t max_val= FIXED_MAX(current_val, max_values[j-1]);
            max_values[j] = max_val;
            int32_t exp_term = exp_fixed_point_taylor(current_val - max_val);
            d = (d>>(max_val-max_values[j-1]))+exp_term;
            input[i * num_cols + j] = exp_term;
        }
        int16_t absolute_max = max_values[num_cols-1];
        for (int j = 0; j < num_cols; j++) {
            input[i * num_cols + j] = (int16_t)  fixed_div( (input[i * num_cols + j]>>(absolute_max-max_values[j])), d);
        }
    }
}

// change divisions to multiplication


void consmax(int16_t* input, size_t num_rows, size_t num_cols, int16_t beta, int16_t gamma_inv) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            // 1. Subtract beta from the input value
            int32_t shifted_input = (int32_t)input[i * num_cols + j] - (int32_t)beta;

            // 2. Calculate the exponent (using your existing exp_fixed_point_taylor function)
            int32_t exp_result = exp_fixed_point_taylor(shifted_input);

            // 3. Divide (multiply by inverse) by gamma_inv for normalization
            int32_t normalized = MUL(exp_result, gamma_inv); // Assuming 'gamma' is precomputed inverse

            // 4. Saturate the result to the desired range (if necessary)
            if (normalized > 32767) normalized = 32767;
            if (normalized < -32768) normalized = -32768;

            input[i * num_cols + j] = (int16_t)normalized;
        }
    }
}



// FOR YUXUAN COMPILER

// #define MUL_LONG(x, y) (int64_t) (((int64_t)(x) * (int64_t)(y)))
// #define MUL_HQ(x, y) (int32_t) (((int32_t)(x) * (int32_t)(y)))
// #define SHIFT(x) ((x) >> NUM_FRACTION_BITS)

// #define N_TAYLOR_COEFF 3
// int32_t factorials[N_TAYLOR_COEFF-1] = {1<<NUM_FRACTION_BITS, 2<<NUM_FRACTION_BITS, 6<<NUM_FRACTION_BITS, 24<<NUM_FRACTION_BITS, 120<<NUM_FRACTION_BITS};


// #define EPSILON 1


// //  === Below values are used for ConSmax ===


// #define NUM_FRACTION_BITS 12
// #define MUL(x, y) (int32_t) (((int32_t)(x) * (int32_t)(y)) >> NUM_FRACTION_BITS)
// #define FP_ONE (1 << NUM_FRACTION_BITS)

// uint8_t Q_ptr[1] = {NUM_FRACTION_BITS};
// int32_t gamma_inv_fxp[1] = {41};  //in Q4.12, 40.96 which is equal to 0.01 (or gamma = 100.0) in float -> 0.01 * 4096 = 41
// int32_t beta_ptr = {6144};  //in Q4.12, 1.5 in float -> 1.5 * 4096 = 6144
// uint32_t num_rows_ptr[1] = {121};
// uint32_t num_cols_ptr[1] = {121};
// int32_t reciprocal_factorials_ptr[N_TAYLOR_COEFF - 1] = {16777216, 8388608, 2796203, 699051, 139810};


// void consmax_comp(uint32_t num_rows_ptr[1], uint32_t num_cols_ptr[1], int32_t beta_ptr[1], int32_t gamma_inv_ptr[1], uint8_t Q_ptr[1], int32_t reciprocal_factorials_ptr[N_TAYLOR_COEFF-1], int32_t * input_ptr) {
//     // INPUT_SIZE = num_rows_ptr[0] * num_cols_ptr[0]
//     for (int i = 0; i < num_rows_ptr[0]; i++) {
//         for (int j = 0; j < num_cols_ptr[0]; j++) {
//             // 1. Subtract beta from the input value
//             int32_t shifted_input = (int32_t)input_ptr[i * num_cols_ptr[0] + j] - (int32_t)beta_ptr[0];

//             // 2. Calculate the exponent (using your existing exp_fixed_point_taylor function)
//             int32_t result = FP_ONE; // 1.0 in Q4.12
//             int32_t x_pow_n = shifted_input;
//             for (int i = 0; i < N_TAYLOR_COEFF-1; i++) {
//                 result += MUL(x_pow_n, reciprocal_factorials[i]);  
//                 x_pow_n = MUL(x_pow_n, shifted_input);
//             }
//             int32_t exp_result = result;

//             // 3. Divide (multiply by inverse) by gamma_inv for normalization
//             int32_t normalized = MUL(exp_result, gamma_inv_ptr[0]); // Assuming 'gamma' is precomputed inverse

//             // 4. Saturate the result to the desired range (if necessary)
//             if (normalized > 32767) normalized = 32767;
//             if (normalized < -32768) normalized = -32768;

//             input_ptr[i * num_cols_ptr[0] + j] = normalized;
//         }
//     }
// }
