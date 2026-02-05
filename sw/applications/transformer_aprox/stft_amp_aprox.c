#include "stft_amp.h"
#include <math.h>
#include <stdio.h>
#include "defines.h"
#include "../param.h"
#include <stdint.h>
#include <limits.h>
#include <stdbool.h> // For bool


// Q4.12 format
#define Q_FORMAT (1 << NUM_FRACTION_BITS)
#define Q_MULT_25(a) (((int32_t)(a) * 25) >> (7)) // Specialized multiplication by 25, optimized for Q4.12
#define MUL_32_32_TO_64(a, b) ((int64_t)(a) * (int64_t)(b))
// #define CLZ(x) __builtin_clz(x)


// Use a 513-entry LUT (can be adjusted)
#define LUT_SIZE 513
static int16_t log_lut[LUT_SIZE]; // int16_t LUT
void init_log_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        double x = 1.0 + (double)i / 256.0;
        log_lut[i] = (int16_t)round(log(x) * (1 << NUM_FRACTION_BITS)); // Q12
    }
}

// Function to compute the logarithm of the amplitude
quant_bit_width compute_log_amp(int32_t real, int32_t imag)
{    
    #if LOG_AMP_IMPL == LOG_AMP_FP
    return compute_log_amp_fp(real, imag);
    #elif LOG_AMP_IMPL == LOG_AMP_FXP_LUT
    return compute_log_amp_fxp_lut(real, imag);
    #elif LOG_AMP_IMPL == LOG_AMP_FXP_APPROX
    return compute_log_amp_fxp_approx(real, imag);
    #elif LOG_AMP_IMPL == MAGNITUDE_FXP
    return compute_magnitude_fxp(real, imag);
    #elif LOG_AMP_IMPL == MAGNITUDE_FXP_OPT
    return compute_magnitude_fxp_opt(real, imag);
    #endif
}

// Function to compute the logarithm of the amplitude
quant_bit_width compute_log_amp_fp(int32_t real, int32_t imag)
{
    real = MUL_HQ(real, 25) >> (NUM_FRACTION_BITS - 9);
    imag = MUL_HQ(imag, 25) >> (NUM_FRACTION_BITS - 9);
    int32_t real2 = MUL_LONG(real, real) >> NUM_FRACTION_BITS;
    int32_t imag2 = MUL_LONG(imag, imag) >> NUM_FRACTION_BITS;
    float pow2 = (float)(real2 + imag2) / (float)(1 << NUM_FRACTION_BITS);
    float amp = sqrtf(pow2);
    float stft = logf(amp + 1e-10f);
    quant_bit_width stft_int = (quant_bit_width)(stft * (1 << NUM_FRACTION_BITS));
    return stft_int;
}

quant_bit_width compute_log_amp_fxp_lut(int32_t real, int32_t imag) {
    // Static LUT and initialization flag
    static int16_t log_lut[LUT_SIZE];
    static bool lut_initialized = false;

    // Initialize LUT only once
    if (!lut_initialized) {
        for (int i = 0; i < LUT_SIZE; ++i) {
            double x = 1.0 + (double)i / 256.0;
            log_lut[i] = (int16_t)round(log(x) * (1 << NUM_FRACTION_BITS)); // Q12
        }
        lut_initialized = true;
    }

    real = MUL_HQ(real, 25) >> (NUM_FRACTION_BITS - 9);
    imag = MUL_HQ(imag, 25) >> (NUM_FRACTION_BITS - 9);

    int32_t real2 = (MUL_LONG(real, real) >> NUM_FRACTION_BITS);
    int32_t imag2 = (MUL_LONG(imag, imag) >> NUM_FRACTION_BITS);
    int32_t sqmag = real2 + imag2;


    if (sqmag == 0) {
        return (quant_bit_width)(-23 * (1 << NUM_FRACTION_BITS)); // Q12, -94208
    }

    // Newton-Raphson with adjusted scaling and iterations
    uint32_t x = (uint32_t)sqmag << 4; // Q16.4 - Less aggressive scaling
    uint32_t guess;

    int clz = __builtin_clz(x);
    int exponent = 31 - clz;
    guess = 1 << (exponent / 2);

    for (int i = 0; i < 5; ++i) {
        uint32_t div = x / guess;
        guess = (guess + div) >> 1;
    }
     uint32_t magnitude = guess;

    // Normalization, mantissa, and LUT index
    clz = __builtin_clz(magnitude);
    exponent = 31 - clz;
    uint32_t normalized = magnitude << clz;
    uint32_t mantissa = (normalized >> 3) & 0x01FFFFFF; //shift by 3 = 12(NUM_FRACTION_BITS) - 9 (LUT index)
    // Linear interpolation
    uint16_t lut_index = (mantissa >> 16); // Top 9 bits
    uint16_t alpha = mantissa & 0xFFFF;  // Fractional part

    int32_t log_mantissa = ((int32_t)log_lut[lut_index] * (int32_t)(0x10000 - alpha) + (int32_t)log_lut[lut_index + 1] * (int32_t)alpha) >> 16;

    // Combine exponent and mantissa.  log(2) * 2^12 = 2841
    int32_t log_amp = (exponent - 14) * 2841 + log_mantissa;  // Adjust exponent bias: 16-4+12-9-1=14

    return (quant_bit_width)log_amp;
}
// an alternative impl, using fixed-point arithmetic and approximations (Newton-Raphson for square root, bit manipulation for logarithm)
quant_bit_width compute_log_amp_fxp_approx(int32_t real, int32_t imag) {
    real = MUL_HQ(real, 25) >> (NUM_FRACTION_BITS - 9);
    imag = MUL_HQ(imag, 25) >> (NUM_FRACTION_BITS - 9);
    int32_t real2 = MUL_LONG(real, real) >> NUM_FRACTION_BITS;
    int32_t imag2 = MUL_LONG(imag, imag) >> NUM_FRACTION_BITS;
    int32_t sum_sq = (real2 + imag2) >> NUM_FRACTION_BITS;

    //  Approximation for sqrt and log.
    if (sum_sq == 0) return 0; // Handle zero case

    // 1. Fast Square Root Approximation (Newton-Raphson single iteration)
    int32_t approx_sqrt = (sum_sq + (1 << NUM_FRACTION_BITS)) >> 1; // Initial guess: average
    approx_sqrt = (approx_sqrt + (sum_sq / approx_sqrt)) >> 1;       // One Newton-Raphson iteration
    //Could improve this with another iteration or a lookup table for refinement

    // 2. Fast Log2 Approximation (using bit manipulation)
    //    This part assumes NUM_FRACTION_BITS is set for your fixed-point representation.
    int leading_zeros = __builtin_clz(approx_sqrt);
    int exponent = (31 - leading_zeros) - NUM_FRACTION_BITS; // Integer part
    int mantissa = (approx_sqrt << leading_zeros) & 0x7FFFFFFF;   // Fractional part

    // You can further refine mantissa using a small lookup table if higher precision is needed.
    // For a very rough approximation, we can use the top bits of the mantissa directly.

    // Combine exponent and mantissa (adjust scaling as needed)
    quant_bit_width log_approx = (exponent << NUM_FRACTION_BITS) + (mantissa >> (31 - NUM_FRACTION_BITS));

    return log_approx;
}

// Function to compute the magnitude of the complex number
quant_bit_width compute_magnitude_fxp(int32_t real, int32_t imag) {
    real = MUL_HQ(real, 25) >> (NUM_FRACTION_BITS - 9);
    imag = MUL_HQ(imag, 25) >> (NUM_FRACTION_BITS - 9);

    int32_t real2 = (MUL_LONG(real, real) >> NUM_FRACTION_BITS);
    int32_t imag2 = (MUL_LONG(imag, imag) >> NUM_FRACTION_BITS);
    int32_t sqmag = real2 + imag2;

    if (sqmag == 0) {
        return 0; // Handle zero case
    }

    // Newton-Raphson with adjusted scaling and iterations
    uint32_t x = (uint32_t)sqmag << 4; // Q16.4 - Less aggressive scaling
    uint32_t guess;

    int clz = __builtin_clz(x);
    int exponent = 31 - clz;
    guess = 1 << (exponent / 2);

    for (int i = 0; i < 5; ++i) {
        uint32_t div = x / guess;
        guess = (guess + div) >> 1;
    }
    return (quant_bit_width)guess;
}


// Reciprocal approximation (1/x) for Q4.12 using uint32_t.
static inline uint32_t reciprocal_approx(uint32_t x) {
    if (x == 0) {
        return UINT32_MAX; // Or handle as an error
    }

    int clz = __builtin_clz(x);
    int shift = clz - 1;  // Normalize to roughly [0.5, 1)
    uint32_t normalized_x = x << shift;

    // Initial guess and single Newton-Raphson iteration.
    uint32_t y = 4294918479u - (normalized_x >> 1);

	// Single Newton-Raphson iteration: y = y * (2 - x * y)
	y = (uint32_t)(MUL_32_32_TO_64(y,(Q_FORMAT * 2 - (MUL_32_32_TO_64(normalized_x,y) >> NUM_FRACTION_BITS))) >> NUM_FRACTION_BITS);

    return y >> (31 - shift);
}


//compute_magnitude_fxp_opt with int32_t inputs
quant_bit_width compute_magnitude_fxp_opt(int32_t real, int32_t imag) {
    // Use int64_t for intermediate calculations to prevent overflow
    int64_t real2 = (int64_t)Q_MULT_25(real) * (int64_t)Q_MULT_25(real);  //keep it in 64 bit
    int64_t imag2 = (int64_t)Q_MULT_25(imag) * (int64_t)Q_MULT_25(imag); //keep it in 64 bit

    int64_t sqmag64 = real2 + imag2;

    if (sqmag64 == 0) {
        return 0;
    }

    // Scale down after summing the squares.  Shift by 9 to get back closer to
    // the original magnitude (since inputs were scaled up by ~2^4.64 and squared).
    int32_t sqmag = (int32_t)(sqmag64 >> 9);

    // --- Optimized Square Root (Now operating on int32_t) ---

    // 1. Initial Guess
    int clz = __builtin_clz(sqmag);
    int exponent = (31 - clz) / 2;
	//Guess and reciprocal should use all 32 bits available.
    uint32_t guess = (uint32_t)(1U << exponent);

    // 2. Newton-Raphson (with Reciprocal, Two Iterations)
     uint32_t reciprocal = reciprocal_approx(guess);
     guess = (guess + ((MUL_32_32_TO_64(sqmag, reciprocal) >> NUM_FRACTION_BITS))) >> 1;

     reciprocal = reciprocal_approx(guess);
     guess = (guess + ((MUL_32_32_TO_64(sqmag, reciprocal) >> NUM_FRACTION_BITS))) >> 1;

    //Adjust final shift based on earlier scaling (9) and the fact guess is uint32_t.
    return (quant_bit_width)(guess >> 5);
}
