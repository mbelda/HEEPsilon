//
// Created by Hossein Taji on 25/2/25.
//
#include "gelu.h"
#include <math.h>
#include "dense_layerC.h"
#include <stdio.h>
#include "defines.h"

#define NUM_FRACTION_BITS 12
#define M1 2048
#define M2 4294966784
#define P 4294965248
#define A 4294967280
#define B 4294966016

#define MUL(x, y) (int32_t) (((int32_t)(x) * (int32_t)(y)) >> NUM_FRACTION_BITS)


// GELU activation function
void gelu(Dense *dense, size_t length, int16_t *input, int16_t *output)
{
    #if GELU_IMPL == GELU_FP
    gelu_fp(dense, length, input, output);
    #elif GELU_IMPL == GELU_PWL
    gelu_pwl(dense, length, input, output);
    #endif
}

void gelu_fp(Dense *dense, size_t length, int16_t *input, int16_t *output)
{
    float in_float, in_tanh;
    int32_t x3, in_tanh_fxp;
    int32_t a, b, c;
    int32_t qb, qc;
    int32_t q_sign, q;
    int32_t q_L, S_L;
    for (int i = 0; i < length; i++)
    {
        x3 = MUL(MUL(input[i], input[i]), input[i]);
        x3 = MUL(x3, 183); // 183 = 0.044715 in fixed-point 12 bit
        x3 += input[i];
        x3 = MUL(x3, 3268); // 3268 = sqrt(2/PI) in fixed-point 12 bit
        in_float = (float)x3 / (float)(1 << NUM_FRACTION_BITS);
        in_tanh = tanhf(in_float);
        in_tanh_fxp = (int16_t)(in_tanh * (1 << NUM_FRACTION_BITS));
        in_tanh_fxp += (1 << NUM_FRACTION_BITS);
        output[i] = MUL(in_tanh_fxp, input[i] >> 1);
    }
}

void gelu_pwl(Dense *dense, size_t length, int16_t *input, int16_t *output)
{
    for (size_t i = 0; i < length; i++)
    {
        int16_t x = input[i]; // Current input
        int16_t result;
        // Piecewise conditions
        if (x >= 0)
        {
            result = x; // x >= 0
        }
        else if (x >= (int32_t)P && x < 0)
        {
            result = MUL((int32_t)M1, x); // P <= x < 0
        }
        else if (x >= (int32_t)A && x < (int32_t)P)
        {
            result = MUL((int32_t)M2, x) + (int32_t)B; // a <= x < P
        }
        else
        {
            result = 0; // x < a
        }
        output[i] = result; // Store the result in the output array
    }
}

uint32_t mse(size_t length, int16_t *a, int16_t *b)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        int32_t diff = a[i] - b[i];
        sum += abs(diff) * abs(diff);
    }
    float mse_float = (float)sum / (float)length;
    uint32_t mse_fixed = (uint32_t)round(mse_float * (1 << NUM_FRACTION_BITS));
    return mse_fixed;
}

void print_fixed_as_float(uint32_t fixed) {
    uint32_t integer_part = fixed >> NUM_FRACTION_BITS;
    uint32_t fractional_part = fixed & ((1 << NUM_FRACTION_BITS) - 1);
    uint32_t fractional_scaled = (fractional_part * 100000) >> NUM_FRACTION_BITS; // 4 decimal places
    printf("%d.%05d\n", integer_part, fractional_scaled);
}