/**
 * cnn_models_c.c  --  CNN_1D_v2 (PYE) forward pass.
 *                     X-HEEP / int32 / STATIC VERSION.
 *
 * Contains only the functions required for CNN_1D_v2 inference:
 *   apply_activation()     -- dispatch to relu_inplace / sigmoid_inplace
 *   make_dummy_input_1d()  -- create a test input tensor
 *   apply_conv_bn_act()    -- fused Conv -> BN -> Activation (dummy path)
 *   apply_dense_bn_act()   -- fused Dense -> (optional BN) -> Activation (dummy path)
 *   run_cnn_1d_v2()        -- full forward pass, public entry point
 *
 * WEIGHT SELECTION:
 *   If weights_cnn_1d_v2.h is present in this directory and the build
 *   defines -DDEEPBINDI_REAL_WEIGHTS, the forward pass uses const arrays
 *   from that header (weights live in .rodata / flash, weight pool unused).
 *   Otherwise, deterministic dummy weights are generated from fixed seeds.
 *
 *   To build with real weights:
 *     make CFLAGS="-DTARGET_PC -DDEEPBINDI_REAL_WEIGHTS"
 *
 * INPUT FORMAT:
 *   Pass a pointer to 570 int32_t values to run_cnn_1d_v2().
 *   Layout: data[ch * 10 + t] for channel ch (0..56), time step t (0..9).
 *   Values in int8 range [-128, 127] widened to int32.
 *   Pass NULL for dummy input (benchmarking only).
 */
#include <stdint.h>
#include <stddef.h>
#include "cnn_models_c.h"
#include "arena.h"
#include "deepbindi_config.h"

/* ---- Real weights (optional) ------------------------------------------- */

#ifdef DEEPBINDI_REAL_WEIGHTS
#  include "weights_cnn_1d_v2.h"
#endif

/* ---- Activation kind (dummy path only) --------------------------------- */

#ifndef DEEPBINDI_REAL_WEIGHTS
typedef enum {
    ACT_RELU    = 0,
    ACT_SIGMOID = 1
} ActivationKind;

static void apply_activation(Tensor *tensor, ActivationKind act) {
    if (act == ACT_RELU) {
        relu_inplace(tensor);
    } else {
        sigmoid_inplace(tensor);
    }
}
#endif /* !DEEPBINDI_REAL_WEIGHTS */

/* ---- Input builder ----------------------------------------------------- */

static Tensor *make_dummy_input_1d(int channels, int length, int seed) {
    Tensor *input = tensor_create(1, channels, 1, length);
    tensor_fill_dummy(input, 50, seed);
    return input;
}

/* ---- Activation dispatch + fused building blocks (dummy path only) ------ */

#ifndef DEEPBINDI_REAL_WEIGHTS
/*
 * apply_conv_bn_act  --  Conv2D -> BatchNorm -> Activation (CGRA fusion target)
 * Used only when DEEPBINDI_REAL_WEIGHTS is NOT defined.
 */
static Tensor *apply_conv_bn_act(
    const Tensor *input,
    int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    ActivationKind activation,
    int seed)
{
    Conv2DLayer    conv = conv2d_layer_create(
        input->c, out_channels,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        groups, seed);
    BatchNormLayer bn   = batchnorm_layer_create(out_channels, seed + 1);
    Tensor *output = conv2d_forward(input, &conv);
    batchnorm_forward_inplace(output, &bn);
    apply_activation(output, activation);
    conv2d_layer_free(&conv);
    batchnorm_layer_free(&bn);
    return output;
}

/*
 * apply_dense_bn_act  --  Dense -> (optional BN) -> Activation (dummy path)
 */
static Tensor *apply_dense_bn_act(
    const Tensor *input,
    int out_features,
    int use_bn,
    ActivationKind activation,
    int seed)
{
    DenseLayer dense = dense_layer_create(input->c, out_features, seed);
    Tensor *output   = dense_forward(input, &dense);
    if (use_bn) {
        BatchNormLayer bn = batchnorm_layer_create(out_features, seed + 1);
        batchnorm_forward_inplace(output, &bn);
        batchnorm_layer_free(&bn);
    }
    apply_activation(output, activation);
    dense_layer_free(&dense);
    return output;
}
#endif /* !DEEPBINDI_REAL_WEIGHTS */

/* ---- CNN_1D_v2 forward pass -------------------------------------------- */

/*
 * run_cnn_1d_v2  (PYE: Model 4 in the DeepBindi paper)
 *
 * Architecture:
 *   Input   : (1, 57, 1, 10)  -- 57 features * 10 time frames
 *
 *   Block 1 : Conv1d(57->32, k=5, pad=1) -> BN(32) -> ReLU -> MaxPool1d(2)
 *             out_w = (10 + 2*1 - 5)/1 + 1 = 8  --> MaxPool -> 4
 *             shape after block: (1, 32, 1, 4)
 *
 *   Block 2 : Conv1d(32->64, k=5, pad=1) -> BN(64) -> ReLU -> MaxPool1d(2)
 *             out_w = (4 + 2*1 - 5)/1 + 1 = 2  --> MaxPool -> 1
 *             shape after block: (1, 64, 1, 1)
 *
 *   Head    : Flatten -> Dense(64->1) -> Threshold(0)
 *             output = 1 (FEAR) if pre-threshold value > 0, else 0 (NO_FEAR)
 *
 * @param input_data  570 int32_t values (channel-first), or NULL for dummy.
 * Returns: pointer to (1,1,1,1) tensor; data[0] == 0 (NO_FEAR) or 1 (FEAR).
 */
Tensor *run_cnn_1d_v2(const int32_t *input_data) {
    Tensor *input;
    Tensor *x;
    Tensor *pooled;
    Tensor *flat;
    Tensor *output;
    int i;

    /* Reset activation arena (always needed between forward passes).
     * Weight pool reset is only needed in dummy mode: real const weights
     * live in .rodata and do not consume the weight pool. */
#ifndef DEEPBINDI_REAL_WEIGHTS
    weight_arena_reset();
#endif
    act_arena_reset();

    /* ---- Input: (1, 57, 1, 10) ---------------------------------------- */
    if (input_data != NULL) {
        input = tensor_create(1, 57, 1, 10);
        for (i = 0; i < 570; ++i) {
            input->data[i] = input_data[i];
        }
    } else {
        input = make_dummy_input_1d(57, 10, 5);
    }

#ifdef DEEPBINDI_REAL_WEIGHTS
    /* ============================================================
     * REAL WEIGHTS PATH (weights_cnn_1d_v2.h)
     * Layer structs point directly into .rodata -- no pool usage.
     * ==================================================== */
    {
        Conv2DLayer    conv1 = conv2d_layer_from_weights(
            57, 32, 1, 5, 1, 1, 0, 1, 1,
            conv1_weight, conv1_bias);
        BatchNormLayer bn1   = batchnorm_layer_from_params(32, bn1_scale, bn1_offset);

        DEEPBINDI_TRACE("DBG: conv1 start\r\n");
        x = conv2d_forward(input, &conv1);
        DEEPBINDI_TRACE("DBG: conv1 done\r\n");
        batchnorm_forward_inplace(x, &bn1);
        DEEPBINDI_TRACE("DBG: bn1 done\r\n");
        relu_inplace(x);
        /* Right-shift 9 to prevent overflow in Conv2.
         * After BN1 (Q7 scale ≤ 377) activations can reach ~13.6M.
         * Conv2 has 160 MACs: 160 × 127 × 13.6M ≈ 276B >> INT32_MAX.
         * Shifting 9 bits: 13.6M >> 9 ≈ 26.6K; Conv2 max ≈ 540M ✓ */
        tensor_rshift_inplace(x, 9);
        tensor_free(input);
        pooled = maxpool2d_forward(x, 1, 2, 1, 2);
        DEEPBINDI_TRACE("DBG: pool1 done\r\n");
        tensor_free(x);

        Conv2DLayer    conv2 = conv2d_layer_from_weights(
            32, 64, 1, 5, 1, 1, 0, 1, 1,
            conv2_weight, conv2_bias);
        BatchNormLayer bn2   = batchnorm_layer_from_params(64, bn2_scale, bn2_offset);

        DEEPBINDI_TRACE("DBG: conv2 start\r\n");
        x = conv2d_forward(pooled, &conv2);
        DEEPBINDI_TRACE("DBG: conv2 done\r\n");
        batchnorm_forward_inplace(x, &bn2);
        DEEPBINDI_TRACE("DBG: bn2 done\r\n");
        relu_inplace(x);
        /* Right-shift 14 to prevent overflow in FC1.
         * After BN2 (Q7 scale ≤ 334) activations can reach ~1.41B.
         * FC1 has 64 inputs: 64 × 127 × 1.41B ≈ 11.5T >> INT32_MAX.
         * Shifting 14 bits: 1.41B >> 14 ≈ 86K; FC1 max ≈ 699M ✓ */
        tensor_rshift_inplace(x, 14);
        tensor_free(pooled);
        flat = maxpool2d_forward(x, 1, 2, 1, 2);
        DEEPBINDI_TRACE("DBG: pool2 done\r\n");
        tensor_free(x);

        flat = flatten_forward(flat);
        DEEPBINDI_TRACE("DBG: flatten done\r\n");
        DenseLayer fc1 = dense_layer_from_weights(64, 1, fc1_weight, fc1_bias);
        output = dense_forward(flat, &fc1);
        DEEPBINDI_TRACE("DBG: dense done\r\n");
        tensor_free(flat);
        sigmoid_inplace(output);
        DEEPBINDI_TRACE("DBG: sigmoid done\r\n");
    }
#else
    /* ============================================================
     * DUMMY WEIGHTS PATH (seeded random, for benchmarking only)
     * ==================================================== */
    x = apply_conv_bn_act(input, 32, 1, 5, 1, 1, 0, 1, 1, ACT_RELU, 180);
    tensor_free(input);
    pooled = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);

    x = apply_conv_bn_act(pooled, 64, 1, 5, 1, 1, 0, 1, 1, ACT_RELU, 190);
    tensor_free(pooled);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);

    x = flatten_forward(flat);
    tensor_free(flat);
    output = apply_dense_bn_act(x, 1, 0, ACT_SIGMOID, 200);
    tensor_free(x);
#endif /* DEEPBINDI_REAL_WEIGHTS */

    return output;
}
