/**
 * mobile_models_c.c  --  MobileCNN-1D and MobileCNN-SE-1D forward passes.
 *                         X-HEEP / int32 / STATIC VERSION.
 *
 * Provides run_mobilecnn_1d() and run_mobilecnn_se_1d().
 * Architecture: depthwise-separable convolution blocks (MobileNetV1 style).
 * SE block uses integer hard-sigmoid and Q7 channel-wise scaling.
 *
 * Real weights:  compile with -DDEEPBINDI_REAL_WEIGHTS.
 *   mobile1d:    include weights_mobile1d.h via export_mobile_weights.py --model mobile1d
 *   mobile_se:   include weights_mobile_se.h via export_mobile_weights.py --model mobile_se
 *
 * Dummy weights: seeded deterministic generation (benchmarking only).
 *
 * OVERFLOW POLICY (real weights path):
 *   batchnorm_rshift_inplace(tensor, &bn, SHIFT) fuses BN and post-shift in a
 *   single int64_t pass, preventing overflow before the int32_t cast.  This is
 *   critical for BN_PW1 where the BN intermediate exceeds INT32_MAX (2.7B for
 *   mobile1d, 4.5B for mobile_se).  All BN layers use the fused call for
 *   uniformity.  MOBILE_SHIFT_* constants are from export_mobile_weights.py.
 *
 * MEMORY:
 *   All tensors are bump-allocated from g_act_arena (no real frees).
 *   Peak: 1731 words (mobile1d), 1803 words (mobile_se) -- both < 2048.
 *   Weight pool is unused in real-weights build (weights in .rodata).
 */
#include <stdint.h>
#include <stddef.h>
#include "mobile_models_c.h"
#include "arena.h"

/* ---- Real weights (optional) --------------------------------------------- */

#ifndef DEEPBINDI_MODEL
#  define DEEPBINDI_MODEL 0
#endif

#if defined(DEEPBINDI_REAL_WEIGHTS) && (DEEPBINDI_MODEL == 1 || DEEPBINDI_MODEL == 2)
#  define DEEPBINDI_MOBILE_REAL_WEIGHTS 1
#else
#  define DEEPBINDI_MOBILE_REAL_WEIGHTS 0
#endif

#if DEEPBINDI_MOBILE_REAL_WEIGHTS
#  if DEEPBINDI_MODEL == 1
#    include "weights_mobile1d.h"
#  elif DEEPBINDI_MODEL == 2
#    include "weights_mobile_se.h"
#  endif
#endif

/* ---- Input helper --------------------------------------------------------- */

static Tensor *make_input_1d(const int32_t *input_data) {
    Tensor *input = tensor_create(1, 57, 1, 10);
    int i;
    if (input_data != NULL) {
        for (i = 0; i < 570; ++i) {
            input->data[i] = input_data[i];
        }
    } else {
        tensor_fill_dummy(input, 50, 5);
    }
    return input;
}

/* ---- Dummy-path helpers --------------------------------------------------- */

#if !DEEPBINDI_MOBILE_REAL_WEIGHTS

/*
 * apply_dw_bn_relu  --  DWConv + BN + ReLU (dummy weights, seeded).
 * groups = input->c  (depthwise: one kernel per input channel).
 */
static Tensor *apply_dw_bn_relu(const Tensor *input, int kw, int pad, int seed) {
    int in_ch = input->c;
    Conv2DLayer   dw = conv2d_layer_create(in_ch, in_ch, 1, kw, 1, 1, 0, pad, in_ch, seed);
    BatchNormLayer bn = batchnorm_layer_create(in_ch, seed + 1);
    Tensor *out = conv2d_forward(input, &dw);
    batchnorm_forward_inplace(out, &bn);
    relu_inplace(out);
    conv2d_layer_free(&dw);
    batchnorm_layer_free(&bn);
    return out;
}

/*
 * apply_pw_bn_relu_pool  --  PWConv (k=1) + BN + ReLU + MaxPool(2) (dummy).
 */
static Tensor *apply_pw_bn_relu_pool(const Tensor *input, int out_ch, int seed) {
    Conv2DLayer    pw   = conv2d_layer_create(input->c, out_ch, 1, 1, 1, 1, 0, 0, 1, seed);
    BatchNormLayer bn   = batchnorm_layer_create(out_ch, seed + 1);
    Tensor *pw_out      = conv2d_forward(input, &pw);
    batchnorm_forward_inplace(pw_out, &bn);
    relu_inplace(pw_out);
    Tensor *pooled = maxpool2d_forward(pw_out, 1, 2, 1, 2);
    tensor_free(pw_out);
    conv2d_layer_free(&pw);
    batchnorm_layer_free(&bn);
    return pooled;
}

/*
 * apply_se_block_dummy  --  SE attention block (dummy weights, benchmarking).
 * Modifies feat in-place; returns feat.
 * In dummy mode all BN scales are 128 (identity) and weights are tiny,
 * so hardsigmoid_se_inplace(x, 0) maps near-zero x to ~64 (0.5 in Q7).
 */
static Tensor *apply_se_block_dummy(Tensor *feat, int ch1, int seed) {
    int se_inner = ch1 / 4;  /* 32/4 = 8 */
    Tensor *se_avg = globalavgpool_forward(feat);

    DenseLayer d1  = dense_layer_create(ch1, se_inner, seed);
    Tensor *se_z   = dense_forward(se_avg, &d1);
    tensor_free(se_avg);
    relu_inplace(se_z);

    DenseLayer d2  = dense_layer_create(se_inner, ch1, seed + 1);
    Tensor *se_w   = dense_forward(se_z, &d2);
    tensor_free(se_z);

    /* shift=0: for dummy values in ~[-100, 100], y = clip(x + 64, 0, 128) */
    hardsigmoid_se_inplace(se_w, 0);
    se_channel_scale_inplace(feat, se_w);
    tensor_free(se_w);

    dense_layer_free(&d1);
    dense_layer_free(&d2);
    return feat;
}

/*
 * apply_fc_threshold  --  Dense(in->1) + threshold(0) -> 0 or 1 (dummy).
 */
static Tensor *apply_fc_threshold(const Tensor *input, int seed) {
    DenseLayer fc  = dense_layer_create(input->c, 1, seed);
    Tensor *out    = dense_forward(input, &fc);
    sigmoid_inplace(out);   /* sign threshold: output = (x > 0) ? 1 : 0 */
    dense_layer_free(&fc);
    return out;
}

#endif /* !DEEPBINDI_MOBILE_REAL_WEIGHTS */


/* ---- run_mobilecnn_1d ----------------------------------------------------- */
/*
 * MobileCNN-1D (Option A): depthwise-separable blocks, no SE attention.
 *
 * Tensor flow (real-weights build, shape after each op):
 *   input         (1, 57, 1, 10)
 *   dw1_out       (1, 57, 1, 8)   DWConv k=5, pad=1, groups=57
 *   [BN_DW1 + ReLU + rshift SHIFT_DW1 in-place]
 *   pw1_out       (1, 32, 1, 8)   PWConv k=1
 *   [BN_PW1 + ReLU + rshift SHIFT_PW1 in-place]
 *   pool1         (1, 32, 1, 4)   MaxPool k=2, s=2
 *   dw2_out       (1, 32, 1, 2)   DWConv k=5, pad=1, groups=32
 *   [BN_DW2 + ReLU + rshift SHIFT_DW2 in-place]
 *   pw2_out       (1, 64, 1, 2)   PWConv k=1
 *   [BN_PW2 + ReLU + rshift SHIFT_PW2 in-place]
 *   pool2         (1, 64, 1, 1)   MaxPool k=2, s=2
 *   flat          (1, 64, 1, 1)   flatten
 *   output        (1,  1, 1, 1)   Dense(64->1) + threshold(0)
 *
 * Activation arena peak: 1731 words (all 9 tensors live at top of arena).
 */
Tensor *run_mobilecnn_1d(const int32_t *input_data) {
    Tensor *input;
    Tensor *dw1_out, *pw1_out, *pool1;
    Tensor *dw2_out, *pw2_out, *pool2;
    Tensor *flat, *output;

#if !DEEPBINDI_MOBILE_REAL_WEIGHTS
    weight_arena_reset();
#endif
    act_arena_reset();

    input = make_input_1d(input_data);

#if DEEPBINDI_MOBILE_REAL_WEIGHTS
    /* ============================================================
     * REAL WEIGHTS PATH  (weights_mobile1d.h)
     * =========================================================== */
    {
        Conv2DLayer    dw1_l = conv2d_layer_from_weights(
            57, 57, 1, 5, 1, 1, 0, 1, 57, mobile_dw1_weight, mobile_dw1_bias);
        BatchNormLayer bn_dw1 = batchnorm_layer_from_params(
            57, mobile_bn_dw1_scale, mobile_bn_dw1_offset);

        dw1_out = conv2d_forward(input, &dw1_l);
        batchnorm_rshift_inplace(dw1_out, &bn_dw1, MOBILE_SHIFT_DW1);
        relu_inplace(dw1_out);
        tensor_free(input);

        Conv2DLayer    pw1_l = conv2d_layer_from_weights(
            57, 32, 1, 1, 1, 1, 0, 0, 1, mobile_pw1_weight, mobile_pw1_bias);
        BatchNormLayer bn_pw1 = batchnorm_layer_from_params(
            32, mobile_bn_pw1_scale, mobile_bn_pw1_offset);

        pw1_out = conv2d_forward(dw1_out, &pw1_l);
        batchnorm_rshift_inplace(pw1_out, &bn_pw1, MOBILE_SHIFT_PW1);
        relu_inplace(pw1_out);
        tensor_free(dw1_out);

        pool1 = maxpool2d_forward(pw1_out, 1, 2, 1, 2);
        tensor_free(pw1_out);

        Conv2DLayer    dw2_l = conv2d_layer_from_weights(
            32, 32, 1, 5, 1, 1, 0, 1, 32, mobile_dw2_weight, mobile_dw2_bias);
        BatchNormLayer bn_dw2 = batchnorm_layer_from_params(
            32, mobile_bn_dw2_scale, mobile_bn_dw2_offset);

        dw2_out = conv2d_forward(pool1, &dw2_l);
        batchnorm_rshift_perchannel(dw2_out, &bn_dw2, mobile_bn_dw2_shift);
        relu_inplace(dw2_out);
        tensor_free(pool1);

        Conv2DLayer    pw2_l = conv2d_layer_from_weights(
            32, 64, 1, 1, 1, 1, 0, 0, 1, mobile_pw2_weight, mobile_pw2_bias);
        BatchNormLayer bn_pw2 = batchnorm_layer_from_params(
            64, mobile_bn_pw2_scale, mobile_bn_pw2_offset);

        pw2_out = conv2d_forward(dw2_out, &pw2_l);
        batchnorm_rshift_inplace(pw2_out, &bn_pw2, MOBILE_SHIFT_PW2);
        relu_inplace(pw2_out);
        tensor_free(dw2_out);

        pool2 = maxpool2d_forward(pw2_out, 1, 2, 1, 2);
        tensor_free(pw2_out);

        flat = flatten_forward(pool2);
        tensor_free(pool2);

        DenseLayer fc = dense_layer_from_weights(64, 1, mobile_fc_weight, mobile_fc_bias);
        output = dense_forward(flat, &fc);
        tensor_free(flat);
        sigmoid_inplace(output);  /* sign threshold: 0 or 1 */
    }
#else
    /* ============================================================
     * DUMMY WEIGHTS PATH (seeded random, benchmarking only)
     * =========================================================== */
    dw1_out = apply_dw_bn_relu(input, 5, 1, 10);
    tensor_free(input);
    pw1_out = apply_pw_bn_relu_pool(dw1_out, 32, 20);
    tensor_free(dw1_out);

    /* pool1 = pw1_out (returned by apply_pw_bn_relu_pool) */
    pool1   = pw1_out;

    dw2_out = apply_dw_bn_relu(pool1, 5, 1, 30);
    tensor_free(pool1);
    pw2_out = apply_pw_bn_relu_pool(dw2_out, 64, 40);
    tensor_free(dw2_out);

    pool2  = pw2_out;
    flat   = flatten_forward(pool2);
    tensor_free(pool2);
    output = apply_fc_threshold(flat, 50);
    tensor_free(flat);
#endif /* DEEPBINDI_MOBILE_REAL_WEIGHTS */

    return output;
}


/* ---- run_mobilecnn_se_1d -------------------------------------------------- */
/*
 * MobileCNN-SE-1D (Option B): same blocks as run_mobilecnn_1d, with SE
 * channel attention inserted after Block 1's MaxPool output.
 *
 * SE block tensor flow (pool1 shape: (1, 32, 1, 4)):
 *   se_avg  (1, 32, 1, 1)  GlobalAvgPool of pool1
 *   se_d1   (1,  8, 1, 1)  Dense(32->8) -> ReLU -> rshift SHIFT_D1
 *   se_d2   (1, 32, 1, 1)  Dense(8->32) -> HardSigmoid Q7 [0,128]
 *   pool1 modified in-place by se_channel_scale_inplace(pool1, se_d2)
 *
 * Activation arena peak: 1803 words (12 live tensors at peak).
 */
Tensor *run_mobilecnn_se_1d(const int32_t *input_data) {
    Tensor *input;
    Tensor *dw1_out, *pw1_out, *pool1;
    Tensor *dw2_out, *pw2_out, *pool2;
    Tensor *flat, *output;

#if !DEEPBINDI_MOBILE_REAL_WEIGHTS
    weight_arena_reset();
#endif
    act_arena_reset();

    input = make_input_1d(input_data);

#if DEEPBINDI_MOBILE_REAL_WEIGHTS
    /* ============================================================
     * REAL WEIGHTS PATH  (weights_mobile_se.h)
     * =========================================================== */
    {
        /* ---- Block 1 ---- */
        Conv2DLayer    dw1_l = conv2d_layer_from_weights(
            57, 57, 1, 5, 1, 1, 0, 1, 57, mobile_dw1_weight, mobile_dw1_bias);
        BatchNormLayer bn_dw1 = batchnorm_layer_from_params(
            57, mobile_bn_dw1_scale, mobile_bn_dw1_offset);

        dw1_out = conv2d_forward(input, &dw1_l);
        batchnorm_rshift_inplace(dw1_out, &bn_dw1, MOBILE_SHIFT_DW1);
        relu_inplace(dw1_out);
        tensor_free(input);

        Conv2DLayer    pw1_l = conv2d_layer_from_weights(
            57, 32, 1, 1, 1, 1, 0, 0, 1, mobile_pw1_weight, mobile_pw1_bias);
        BatchNormLayer bn_pw1 = batchnorm_layer_from_params(
            32, mobile_bn_pw1_scale, mobile_bn_pw1_offset);

        pw1_out = conv2d_forward(dw1_out, &pw1_l);
        batchnorm_rshift_inplace(pw1_out, &bn_pw1, MOBILE_SHIFT_PW1);
        relu_inplace(pw1_out);
        tensor_free(dw1_out);

        pool1 = maxpool2d_forward(pw1_out, 1, 2, 1, 2);
        tensor_free(pw1_out);

        /* ---- SE block (applied to pool1 in-place, MODEL 2 only) ---- */
#if DEEPBINDI_MODEL == 2
        {
            Tensor *se_avg, *se_d1, *se_d2;
            DenseLayer se_l1 = dense_layer_from_weights(
                32, 8, mobile_se_fc1_weight, mobile_se_fc1_bias);
            DenseLayer se_l2 = dense_layer_from_weights(
                8, 32, mobile_se_fc2_weight, mobile_se_fc2_bias);

            se_avg = globalavgpool_forward(pool1);           /* (1,32,1,1) */
            se_d1  = dense_forward(se_avg, &se_l1);          /* (1, 8,1,1) */
            tensor_free(se_avg);
            relu_inplace(se_d1);
            tensor_rshift_inplace(se_d1, MOBILE_SE_SHIFT_D1);

            se_d2  = dense_forward(se_d1, &se_l2);           /* (1,32,1,1) */
            tensor_free(se_d1);
            hardsigmoid_se_inplace(se_d2, MOBILE_SE_SHIFT_HS);

            se_channel_scale_inplace(pool1, se_d2);          /* in-place on pool1 */
            tensor_free(se_d2);
        }
#endif /* DEEPBINDI_MODEL == 2 */

        /* ---- Block 2 ---- */
        Conv2DLayer    dw2_l = conv2d_layer_from_weights(
            32, 32, 1, 5, 1, 1, 0, 1, 32, mobile_dw2_weight, mobile_dw2_bias);
        BatchNormLayer bn_dw2 = batchnorm_layer_from_params(
            32, mobile_bn_dw2_scale, mobile_bn_dw2_offset);

        dw2_out = conv2d_forward(pool1, &dw2_l);
        batchnorm_rshift_perchannel(dw2_out, &bn_dw2, mobile_bn_dw2_shift);
        relu_inplace(dw2_out);
        tensor_free(pool1);

        Conv2DLayer    pw2_l = conv2d_layer_from_weights(
            32, 64, 1, 1, 1, 1, 0, 0, 1, mobile_pw2_weight, mobile_pw2_bias);
        BatchNormLayer bn_pw2 = batchnorm_layer_from_params(
            64, mobile_bn_pw2_scale, mobile_bn_pw2_offset);

        pw2_out = conv2d_forward(dw2_out, &pw2_l);
        batchnorm_rshift_inplace(pw2_out, &bn_pw2, MOBILE_SHIFT_PW2);
        relu_inplace(pw2_out);
        tensor_free(dw2_out);

        pool2 = maxpool2d_forward(pw2_out, 1, 2, 1, 2);
        tensor_free(pw2_out);

        flat = flatten_forward(pool2);
        tensor_free(pool2);

        DenseLayer fc = dense_layer_from_weights(64, 1, mobile_fc_weight, mobile_fc_bias);
        output = dense_forward(flat, &fc);
        tensor_free(flat);
        sigmoid_inplace(output);
    }
#else
    /* ============================================================
     * DUMMY WEIGHTS PATH
     * =========================================================== */
    dw1_out = apply_dw_bn_relu(input, 5, 1, 10);
    tensor_free(input);
    pw1_out = apply_pw_bn_relu_pool(dw1_out, 32, 20);
    tensor_free(dw1_out);

    pool1 = pw1_out;

    /* SE block: modifies pool1 in-place */
    apply_se_block_dummy(pool1, 32, 60);

    dw2_out = apply_dw_bn_relu(pool1, 5, 1, 30);
    tensor_free(pool1);
    pw2_out = apply_pw_bn_relu_pool(dw2_out, 64, 40);
    tensor_free(dw2_out);

    pool2  = pw2_out;
    flat   = flatten_forward(pool2);
    tensor_free(pool2);
    output = apply_fc_threshold(flat, 50);
    tensor_free(flat);
#endif /* DEEPBINDI_MOBILE_REAL_WEIGHTS */

    return output;
}
