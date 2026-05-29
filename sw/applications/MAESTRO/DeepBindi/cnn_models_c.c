/**
 * cnn_models_c.c  –  Forward-pass implementations of all CNN architectures.
 *                    STATIC VERSION: no malloc anywhere.
 *
 * This file is nearly identical to c_port/cnn_models_c.c.  The only two
 * differences are:
 *
 *   1.  #include "arena.h" is added so that act_arena_reset() is available.
 *
 *   2.  Each run_*() function begins with act_arena_reset(), which reclaims
 *       the activation scratch arena and tensor-struct pool from the previous
 *       model run before allocating new tensors.
 *       (Weight allocations are NOT reset; weights accumulate in g_weight_pool
 *       across the entire program run.)
 *
 * All layer computations, shapes, and numerical values are identical to the
 * dynamic port.  Running both versions and comparing tensor_checksum() output
 * is the recommended validation step for a CGRA implementation.
 *
 * Model index:
 *   run_cnn_2d_v1()               → CNN_2D_v1   (PYA)
 *   run_cnn_2d_v2()               → CNN_2D_v2   (PYB)
 *   run_cnn_2d_v3()               → CNN_2D_v3   (PYC)
 *   run_cnn_1d_v1()               → CNN_1D_v1   (PYD)
 *   run_cnn_1d_v2()               → CNN_1D_v2   (PYE)
 *   run_cnn_1d_v3()               → CNN_1D_v3   (PYF)
 *   run_mobilenet_v3_custom()     → MobileNetV3Custom  (PYG)
 *   run_cnn_1d_tensorflow_sigmoid() → CNN_1d_tensorflow_sigmoid  (TF1)
 *   run_cnn_1d_tensorflow_softmax() → CNN_1d_tensorflow_softmax  (TF2)
 *   run_cnn_2d_tensorflow_softmax() → CNN_2d_tensorflow_softmax  (TF3)
 */
#include <stdio.h>
#include <stddef.h>  /* for NULL */
#include "cnn_models_c.h"
#include "arena.h"   /* for act_arena_reset() */

typedef enum {
    ACT_NONE = 0,   /* no activation (e.g. projection conv in MobileNet) */
    ACT_RELU,       /* max(0, x) */
    ACT_SIGMOID,    /* 1/(1+exp(-x)) – binary classification output */
    ACT_SOFTMAX,    /* normalised exponential – multi-class output */
    ACT_HARDSIGMOID,/* clip((x+3)/6, 0, 1) – SE block gate */
    ACT_HARDSWISH   /* x * clip((x+3)/6, 0, 1) – MobileNetV3 main activation */
} ActivationKind;

/* Specification of one MobileNetV3 inverted-residual (bottleneck) block.
 * Computed from the Python InvertedResidualConfig with width_mult=0.35. */
typedef struct {
    int input_channels;
    int kernel;            /* depthwise kernel size (3 or 5) */
    int expanded_channels; /* width of the expansion stage */
    int out_channels;
    int use_se;            /* 1 = include a Squeeze-and-Excitation block */
    int use_hs;            /* 1 = HardSwish activation, 0 = ReLU */
    int stride;            /* depthwise stride (1 or 2) */
    int squeeze_channels;  /* SE bottleneck width (expanded_channels / 4, rounded) */
} MBConvSpec;

static void apply_activation(Tensor *tensor, ActivationKind activation) {
    switch (activation) {
        case ACT_NONE:
            break;
        case ACT_RELU:
            relu_inplace(tensor);
            break;
        case ACT_SIGMOID:
            sigmoid_inplace(tensor);
            break;
        case ACT_SOFTMAX:
            softmax_inplace(tensor);
            break;
        case ACT_HARDSIGMOID:
            hardsigmoid_inplace(tensor);
            break;
        case ACT_HARDSWISH:
            hardswish_inplace(tensor);
            break;
    }
}

static Tensor *make_dummy_input_1d(int channels, int length, int seed) {
    /* 1-D signals are stored as (1, channels, 1, length) tensors (NCHW). */
    Tensor *input = tensor_create(1, channels, 1, length);
    tensor_fill_dummy(input, 0.5f, seed);
    return input;
}

static Tensor *make_dummy_input_2d(int channels, int height, int width, int seed) {
    Tensor *input = tensor_create(1, channels, height, width);
    tensor_fill_dummy(input, 0.5f, seed);
    return input;
}

/*
 * apply_conv_bn_act  –  Most common fused pattern  *** KEY CGRA FUSION TARGET ***
 *
 *   Conv2D  →  BatchNorm  →  Activation
 *
 * In a CGRA implementation these three stages can share a single pass over the
 * output buffer: compute the conv sum, immediately apply the BN scale+shift,
 * then apply the non-linearity before writing to memory.  This avoids two
 * extra read/write cycles per output element.
 *
 * Parameters map directly to the Python layer constructors:
 *   kernel_h, kernel_w  – filter spatial dimensions
 *   groups              – 1 for standard conv, in_channels for depthwise
 *   bn_eps              – 1e-5 (PyTorch default) or 1e-3 (MobileNetV3 BN)
 */
static Tensor *apply_conv_bn_act(
    const Tensor *input,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int groups,
    float bn_eps,
    ActivationKind activation,
    int seed
) {
    Conv2DLayer conv = conv2d_layer_create(
        input->c, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, groups, seed
    );
    BatchNormLayer bn = batchnorm_layer_create(out_channels, bn_eps, seed + 1);
    Tensor *output = conv2d_forward(input, &conv);
    batchnorm_forward_inplace(output, &bn);
    apply_activation(output, activation);
    conv2d_layer_free(&conv);
    batchnorm_layer_free(&bn);
    return output;
}

/*
 * apply_dense_bn_act  –  Fully-connected block helper
 *   Dense  →  (optional BatchNorm)  →  Activation
 *
 * use_bn == 0 for the final classification head; == 1 for hidden FC layers.
 */
static Tensor *apply_dense_bn_act(
    const Tensor *input,
    int out_features,
    int use_bn,
    float bn_eps,
    ActivationKind activation,
    int seed
) {
    DenseLayer dense = dense_layer_create(input->c, out_features, seed);
    Tensor *output = dense_forward(input, &dense);
    if (use_bn) {
        BatchNormLayer bn = batchnorm_layer_create(out_features, bn_eps, seed + 1);
        batchnorm_forward_inplace(output, &bn);
        batchnorm_layer_free(&bn);
    }
    apply_activation(output, activation);
    dense_layer_free(&dense);
    return output;
}

/*
 * apply_se_block  –  Squeeze-and-Excitation block (MobileNetV3)
 *
 * Computes channel attention weights and scales the input feature map:
 *   1. Global average pool:   (N,C,H,W) → (N,C,1,1)
 *   2. FC squeeze:            (N,C,1,1) → (N,squeeze_channels,1,1)  + ReLU
 *   3. FC excitation:         (N,squeeze_channels,1,1) → (N,C,1,1)  + HardSigmoid
 *   4. Channel-wise multiply: output = input * attention_weights
 */
static Tensor *apply_se_block(const Tensor *input, int squeeze_channels, int seed) {
    /* Steps 2 and 3 are implemented as 1×1 convolutions (pointwise). */
    Conv2DLayer fc1 = conv2d_layer_create(input->c, squeeze_channels, 1, 1, 1, 1, 0, 0, 1, seed);
    Conv2DLayer fc2 = conv2d_layer_create(squeeze_channels, input->c, 1, 1, 1, 1, 0, 0, 1, seed + 3);
    Tensor *pooled = adaptive_avg_pool2d_forward(input, 1, 1); /* global avg pool */
    Tensor *hidden = conv2d_forward(pooled, &fc1);
    Tensor *scale_logits;
    Tensor *scaled;

    relu_inplace(hidden);
    scale_logits = conv2d_forward(hidden, &fc2);
    hardsigmoid_inplace(scale_logits);             /* gate values in [0, 1] */
    scaled = channel_scale_forward(input, scale_logits); /* re-weight channels */

    tensor_free(pooled);
    tensor_free(hidden);
    tensor_free(scale_logits);
    conv2d_layer_free(&fc1);
    conv2d_layer_free(&fc2);
    return scaled;
}

/*
 * apply_mobilenet_block  –  MobileNetV3 Inverted Residual block
 *
 * Structure (from the paper, Table 2):
 *   [Expansion]  pointwise conv 1×1  (if expanded_channels != input_channels)
 *   [Depthwise]  depthwise conv k×k  (groups == expanded_channels)
 *   [SE]         squeeze-and-excitation block (optional)
 *   [Projection] pointwise conv 1×1 → out_channels,  NO activation
 *   [Residual]   add input if stride == 1 and channels unchanged
 *
 * The depthwise conv is the most important CGRA target within this block:
 * it has groups == expanded_channels, so in_per_group == 1 – the innermost
 * channel loop collapses, leaving a pure 2-D (kh×kw) MAC per output position.
 */
static Tensor *apply_mobilenet_block(const Tensor *input, MBConvSpec spec, int seed) {
    ActivationKind activation = spec.use_hs ? ACT_HARDSWISH : ACT_RELU;
    Tensor *expanded = NULL;
    const Tensor *stage_input = input;
    Tensor *depthwise;
    Tensor *projected;

    if (spec.expanded_channels != spec.input_channels) {
        expanded = apply_conv_bn_act(input, spec.expanded_channels, 1, 1, 1, 1, 0, 0, 1, 0.001f, activation, seed);
        stage_input = expanded;
    }

    depthwise = apply_conv_bn_act(
        stage_input,
        spec.expanded_channels,
        spec.kernel,
        spec.kernel,
        spec.stride,
        spec.stride,
        spec.kernel / 2,
        spec.kernel / 2,
        spec.expanded_channels,
        0.001f,
        activation,
        seed + 10
    );

    if (expanded != NULL) {
        tensor_free(expanded);
    }

    if (spec.use_se) {
        Tensor *scaled = apply_se_block(depthwise, spec.squeeze_channels, seed + 20);
        tensor_free(depthwise);
        depthwise = scaled;
    }

    projected = apply_conv_bn_act(depthwise, spec.out_channels, 1, 1, 1, 1, 0, 0, 1, 0.001f, ACT_NONE, seed + 30);
    tensor_free(depthwise);

    if (spec.stride == 1 && spec.input_channels == spec.out_channels) {
        Tensor *residual = add_forward(projected, input);
        tensor_free(projected);
        projected = residual;
    }
    return projected;
}

/* Utility from torchvision: round channel count to the nearest multiple of
 * divisor (always >= divisor) ensuring no more than 10% reduction. */
static int make_divisible(float value, int divisor) {
    int new_value = (int)(value + divisor / 2.0f) / divisor * divisor;
    int minimum = divisor;
    if (new_value < minimum) {
        new_value = minimum;
    }
    if ((float)new_value < 0.9f * value) {
        new_value += divisor;
    }
    return new_value;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYA: CNN_2D_v1  –  2D-CNN  (2 conv + 2 FC)
 *
 * Input:  (1, 1, 57, 57)  –  single-channel 57×57 feature map
 * Layer 1: Conv2d(1→16, k=5, pad=1) → BN → ReLU → MaxPool(2×1, stride 2×1)
 * Layer 2: Conv2d(16→32, k=5, pad=1) → BN → ReLU → MaxPool(2×2, stride 2×2)
 * Flatten → Dense(→32) → BN → ReLU → Dense(→1) → Sigmoid
 * Output:  (1, 1, 1, 1)  –  binary classification score
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_2d_v1(void) {
    act_arena_reset(); /* reclaim scratch arena from previous model */
    Tensor *input = make_dummy_input_2d(1, 57, 57, 1);
    Tensor *x = apply_conv_bn_act(input, 16, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 10);
    Tensor *pooled = maxpool2d_forward(x, 2, 1, 2, 1);
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    tensor_free(x);
    x = pooled;

    pooled = apply_conv_bn_act(x, 32, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 20);
    tensor_free(x);
    x = maxpool2d_forward(pooled, 2, 2, 2, 2);
    tensor_free(pooled);

    flat = flatten_forward(x);
    tensor_free(x);
    x = apply_dense_bn_act(flat, 32, 1, 1e-5f, ACT_RELU, 30);
    tensor_free(flat);
    output = apply_dense_bn_act(x, 1, 0, 0.0f, ACT_SIGMOID, 40);
    tensor_free(x);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYB: CNN_2D_v2  –  2D-CNN with parallel multi-scale branches (2 conv + 2 FC)
 *
 * Three parallel branches on the same input capture features at different
 * vertical scales (5×5, 28×5, 57×5), then concatenate along H before conv2:
 *
 *   input (1,1,57,57) ─┬── Conv1_1 (5×5) → BN → ReLU → MaxPool(2×1) → x1
 *                      ├── Conv1_2 (28×5) → BN → ReLU → MaxPool(1×1) → x2
 *                      └── Conv1_3 (57×5) → BN → ReLU → MaxPool(1×1) → x3
 *                                    concat([x1, x2, x3], dim=H)
 *   → Conv2(5×5,16→32) → BN → ReLU → MaxPool(2×2)
 *   → Flatten → Dense(→32) → BN → ReLU → Dense(→1) → Sigmoid
 *
 * CGRA note: the three branch convolutions are data-independent and can be
 * scheduled in parallel on the CGRA array.
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_2d_v2(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_2d(1, 57, 57, 2);
    Tensor *x1 = apply_conv_bn_act(input, 16, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 50);
    Tensor *x2 = apply_conv_bn_act(input, 16, 28, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 60);
    Tensor *x3 = apply_conv_bn_act(input, 16, 57, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 70);
    Tensor *p1 = maxpool2d_forward(x1, 2, 1, 2, 1);
    Tensor *p2 = maxpool2d_forward(x2, 1, 1, 1, 1);
    Tensor *p3 = maxpool2d_forward(x3, 1, 1, 1, 1);
    Tensor *cat12;
    Tensor *cat123;
    Tensor *x;
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    tensor_free(x1);
    tensor_free(x2);
    tensor_free(x3);

    /* Merge branches by stacking along the height dimension */
    cat12 = concat_height(p1, p2);
    cat123 = concat_height(cat12, p3);
    tensor_free(p1);
    tensor_free(p2);
    tensor_free(p3);
    tensor_free(cat12);

    x = apply_conv_bn_act(cat123, 32, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 80);
    tensor_free(cat123);
    flat = maxpool2d_forward(x, 2, 2, 2, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    flat = apply_dense_bn_act(x, 32, 1, 1e-5f, ACT_RELU, 90);
    tensor_free(x);
    output = apply_dense_bn_act(flat, 1, 0, 0.0f, ACT_SIGMOID, 100);
    tensor_free(flat);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYC: CNN_2D_v3  –  2D-CNN with one extra branch (2 conv + 2 FC)
 *
 * Like CNN_2D_v2 but without the 28×5 intermediate branch:
 *   input ─┬── Conv1_1 (5×5) → MaxPool(2×1) → x1
 *           └── Conv1_3 (57×5) → MaxPool(1×1) → x3
 *                         concat([x1, x3], dim=H)
 *   → Conv2 → MaxPool → Flatten → Dense(→32) → Dense(→1) → Sigmoid
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_2d_v3(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_2d(1, 57, 57, 3);
    Tensor *x1 = apply_conv_bn_act(input, 16, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 110);
    Tensor *x3 = apply_conv_bn_act(input, 16, 57, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 120);
    Tensor *p1 = maxpool2d_forward(x1, 2, 1, 2, 1);
    Tensor *p3 = maxpool2d_forward(x3, 1, 1, 1, 1);
    Tensor *cat;
    Tensor *x;
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    tensor_free(x1);
    tensor_free(x3);

    cat = concat_height(p1, p3);
    tensor_free(p1);
    tensor_free(p3);

    x = apply_conv_bn_act(cat, 32, 5, 5, 1, 1, 1, 1, 1, 1e-5f, ACT_RELU, 130);
    tensor_free(cat);
    flat = maxpool2d_forward(x, 2, 2, 2, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    flat = apply_dense_bn_act(x, 32, 1, 1e-5f, ACT_RELU, 140);
    tensor_free(x);
    output = apply_dense_bn_act(flat, 1, 0, 0.0f, ACT_SIGMOID, 150);
    tensor_free(flat);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYD: CNN_1D_v1  –  1D-CNN  (1 conv + 1 FC)
 *
 * Input:  (1, 57, 1, 10)  –  57 features × 10 time steps
 * 1-D convolution is represented as a 2-D convolution with kernel (1×5):
 *   Conv1d(57→64, k=5, pad=1) → BN → ReLU → MaxPool1d(2) → Flatten
 *   → Dense(→1) → Sigmoid
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_1d_v1(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_1d(57, 10, 4);
    Tensor *x = apply_conv_bn_act(input, 64, 1, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 160);
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    output = apply_dense_bn_act(x, 1, 0, 0.0f, ACT_SIGMOID, 170);
    tensor_free(x);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYE: CNN_1D_v2  –  1D-CNN with two conv blocks (2 conv + 1 FC)
 *
 *   Conv1d(57→32, k=5) → BN → ReLU → MaxPool1d(2)
 *   → Conv1d(32→64, k=5) → BN → ReLU → MaxPool1d(2)
 *   → Flatten → Dense(→1) → Sigmoid
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_1d_v2(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_1d(57, 10, 5);
    Tensor *x = apply_conv_bn_act(input, 32, 1, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 180);
    Tensor *pooled = maxpool2d_forward(x, 1, 2, 1, 2);
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    tensor_free(x);
    x = apply_conv_bn_act(pooled, 64, 1, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 190);
    tensor_free(pooled);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    output = apply_dense_bn_act(x, 1, 0, 0.0f, ACT_SIGMOID, 200);
    tensor_free(x);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYF: CNN_1D_v3  –  1D-CNN with two FC layers (1 conv + 2 FC)
 *
 *   Conv1d(57→64, k=5) → BN → ReLU → MaxPool1d(2) → Flatten
 *   → Dense(→32) → BN → ReLU → Dense(→1) → Sigmoid
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_1d_v3(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_1d(57, 10, 6);
    Tensor *x = apply_conv_bn_act(input, 64, 1, 5, 1, 1, 0, 1, 1, 1e-5f, ACT_RELU, 210);
    Tensor *flat;
    Tensor *hidden;
    Tensor *output;

    tensor_free(input);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    hidden = apply_dense_bn_act(x, 32, 1, 1e-5f, ACT_RELU, 220);
    tensor_free(x);
    output = apply_dense_bn_act(hidden, 1, 0, 0.0f, ACT_SIGMOID, 230);
    tensor_free(hidden);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * PYG: MobileNetV3Custom  –  MobileNetV3-Small (width_mult=0.35)
 *
 * This is the most complex architecture in the set.  Adapted for single-channel
 * input (1-channel vs. the original 3-channel ImageNet model) and with a custom
 * lightweight classifier head [192 → 64 → 1] ending in Sigmoid.
 *
 * Global structure:
 *   Stem: Conv2d(1→8, 3×3, stride=2) → BN → HardSwish
 *   11 × MobileNetV3 Inverted Residual blocks (see apply_mobilenet_block)
 *   Last conv: Conv2d(→192, 1×1) → BN → HardSwish
 *   Global avg pool → Flatten
 *   Dense(192→64) → HardSwish
 *   Dense(64→1) → Sigmoid
 *
 * Channel widths at width_mult=0.35 (all rounded to nearest multiple of 8):
 *   first_channels = 8, last_conv_channels = 192
 *
 * Block specs (input_ch, kernel, expand_ch, out_ch, SE, HS, stride):
 *   [0]  8,3,  8,  8, SE, RE, s=2   [1]  8,3, 24,  8, --, RE, s=2
 *   [2]  8,3, 32,  8, --, RE, s=1   [3]  8,5, 32, 16, SE, HS, s=2
 *   [4] 16,5, 88, 16, SE, HS, s=1   [5] 16,5, 88, 16, SE, HS, s=1
 *   [6] 16,5, 40, 16, SE, HS, s=1   [7] 16,5, 48, 16, SE, HS, s=1
 *   [8] 16,5,104, 32, SE, HS, s=2   [9] 32,5,200, 32, SE, HS, s=1
 *  [10] 32,5,200, 32, SE, HS, s=1
 *
 * CGRA note: blocks [4]–[10] include SE sub-blocks; block [0] is the only
 * one without expansion (expand == input).
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_mobilenet_v3_custom(void) {
    act_arena_reset();
    const float width_mult = 0.35f;
    const int first_channels = make_divisible(16.0f * width_mult, 8);
    const int last_conv_channels = 6 * make_divisible(96.0f * width_mult, 8);
    /*
     * Base config table: {in_ch, kernel, expand_ch, out_ch, use_se, use_hs, stride, unused}
     * All channel counts are scaled by width_mult and rounded before use.
     */
    const int base_cfg[11][8] = {
        {16, 3,  16, 16, 1, 0, 2, 0},  /* block 0: SE, RE, stride 2 */
        {16, 3,  72, 24, 0, 0, 2, 0},  /* block 1: no SE, RE, stride 2 */
        {24, 3,  88, 24, 0, 0, 1, 0},  /* block 2: no SE, RE, stride 1 */
        {24, 5,  96, 40, 1, 1, 2, 0},  /* block 3: SE, HS, stride 2 */
        {40, 5, 240, 40, 1, 1, 1, 0},  /* block 4: SE, HS */
        {40, 5, 240, 40, 1, 1, 1, 0},  /* block 5: SE, HS */
        {40, 5, 120, 48, 1, 1, 1, 0},  /* block 6: SE, HS */
        {48, 5, 144, 48, 1, 1, 1, 0},  /* block 7: SE, HS */
        {48, 5, 288, 96, 1, 1, 2, 0},  /* block 8: SE, HS, stride 2 */
        {96, 5, 576, 96, 1, 1, 1, 0},  /* block 9: SE, HS */
        {96, 5, 576, 96, 1, 1, 1, 0}   /* block 10: SE, HS */
    };
    MBConvSpec specs[11];
    Tensor *input = make_dummy_input_2d(1, 57, 57, 7);
    Tensor *x;
    Tensor *pooled;
    Tensor *flat;
    Tensor *hidden;
    Tensor *output;
    int i;

    /* Scale all channel counts with width_mult and round to multiples of 8 */
    for (i = 0; i < 11; ++i) {
        specs[i].input_channels    = make_divisible(base_cfg[i][0] * width_mult, 8);
        specs[i].kernel            = base_cfg[i][1];
        specs[i].expanded_channels = make_divisible(base_cfg[i][2] * width_mult, 8);
        specs[i].out_channels      = make_divisible(base_cfg[i][3] * width_mult, 8);
        specs[i].use_se            = base_cfg[i][4];
        specs[i].use_hs            = base_cfg[i][5];
        specs[i].stride            = base_cfg[i][6];
        specs[i].squeeze_channels  = make_divisible((float)(specs[i].expanded_channels / 4), 8);
    }

    /* Stem convolution (original first_conv adapted to 1 input channel) */
    x = apply_conv_bn_act(input, first_channels, 3, 3, 2, 2, 1, 1, 1, 0.001f, ACT_HARDSWISH, 240);
    tensor_free(input);

    /* 11 inverted residual blocks (each potentially with SE + residual add) */
    for (i = 0; i < 11; ++i) {
        Tensor *next = apply_mobilenet_block(x, specs[i], 260 + i * 20);
        tensor_free(x);
        x = next;
    }

    /* Last pointwise conv to 192 channels */
    pooled = apply_conv_bn_act(x, last_conv_channels, 1, 1, 1, 1, 0, 0, 1, 0.001f, ACT_HARDSWISH, 520);
    tensor_free(x);
    x = adaptive_avg_pool2d_forward(pooled, 1, 1); /* global average pool */
    tensor_free(pooled);
    flat = flatten_forward(x);
    tensor_free(x);
    /* Custom classifier head: 192 → 64 → 1 */
    hidden = apply_dense_bn_act(flat, 64, 0, 0.0f, ACT_HARDSWISH, 530);
    tensor_free(flat);
    output = apply_dense_bn_act(hidden, 1, 0, 0.0f, ACT_SIGMOID, 540);
    tensor_free(hidden);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * TF1: CNN_1d_tensorflow_sigmoid  –  Keras 1D-CNN for TFLite deployment
 *
 * Identical compute graph to CNN_1D_v3 (PYF) but with:
 *   • Conv1D padding="valid" (no padding, hence pad_w=0 here)
 *   • BN eps=1e-3 (Keras default vs PyTorch's 1e-5)
 *   • Final activation: Sigmoid (binary classification)
 *
 *   Conv1d(57→64, k=5, no pad) → BN → ReLU → MaxPool(2)
 *   → Flatten → Dense(→32) → BN → ReLU → Dense(→1) → Sigmoid
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_1d_tensorflow_sigmoid(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_1d(57, 10, 8);
    Tensor *x = apply_conv_bn_act(input, 64, 1, 5, 1, 1, 0, 0, 1, 1e-3f, ACT_RELU, 550);
    Tensor *flat;
    Tensor *hidden;
    Tensor *output;

    tensor_free(input);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    hidden = apply_dense_bn_act(x, 32, 1, 1e-3f, ACT_RELU, 560);
    tensor_free(x);
    output = apply_dense_bn_act(hidden, 1, 0, 0.0f, ACT_SIGMOID, 570);
    tensor_free(hidden);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * TF2: CNN_1d_tensorflow_softmax  –  Keras 1D-CNN for TFLite deployment
 *
 * Same as TF1 but final layer outputs 2 classes with Softmax.
 * Used with SparseCategoricalCrossentropy loss in training.
 *   → Dense(→2) → Softmax
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_1d_tensorflow_softmax(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_1d(57, 10, 9);
    Tensor *x = apply_conv_bn_act(input, 64, 1, 5, 1, 1, 0, 0, 1, 1e-3f, ACT_RELU, 580);
    Tensor *flat;
    Tensor *hidden;
    Tensor *output;

    tensor_free(input);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    hidden = apply_dense_bn_act(x, 32, 1, 1e-3f, ACT_RELU, 590);
    tensor_free(x);
    output = apply_dense_bn_act(hidden, 2, 0, 0.0f, ACT_SOFTMAX, 600);
    tensor_free(hidden);
    return output;
}

/* ──────────────────────────────────────────────────────────────────────────
 * TF3: CNN_2d_tensorflow_softmax  –  Keras 2D-CNN for microcontroller deploy
 *
 * Uses 2D convolution with kernel (1×5) to emulate 1D convolution, which
 * allows deployment on microcontrollers where only Conv2D is supported by the
 * TFLite runtime.  No hidden FC layer (simpler head):
 *
 *   Conv2d(57→64, 1×5, no pad) → BN → ReLU → MaxPool2d(1×2)
 *   → Flatten → Dense(→2) → Softmax
 *
 * Input shape: (1, 57, 10, 10) – the two spatial dims are time × features.
 * ────────────────────────────────────────────────────────────────────────── */
Tensor *run_cnn_2d_tensorflow_softmax(void) {
    act_arena_reset();
    Tensor *input = make_dummy_input_2d(1, 57, 10, 10);
    Tensor *x = apply_conv_bn_act(input, 64, 1, 5, 1, 1, 0, 0, 1, 1e-3f, ACT_RELU, 610);
    Tensor *flat;
    Tensor *output;

    tensor_free(input);
    flat = maxpool2d_forward(x, 1, 2, 1, 2);
    tensor_free(x);
    x = flatten_forward(flat);
    tensor_free(flat);
    output = apply_dense_bn_act(x, 2, 0, 0.0f, ACT_SOFTMAX, 620);
    tensor_free(x);
    return output;
}
