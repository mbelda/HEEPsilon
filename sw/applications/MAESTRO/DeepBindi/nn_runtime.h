/**
 * nn_runtime.h  --  Minimal int32 inference runtime for CNN_1D_v2 (DeepBindi).
 *                   X-HEEP / STATIC VERSION: no heap, no float, no math.h.
 *
 * DATA LAYOUT
 * -----------
 * All tensors use NCHW layout (batch x channels x height x width).
 * 1-D signals are stored as (N, C, 1, W); same conv2d_forward handles both.
 * Index: data[((n*C + c)*H + h)*W + w]
 *
 * ARITHMETIC
 * ----------
 * All weights, activations, and layer parameters are int32_t.
 * BatchNorm parameters are pre-folded at layer creation into:
 *   scale[c]  Q7 fixed-point: scale[c] = round(gamma[c]/sqrt(var[c]+eps) * 128)
 *   offset[c] = round(beta[c] - gamma[c]*mean[c]/sqrt(var[c]+eps))
 * Forward pass: y = (int32_t)(((int64_t)x * scale[c]) >> 7) + offset[c]
 * Sigmoid is replaced by a sign threshold: output = (x > 0) ? 1 : 0
 * ReLU: output = (x < 0) ? 0 : x  (unchanged logic)
 *
 * STATIC MEMORY MODEL
 * -------------------
 * tensor_create()   allocates from g_act_arena (arena.c).
 * tensor_free()     is a no-op; memory is reclaimed by act_arena_reset().
 * *_layer_create()  allocates from g_weight_pool (arena.c).
 * *_layer_free()    is a no-op; weights persist for the program lifetime.
 *
 * CGRA ACCELERATION
 * -----------------
 * conv2d_forward()            PRIMARY target: innermost (kh, kw) MAC loops.
 * dense_forward()             SECONDARY target: dot-product over in_features.
 * batchnorm_forward_inplace() FUSIBLE with preceding conv output stage.
 */

#ifndef NN_RUNTIME_H
#define NN_RUNTIME_H

#include <stdint.h>

/* ---- Tensor ------------------------------------------------------------ */

typedef struct {
    int      n;     /* batch size                               */
    int      c;     /* number of channels                       */
    int      h;     /* spatial height (1 for 1-D signals)       */
    int      w;     /* spatial width  (sequence length for 1-D) */
    int32_t *data;  /* contiguous int32_t[n*c*h*w] in NCHW order */
} Tensor;

/* ---- Conv2DLayer ------------------------------------------------------- */

typedef struct {
    int      in_channels;
    int      out_channels;
    int      kernel_h;
    int      kernel_w;
    int      stride_h;
    int      stride_w;
    int      pad_h;
    int      pad_w;
    int      groups;     /* 1 = standard; in_channels = depthwise */
    int32_t *weights;    /* [out_ch][in_ch/groups][kh][kw], row-major */
    int32_t *bias;       /* [out_ch] */
} Conv2DLayer;

/* ---- BatchNormLayer ---------------------------------------------------- */
/*
 * Parameters are pre-folded at layer creation:
 *   scale[c]  = round(gamma[c] / sqrt(var[c] + eps) * 128)   Q7 fixed-point
 *   offset[c] = round(beta[c]  - gamma[c]*mean[c]/sqrt(var[c]+eps))
 * Forward: y = (int32_t)(((int64_t)x * scale[c]) >> 7) + offset[c]
 * Dummy values: scale=128 (identity), offset=0.
 */
typedef struct {
    int      num_features;
    int32_t *scale;   /* Q7: 128 = 1.0 */
    int32_t *offset;  /* additive shift after the Q7 multiply */
} BatchNormLayer;

/* ---- DenseLayer -------------------------------------------------------- */

typedef struct {
    int      in_features;
    int      out_features;
    int32_t *weights; /* [out_features][in_features], row-major */
    int32_t *bias;    /* [out_features] */
} DenseLayer;

/* ---- Tensor helpers ---------------------------------------------------- */

Tensor  *tensor_create(int n, int c, int h, int w);
Tensor  *tensor_clone(const Tensor *src);
void     tensor_free(Tensor *tensor);       /* no-op in static port */
int      tensor_numel(const Tensor *tensor);
void     tensor_fill_dummy(Tensor *tensor, int32_t scale, int seed);
int32_t  tensor_checksum(const Tensor *tensor);

/** Print shape + checksum + first max_values elements as integers. */
void tensor_print_values(const char *name, const Tensor *tensor, int max_values);

/* ---- Layer constructors (dummy / seeded) -------------------------------- */

Conv2DLayer conv2d_layer_create(
    int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups, int seed);
void conv2d_layer_free(Conv2DLayer *layer);

BatchNormLayer batchnorm_layer_create(int num_features, int seed);
void batchnorm_layer_free(BatchNormLayer *layer);

DenseLayer dense_layer_create(int in_features, int out_features, int seed);
void dense_layer_free(DenseLayer *layer);

/* ---- Layer constructors (real weights from const ROM) ------------------- */
/*
 * These variants set the weight/bias pointers directly to caller-supplied
 * const arrays -- no pool allocation occurs.  Use when trained weights are
 * available as const int32_t arrays in flash (.rodata).
 *
 * The caller must guarantee that the arrays remain valid for the lifetime of
 * the returned layer struct (trivially true for static const arrays).
 * Cast from const to non-const is intentional and safe: the forward-pass
 * code never writes through these pointers.
 */

Conv2DLayer conv2d_layer_from_weights(
    int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    const int32_t *w, const int32_t *b);

BatchNormLayer batchnorm_layer_from_params(
    int num_features,
    const int32_t *scale, const int32_t *offset);

DenseLayer dense_layer_from_weights(
    int in_features, int out_features,
    const int32_t *w, const int32_t *b);

/* ---- Forward-pass primitives ------------------------------------------ */

/** PRIMARY CGRA TARGET: 2-D (or 1-D) convolution with zero-padding. */
Tensor *conv2d_forward(const Tensor *input, const Conv2DLayer *layer);

/** Apply inference-mode batch norm in-place (CGRA fusion target). */
void batchnorm_forward_inplace(Tensor *input, const BatchNormLayer *layer);

/**
 * batchnorm_rshift_inplace  --  BN + post-shift fused in a single int64 pass.
 *
 * Equivalent to batchnorm_forward_inplace() followed by tensor_rshift_inplace(),
 * but avoids intermediate int32_t overflow when the raw BN output exceeds INT32_MAX.
 *
 * Formula: y = (((int64_t)x * scale[c]) >> 7 + offset[c]) >> post_shift
 *
 * The int64_t accumulator is held throughout; only the final right-shifted
 * result is cast to int32_t.  Use this for BN layers in MobileCNN models
 * where large BN scales (> ~300) can push the intermediate BN output above
 * INT32_MAX before the shift -- in particular BN_PW1 of both mobile models.
 *
 * In CNN_1D_v2 the BN intermediate stays below INT32_MAX, so the existing
 * two-step (batchnorm_forward_inplace + tensor_rshift_inplace) is fine there.
 */
void batchnorm_rshift_inplace(Tensor *input, const BatchNormLayer *layer, int post_shift);

/**
 * batchnorm_rshift_perchannel  --  BN + per-channel post-shift.
 *
 * Same as batchnorm_rshift_inplace() but the post-shift amount is specified
 * per output channel via shifts[c].  This is critical for depthwise BN layers
 * whose Q7 scale factors vary widely across channels (e.g. 308 to 1574): a
 * global shift calibrated for the worst channel destroys signal in all others.
 * Per-channel shifts let each channel use the minimum shift required to keep
 * its output in int32_t range while maximising retained dynamic range.
 *
 * Formula: y[c] = (((int64_t)x[c] * scale[c]) >> 7 + offset[c]) >> shifts[c]
 */
void batchnorm_rshift_perchannel(Tensor *input, const BatchNormLayer *layer,
                                  const int32_t *shifts);

/** 2-D max pooling (no padding). */
Tensor *maxpool2d_forward(const Tensor *input,
                           int kernel_h, int kernel_w,
                           int stride_h, int stride_w);

/** Reshape (N,C,H,W) -> (N, C*H*W, 1, 1). */
Tensor *flatten_forward(const Tensor *input);

/** SECONDARY CGRA TARGET: fully-connected layer y = x @ W^T + b. */
Tensor *dense_forward(const Tensor *input, const DenseLayer *layer);

/* ---- Activation functions (in-place) ---------------------------------- */

/** ReLU: clamp negative values to zero. */
void relu_inplace(Tensor *input);

/** Sigmoid replacement for binary classification: output = (x > 0) ? 1 : 0.
 *  Avoids expf() entirely; valid because final layer has one output neuron. */
void sigmoid_inplace(Tensor *input);

/**
 * tensor_rshift_inplace  --  arithmetic right-shift every element by `shift` bits.
 *
 * Used after BatchNorm in the real-weights path to prevent accumulator overflow
 * in the next convolution layer.  The shift preserves the sign (arithmetic >>).
 *
 *   REAL-WEIGHTS overflow analysis (input in int8 range, Q7 BN scales):
 *     After BN1 (scale ≤ 377, input ≤ 128, conv1 285 MACs): max ≈ 13.6M
 *       -> right-shift 9:  13.6M >> 9 ≈ 26.6K  (Conv2 max acc ≈ 540M OK)
 *     After BN2 (scale ≤ 334, conv2 160 MACs):  max ≈ 1.41B
 *       -> right-shift 14: 1.41B >> 14 ≈ 86K    (FC1  max acc ≈ 699M OK)
 *
 * In the dummy-weight path no overflow occurs (scale=128, tiny weights),
 * so this function is only called from the DEEPBINDI_REAL_WEIGHTS code path.
 */
void tensor_rshift_inplace(Tensor *input, int shift);

/* ---- SE (Squeeze-and-Excitation) primitives --------------------------------
 *
 * Used by MobileCNN-SE-1D (DEEPBINDI_MODEL == 2).
 * All three functions below are available regardless of model selection;
 * the linker discards them if not called.
 */

/**
 * globalavgpool_forward  --  global average pooling over spatial dims H and W.
 * Output: (N, C, 1, 1).  Uses int64_t accumulator to prevent overflow.
 */
Tensor *globalavgpool_forward(const Tensor *input);

/**
 * hardsigmoid_se_inplace  --  integer hard-sigmoid for SE excitation weights.
 * Maps x -> clip((x >> shift) + 64, 0, 128)  where 128 = 1.0 in Q7.
 * shift is calibrated at export time so max(|x|) >> shift <= 64.
 */
void hardsigmoid_se_inplace(Tensor *input, int shift);

/**
 * se_channel_scale_inplace  --  SE feature recalibration.
 * For each channel c: feat[c,h,w] = ((int64_t)feat[c,h,w] * se[c]) >> 7
 * feat:       (N, C, H, W) -- feature map
 * se_weights: (N, C, 1, 1) -- Q7 excitation weights from hardsigmoid_se_inplace
 */
void se_channel_scale_inplace(Tensor *feat, const Tensor *se_weights);

#endif /* NN_RUNTIME_H */
