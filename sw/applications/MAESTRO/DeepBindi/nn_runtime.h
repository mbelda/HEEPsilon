/**
 * nn_runtime.h  –  Minimal inference runtime for CNN models (DeepBindi / EPFL)
 *                  STATIC VERSION: no heap allocation (no malloc/free).
 *
 * This header is identical to the dynamic version's nn_runtime.h.
 * The difference is entirely in nn_runtime.c: tensor_create() and
 * layer_create() functions allocate from the static pools in arena.c
 * instead of calling malloc/calloc.  No API change is visible here,
 * which means cnn_models_c.c is also unchanged (except for one
 * act_arena_reset() call added at the start of each run_*() function).
 *
 * This header defines the data types and function prototypes that implement
 * the basic building blocks needed for forward-pass inference on the
 * architectures defined in cnn_models.py (translated from PyTorch / Keras to C).
 *
 * DATA LAYOUT
 * -----------
 * All tensors use the NCHW (batch × channels × height × width) layout, which is
 * the same convention used by PyTorch and matches the weight ordering in the
 * original Python code.  1-D signals are stored as (N, C, 1, W) tensors so that
 * the same Conv2D and MaxPool2D primitives handle both 1-D and 2-D cases.
 *
 * CGRA ACCELERATION NOTES
 * -----------------------
 * A Coarse-Grained Reconfigurable Array (CGRA) can accelerate the computation
 * by mapping the innermost loops of the compute-intensive operations onto its
 * functional-unit array.  The two most important entry points are:
 *
 *   1.  conv2d_forward()   – see nn_runtime.c
 *       Five nested loops (n, oc, oh, ow, ic, kh, kw).  The innermost two
 *       (kh × kw) iterate over the kernel window and perform multiply-accumulate
 *       (MAC) operations.  These are the primary CGRA targets.
 *
 *   2.  dense_forward()   – see nn_runtime.c
 *       A matrix-vector multiply: for each output neuron, a dot product over
 *       `in_features` elements.  Maps directly onto a MAC array.
 *
 *   3.  batchnorm_forward_inplace() – scale + shift per channel: easily fused
 *       with the previous conv2d kernel on a CGRA to avoid memory round-trips.
 *
 *   4.  maxpool2d_forward() – reduction over a small window; can be scheduled
 *       as a post-conv stage on the same CGRA configuration.
 *
 * To replace any primitive with a CGRA-accelerated version, implement the same
 * function signature declared below and link against your CGRA driver instead
 * of nn_runtime.c.
 */

#ifndef NN_RUNTIME_H
#define NN_RUNTIME_H

/* ── Tensor ─────────────────────────────────────────────────────────────────
 * Heap-allocated 4-D array in NCHW order.
 * data[((n*C + c)*H + h)*W + w]
 */
typedef struct {
    int n;       /* batch size */
    int c;       /* number of channels */
    int h;       /* spatial height (1 for 1-D signals) */
    int w;       /* spatial width  (sequence length for 1-D signals) */
    float *data; /* contiguous float array of size n*c*h*w */
} Tensor;

/* ── Conv2DLayer ─────────────────────────────────────────────────────────────
 * Parameters for a 2-D convolution (also used for 1-D convolutions modelled
 * as (1×kernel_w) convolutions, matching the TFLite deployment approach).
 * When groups == in_channels == out_channels the layer is depthwise-separable,
 * as used in the MobileNetV3 bottleneck blocks.
 *
 * CGRA note: the weight tensor layout mirrors PyTorch:
 *   weights[oc][ic][kh][kw]  with ic = in_channels / groups
 */
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int groups;  /* 1 = standard conv; in_channels = depthwise conv */
    float *weights;
    float *bias;
} Conv2DLayer;

/* ── BatchNormLayer ─────────────────────────────────────────────────────────
 * Inference-mode batch normalisation: output = gamma*(x - mean)/sqrt(var+eps) + beta
 * gamma, beta, mean and var are all per-channel vectors of length num_features.
 *
 * CGRA note: this is a cheap element-wise operation that can be fused with the
 * preceding convolution into a single pass over the output tensor.
 */
typedef struct {
    int num_features;
    float eps;    /* small constant added to variance for numerical stability */
    float *gamma; /* learnable scale */
    float *beta;  /* learnable shift */
    float *mean;  /* running mean  (frozen at inference time) */
    float *var;   /* running variance (frozen at inference time) */
} BatchNormLayer;

/* ── DenseLayer ─────────────────────────────────────────────────────────────
 * Fully-connected (linear) layer: output = input @ weights^T + bias
 * weights layout: [out_features][in_features]  (row-major)
 *
 * CGRA note: this is a matrix-vector multiply; maps directly onto a MAC array.
 */
typedef struct {
    int in_features;
    int out_features;
    float *weights; /* [out_features × in_features], row-major */
    float *bias;    /* [out_features] */
} DenseLayer;

/* ── Tensor helpers ─────────────────────────────────────────────────────── */

/** Allocate a zero-initialised tensor of shape (n, c, h, w). */
Tensor *tensor_create(int n, int c, int h, int w);

/** Deep-copy a tensor into a new allocation. */
Tensor *tensor_clone(const Tensor *src);

/** Free tensor data and the Tensor struct itself. */
void tensor_free(Tensor *tensor);

/** Return the total number of scalar elements (n*c*h*w). */
int tensor_numel(const Tensor *tensor);

/** Fill tensor with deterministic pseudo-random values (for dummy inputs / weights). */
void tensor_fill_dummy(Tensor *tensor, float scale, int seed);

/** Compute a weighted checksum of all elements (useful for quick output comparison). */
float tensor_checksum(const Tensor *tensor);

/** Print shape, checksum and up to max_values individual elements. */
void tensor_print_values(const char *name, const Tensor *tensor, int max_values);

/* ── Layer constructors / destructors ───────────────────────────────────── */

/**
 * Allocate a Conv2DLayer and populate weights + bias with deterministic
 * dummy values derived from 'seed'.  Replace with real weight loading for
 * actual inference.
 */
Conv2DLayer conv2d_layer_create(
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int groups,
    int seed
);
void conv2d_layer_free(Conv2DLayer *layer);

/** Allocate a BatchNormLayer with deterministic dummy parameters. */
BatchNormLayer batchnorm_layer_create(int num_features, float eps, int seed);
void batchnorm_layer_free(BatchNormLayer *layer);

/** Allocate a DenseLayer with deterministic dummy parameters. */
DenseLayer dense_layer_create(int in_features, int out_features, int seed);
void dense_layer_free(DenseLayer *layer);

/* ── Forward-pass primitives ────────────────────────────────────────────── */

/**
 * conv2d_forward  *** PRIMARY CGRA TARGET ***
 *
 * 2-D (or 1-D) convolution with padding and optional groups (depthwise).
 * Output shape: (N, out_channels, out_h, out_w) where
 *   out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1
 *   out_w = (in_w + 2*pad_w - kernel_w) / stride_w + 1
 *
 * Caller takes ownership of the returned tensor.
 */
Tensor *conv2d_forward(const Tensor *input, const Conv2DLayer *layer);

/**
 * batchnorm_forward_inplace
 *
 * Applies inference-mode batch norm to every element of 'input' in-place.
 * Per-channel parameters (gamma, beta, mean, var) are read from 'layer'.
 */
void batchnorm_forward_inplace(Tensor *input, const BatchNormLayer *layer);

/**
 * maxpool2d_forward
 *
 * 2-D max pooling with given kernel and stride (no padding).
 * Caller takes ownership of the returned tensor.
 */
Tensor *maxpool2d_forward(const Tensor *input, int kernel_h, int kernel_w, int stride_h, int stride_w);

/**
 * adaptive_avg_pool2d_forward
 *
 * Global (or partial) average pooling that outputs a spatial map of size
 * (out_h, out_w).  Used after the last conv block of MobileNetV3.
 * Caller takes ownership of the returned tensor.
 */
Tensor *adaptive_avg_pool2d_forward(const Tensor *input, int out_h, int out_w);

/** Flatten spatial dimensions: output shape is (N, C*H*W, 1, 1). */
Tensor *flatten_forward(const Tensor *input);

/**
 * dense_forward  *** SECONDARY CGRA TARGET ***
 *
 * Fully-connected layer: y = x @ W^T + b
 * Input is assumed to have been flattened to (N, in_features, 1, 1).
 * Output shape is (N, out_features, 1, 1).
 * Caller takes ownership of the returned tensor.
 */
Tensor *dense_forward(const Tensor *input, const DenseLayer *layer);

/**
 * concat_height
 *
 * Concatenate two tensors along the H dimension (torch.cat([a,b], dim=2)).
 * Tensors must share the same N, C and W dimensions.
 * Used in CNN_2D_v2 and CNN_2D_v3 to merge parallel conv branches.
 * Caller takes ownership of the returned tensor.
 */
Tensor *concat_height(const Tensor *a, const Tensor *b);

/**
 * add_forward
 *
 * Element-wise addition of two same-shape tensors.
 * Used for residual connections in MobileNetV3 bottleneck blocks.
 * Caller takes ownership of the returned tensor.
 */
Tensor *add_forward(const Tensor *a, const Tensor *b);

/**
 * channel_scale_forward
 *
 * Multiply each channel of 'input' by the corresponding scalar in 'scale'
 * (shape N×C×1×1).  Used in the Squeeze-and-Excitation (SE) blocks of
 * MobileNetV3 to re-weight channels based on global context.
 * Caller takes ownership of the returned tensor.
 */
Tensor *channel_scale_forward(const Tensor *input, const Tensor *scale);

/* ── Activation functions (all in-place) ───────────────────────────────── */

/** ReLU: max(0, x) */
void relu_inplace(Tensor *input);

/** Sigmoid: 1 / (1 + exp(-x)) – binary classification output */
void sigmoid_inplace(Tensor *input);

/** Softmax: stable exp-normalised distribution – multi-class output */
void softmax_inplace(Tensor *input);

/** Hard-Sigmoid: clip((x+3)/6, 0, 1) – used in SE blocks */
void hardsigmoid_inplace(Tensor *input);

/** Hard-Swish: x * clip((x+3)/6, 0, 1) – used in MobileNetV3 activations */
void hardswish_inplace(Tensor *input);

#endif
