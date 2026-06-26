/**
 * nn_runtime.c  --  int32 inference primitives for CNN_1D_v2.
 *                   X-HEEP / STATIC VERSION.
 *
 * KEY PROPERTIES:
 *   - All weights and activations are int32_t; no float anywhere.
 *   - No math.h: no expf, sqrtf, fabsf.
 *   - No memset / memcpy: zero-fill and copy use explicit scalar loops.
 *   - No %f in any printf call.
 *   - BatchNorm uses pre-folded Q7 scale+offset (no per-element sqrt).
 *   - Sigmoid replaced by sign threshold: output = (x > 0) ? 1 : 0.
 *
 * OVERFLOW ANALYSIS (dummy weights, int8-range inputs):
 *   Conv1 accumulator: 57*5 MACs * (127 * 8) max per MAC = ~289K  < INT32_MAX OK
 *   Conv2 accumulator: 32*5 MACs * (289K * 8) max per MAC = ~370M < INT32_MAX OK
 *   BatchNorm Q7 multiply: uses int64_t intermediate to avoid overflow.
 *
 * CGRA ACCELERATION:
 *   conv2d_forward()            -- innermost (kh, kw) are MAC chains.
 *   dense_forward()             -- matrix-vector multiply.
 *   batchnorm_forward_inplace() -- fuse with conv output stage.
 */
#include "deepbindi_config.h"
#include "nn_runtime.h"
#include "arena.h"

/* ---- Internal helpers -------------------------------------------------- */

static int tensor_index(const Tensor *t, int n, int c, int h, int w) {
    return (((n * t->c + c) * t->h + h) * t->w + w);
}

/* Deterministic pseudo-random seed for dummy weight generation (int32 range). */
static int32_t seeded_value_int32(int index, int seed, int range) {
    int raw = ((index + 1) * (seed + 3) * 17) % 257;  /* raw in [0, 256] */
    return (int32_t)((raw - 128) * range / 128);       /* map to [-range, +range] */
}

/* ---- Tensor helpers ---------------------------------------------------- */

Tensor *tensor_create(int n, int c, int h, int w) {
    int count = n * c * h * w;
    Tensor *tensor;
    if (g_tensor_top >= MAX_LIVE_TENSORS) {
        DEEPBINDI_LOG_ERROR(
            "ERROR: tensor pool exhausted (top=%d max=%d)\r\n",
            g_tensor_top, MAX_LIVE_TENSORS);
        DEEPBINDI_FATAL("tensor pool exhausted");
    }
    tensor = &g_tensor_pool[g_tensor_top++];
    tensor->n    = n;
    tensor->c    = c;
    tensor->h    = h;
    tensor->w    = w;
    tensor->data = act_alloc(count);
    return tensor;
}

Tensor *tensor_clone(const Tensor *src) {
    int i;
    int total = tensor_numel(src);
    Tensor *dst = tensor_create(src->n, src->c, src->h, src->w);
    for (i = 0; i < total; ++i) {
        dst->data[i] = src->data[i];
    }
    return dst;
}

void tensor_free(Tensor *tensor) {
    (void)tensor; /* no-op: memory reclaimed by act_arena_reset() */
}

int tensor_numel(const Tensor *tensor) {
    return tensor->n * tensor->c * tensor->h * tensor->w;
}

void tensor_fill_dummy(Tensor *tensor, int32_t scale, int seed) {
    int i;
    int total = tensor_numel(tensor);
    for (i = 0; i < total; ++i) {
        tensor->data[i] = seeded_value_int32(i, seed, (int)scale);
    }
}

int32_t tensor_checksum(const Tensor *tensor) {
    int32_t acc = 0;
    int i;
    int total = tensor_numel(tensor);
    for (i = 0; i < total; ++i) {
        acc += tensor->data[i] * (int32_t)((i % 7) + 1);
    }
    return acc;
}

void tensor_print_values(const char *name, const Tensor *tensor, int max_values) {
#ifdef DEEPBINDI_ENABLE_LOGGING
    int i;
    int total = tensor_numel(tensor);
    int limit = total < max_values ? total : max_values;
    DEEPBINDI_PRINTF("%s shape=(%d,%d,%d,%d) csum=%d vals=[",
           name, tensor->n, tensor->c, tensor->h, tensor->w,
           (int)tensor_checksum(tensor));
    for (i = 0; i < limit; ++i) {
        DEEPBINDI_PRINTF("%s%d", i == 0 ? "" : ",", (int)tensor->data[i]);
    }
    if (limit < total) {
        DEEPBINDI_PRINTF(",...");
    }
    DEEPBINDI_PRINTF("]\r\n");
#else
    (void)name; (void)tensor; (void)max_values;
#endif
}

/* ---- Layer constructors ------------------------------------------------ */

Conv2DLayer conv2d_layer_create(
    int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups, int seed)
{
    Conv2DLayer layer;
    int i;
    int in_per_group  = in_channels / groups;
    int weight_count  = out_channels * in_per_group * kernel_h * kernel_w;

    layer.in_channels  = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_h     = kernel_h;
    layer.kernel_w     = kernel_w;
    layer.stride_h     = stride_h;
    layer.stride_w     = stride_w;
    layer.pad_h        = pad_h;
    layer.pad_w        = pad_w;
    layer.groups       = groups;

    layer.weights = weight_alloc(weight_count);
    layer.bias    = weight_alloc(out_channels);

    /* Dummy weights: int8 range [-8, 8]. */
    for (i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value_int32(i, seed, 8);
    }
    /* Dummy biases: slightly larger range [-100, 100]. */
    for (i = 0; i < out_channels; ++i) {
        layer.bias[i] = seeded_value_int32(i, seed + 11, 100);
    }
    return layer;
}

void conv2d_layer_free(Conv2DLayer *layer) {
    (void)layer; /* no-op */
}

/*
 * batchnorm_layer_create
 *
 * Dummy mode: scale=128 (Q7 identity: scale/128 = 1.0), offset=0.
 * With real trained weights: pre-fold gamma/mean/var/beta here to avoid
 * per-sample sqrt in the forward pass.
 */
BatchNormLayer batchnorm_layer_create(int num_features, int seed) {
    BatchNormLayer layer;
    int i;
    (void)seed; /* seed unused for dummy (identity); keep for API consistency */
    layer.num_features = num_features;
    layer.scale  = weight_alloc(num_features);
    layer.offset = weight_alloc(num_features);
    for (i = 0; i < num_features; ++i) {
        layer.scale[i]  = 128; /* Q7: 1.0 -- identity pass-through */
        layer.offset[i] = 0;
    }
    return layer;
}

void batchnorm_layer_free(BatchNormLayer *layer) {
    (void)layer; /* no-op */
}

DenseLayer dense_layer_create(int in_features, int out_features, int seed) {
    DenseLayer layer;
    int i;
    int weight_count = in_features * out_features;

    layer.in_features  = in_features;
    layer.out_features = out_features;

    layer.weights = weight_alloc(weight_count);
    layer.bias    = weight_alloc(out_features);

    for (i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value_int32(i, seed, 8);
    }
    for (i = 0; i < out_features; ++i) {
        layer.bias[i] = seeded_value_int32(i, seed + 19, 100);
    }
    return layer;
}

void dense_layer_free(DenseLayer *layer) {
    (void)layer; /* no-op */
}

/* ---- Layer constructors (from const ROM arrays) ------------------------ */

Conv2DLayer conv2d_layer_from_weights(
    int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    const int32_t *w, const int32_t *b)
{
    Conv2DLayer layer;
    layer.in_channels  = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_h     = kernel_h;
    layer.kernel_w     = kernel_w;
    layer.stride_h     = stride_h;
    layer.stride_w     = stride_w;
    layer.pad_h        = pad_h;
    layer.pad_w        = pad_w;
    layer.groups       = groups;
    /* Intentional cast: forward pass only reads these pointers, never writes. */
    layer.weights = (int32_t *)w;
    layer.bias    = (int32_t *)b;
    return layer;
}

BatchNormLayer batchnorm_layer_from_params(
    int num_features,
    const int32_t *scale, const int32_t *offset)
{
    BatchNormLayer layer;
    layer.num_features = num_features;
    layer.scale  = (int32_t *)scale;
    layer.offset = (int32_t *)offset;
    return layer;
}

DenseLayer dense_layer_from_weights(
    int in_features, int out_features,
    const int32_t *w, const int32_t *b)
{
    DenseLayer layer;
    layer.in_features  = in_features;
    layer.out_features = out_features;
    layer.weights = (int32_t *)w;
    layer.bias    = (int32_t *)b;
    return layer;
}

/* ---- Forward-pass primitives ------------------------------------------ */

/*
 * conv2d_forward  --  PRIMARY CGRA KERNEL
 *
 * Loop nest (7 levels):
 *   n -> oc -> oh -> ow  [spatial tile]
 *     sum = bias[oc]
 *     icg -> kh -> kw    [innermost MAC chain --> CGRA FU target]
 *       sum += input[n,ic,oh*s+kh-p,ow*s+kw-p] * weight[oc,icg,kh,kw]
 *   output[n,oc,oh,ow] = sum
 *
 * For CNN_1D_v2: kernel_h=1 (trivial kh loop), kw=5 MACs per output position.
 */
Tensor *conv2d_forward(const Tensor *input, const Conv2DLayer *layer) {
    int out_h = (input->h + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int out_w = (input->w + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;
    int in_per_group  = layer->in_channels  / layer->groups;
    int out_per_group = layer->out_channels / layer->groups;
    int n, oc, oh, ow, icg, kh, kw;
    Tensor *output = tensor_create(input->n, layer->out_channels, out_h, out_w);

    DEEPBINDI_TRACE(
        "DBG: conv dims n=%d ic=%d oc=%d ih=%d iw=%d oh=%d ow=%d kh=%d kw=%d\r\n",
        input->n, layer->in_channels, layer->out_channels,
        input->h, input->w, out_h, out_w, layer->kernel_h, layer->kernel_w);

    for (n = 0; n < input->n; ++n) {
        for (oc = 0; oc < layer->out_channels; ++oc) {
            int group    = oc / out_per_group;
            int in_start = group * in_per_group;
            for (oh = 0; oh < out_h; ++oh) {
                for (ow = 0; ow < out_w; ++ow) {
                    int32_t sum = layer->bias[oc];
                    for (icg = 0; icg < in_per_group; ++icg) {
                        int ic = in_start + icg;
                        for (kh = 0; kh < layer->kernel_h; ++kh) {
                            for (kw = 0; kw < layer->kernel_w; ++kw) {
                                int ih = oh * layer->stride_h + kh - layer->pad_h;
                                int iw = ow * layer->stride_w + kw - layer->pad_w;
                                int weight_idx;
                                if (ih < 0 || ih >= input->h ||
                                    iw < 0 || iw >= input->w) {
                                    continue; /* zero-padding */
                                }
                                weight_idx =
                                    ((oc * in_per_group + icg) * layer->kernel_h + kh)
                                    * layer->kernel_w + kw;
                                /* *** MAC: CGRA FU target *** */
                                sum += input->data[tensor_index(input, n, ic, ih, iw)]
                                     * layer->weights[weight_idx];
                            }
                        }
                    }
                    output->data[tensor_index(output, n, oc, oh, ow)] = sum;
                }
            }
#ifdef DEEPBINDI_TRACE_CONV_CHANNELS
            DEEPBINDI_TRACE("DBG: conv oc %d/%d done\r\n",
                            oc + 1, layer->out_channels);
#endif
        }
    }
    return output;
}

/*
 * batchnorm_forward_inplace
 *
 * y[c] = (int32_t)(((int64_t)x[c] * scale[c]) >> 7) + offset[c]
 *
 * int64_t intermediate prevents overflow when |x| * |scale| > INT32_MAX.
 * CGRA fusion target: fold into the conv2d output stage.
 */
void batchnorm_forward_inplace(Tensor *input, const BatchNormLayer *layer) {
    int n, c, h, w;
    for (n = 0; n < input->n; ++n) {
        for (c = 0; c < input->c; ++c) {
            int32_t scale  = layer->scale[c];
            int32_t offset = layer->offset[c];
            for (h = 0; h < input->h; ++h) {
                for (w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    input->data[idx] =
                        (int32_t)(((int64_t)input->data[idx] * (int64_t)scale) >> 7)
                        + offset;
                }
            }
        }
    }
}

/*
 * maxpool2d_forward  --  sliding-window max reduction.
 * For CNN_1D_v2: kernel_h=1 (1-D pool along W only).
 */
Tensor *maxpool2d_forward(const Tensor *input,
                           int kernel_h, int kernel_w,
                           int stride_h, int stride_w) {
    int out_h = (input->h - kernel_h) / stride_h + 1;
    int out_w = (input->w - kernel_w) / stride_w + 1;
    int n, c, oh, ow, kh, kw;
    Tensor *output = tensor_create(input->n, input->c, out_h, out_w);
    /* INT32_MIN without limits.h */
    const int32_t INT32_MIN_VAL = (int32_t)(-2147483647 - 1);

    for (n = 0; n < input->n; ++n) {
        for (c = 0; c < input->c; ++c) {
            for (oh = 0; oh < out_h; ++oh) {
                for (ow = 0; ow < out_w; ++ow) {
                    int32_t max_val = INT32_MIN_VAL;
                    for (kh = 0; kh < kernel_h; ++kh) {
                        for (kw = 0; kw < kernel_w; ++kw) {
                            int32_t v = input->data[tensor_index(input, n, c,
                                                       oh * stride_h + kh,
                                                       ow * stride_w + kw)];
                            if (v > max_val) {
                                max_val = v;
                            }
                        }
                    }
                    output->data[tensor_index(output, n, c, oh, ow)] = max_val;
                }
            }
        }
    }
    return output;
}

/*
 * flatten_forward  --  reshape (N,C,H,W) -> (N, C*H*W, 1, 1).
 * Pure data movement; explicit loop (no memcpy).
 */
Tensor *flatten_forward(const Tensor *input) {
    int i;
    int total = tensor_numel(input);
    Tensor *output = tensor_create(input->n, total / input->n, 1, 1);
    for (i = 0; i < total; ++i) {
        output->data[i] = input->data[i];
    }
    return output;
}

/*
 * dense_forward  --  SECONDARY CGRA KERNEL
 *
 * y[out] = bias[out] + sum_in x[in] * W[out][in]
 * Matrix-vector multiply: map `out` across CGRA rows, pipeline `in` inside.
 */
Tensor *dense_forward(const Tensor *input, const DenseLayer *layer) {
    int features = tensor_numel(input) / input->n;
    int n, out, in;
    Tensor *output = tensor_create(input->n, layer->out_features, 1, 1);

    if (features != layer->in_features) {
        DEEPBINDI_LOG_ERROR("dense_forward: shape mismatch (expected %d, got %d)\r\n",
                layer->in_features, features);
        DEEPBINDI_FATAL("dense_forward shape mismatch");
    }

    for (n = 0; n < input->n; ++n) {
        const int32_t *src = input->data + n * layer->in_features;
        for (out = 0; out < layer->out_features; ++out) {
            int32_t sum = layer->bias[out];
            for (in = 0; in < layer->in_features; ++in) {
                sum += src[in] * layer->weights[out * layer->in_features + in];
            }
            output->data[n * layer->out_features + out] = sum;
        }
    }
    return output;
}

/* ---- Activations (in-place) ------------------------------------------- */

void relu_inplace(Tensor *input) {
    int i;
    int total = tensor_numel(input);
    for (i = 0; i < total; ++i) {
        if (input->data[i] < 0) {
            input->data[i] = 0;
        }
    }
}

/*
 * sigmoid_inplace  --  sign threshold for binary classification.
 *
 * Replaces the float sigmoid (which requires expf) with a hard threshold
 * at zero: output = (x > 0) ? 1 : 0.  Valid for the single-output binary
 * head of CNN_1D_v2 where the sign of the pre-sigmoid value determines the
 * predicted class.  Output is 1 (FEAR) or 0 (NO_FEAR).
 */
void sigmoid_inplace(Tensor *input) {
    int i;
    int total = tensor_numel(input);
    for (i = 0; i < total; ++i) {
        input->data[i] = (input->data[i] > 0) ? 1 : 0;
    }
}

/*
 * tensor_rshift_inplace  --  arithmetic right-shift to prevent overflow.
 *
 * Called after BN in the real-weights path before the next convolution.
 * C99 right-shift of a signed int32_t is implementation-defined, but
 * arithmetic (sign-extending) shift is the universal behaviour on all
 * two's-complement targets including RISC-V rv32im.
 */
void tensor_rshift_inplace(Tensor *input, int shift) {
    int i;
    int total = tensor_numel(input);
    for (i = 0; i < total; ++i) {
        input->data[i] >>= shift;
    }
}
