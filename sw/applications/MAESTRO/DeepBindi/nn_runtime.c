/**
 * nn_runtime.c  –  Reference C implementation of inference primitives.
 *                  STATIC VERSION: no malloc / calloc / free anywhere.
 *
 * This file is a drop-in replacement for the dynamic c_port/nn_runtime.c.
 * The ONLY changes are in memory management:
 *
 *   tensor_create()         → allocates from g_act_arena  (via act_alloc)
 *   tensor_free()           → NO-OP (memory is reclaimed by act_arena_reset)
 *   tensor_clone()          → allocates from g_act_arena  (via act_alloc)
 *   *_layer_create()        → allocates from g_weight_pool (via weight_alloc)
 *   *_layer_free()          → NO-OP (weights persist for the program lifetime)
 *
 * All compute functions (conv2d_forward, batchnorm_forward_inplace, …) are
 * IDENTICAL to the dynamic version.  Replacing the memory backend does not
 * change the numerical results; checksums from main.c match the dynamic port.
 *
 * CGRA note: with static allocation the weight and activation base addresses
 * are known at link time, making it straightforward to pre-program DMA
 * descriptors and CGRA memory access patterns before the forward pass begins.
 */
#include "deepbindi_config.h"  /* logging + fatal error macros – include first */
#include "nn_runtime.h"
#include "arena.h"

#include <math.h>
#include <string.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

/* Row-major NCHW index: data[((n*C+c)*H+h)*W+w] */
static int tensor_index(const Tensor *t, int n, int c, int h, int w) {
    return (((n * t->c + c) * t->h + h) * t->w + w);
}

/* Deterministic pseudo-random value for dummy weight / input generation.
 * Same formula as the dynamic port – produces identical numerical results. */
static float seeded_value(int index, int seed, float scale) {
    int raw = ((index + 1) * (seed + 3) * 17) % 257;
    return scale * (((float)raw / 128.0f) - 1.0f);
}

/* ── Tensor helpers ──────────────────────────────────────────────────────── */

/*
 * tensor_create  –  Allocate a zero-initialised tensor.
 *
 * STATIC CHANGE: instead of malloc(Tensor) + calloc(data),
 * we grab one slot from g_tensor_pool and n*c*h*w floats from g_act_arena.
 * Both pools are reset together by act_arena_reset() between forward passes.
 */
Tensor *tensor_create(int n, int c, int h, int w) {
    int count = n * c * h * w;
    /* Grab a Tensor descriptor from the struct pool */
    Tensor *tensor = &g_tensor_pool[g_tensor_top++];
    tensor->n    = n;
    tensor->c    = c;
    tensor->h    = h;
    tensor->w    = w;
    /* Grab float storage from the activation arena (already zero from memset) */
    tensor->data = act_alloc(count);
    return tensor;
}

/*
 * tensor_clone  –  Deep-copy into a new arena allocation.
 */
Tensor *tensor_clone(const Tensor *src) {
    Tensor *dst = tensor_create(src->n, src->c, src->h, src->w);
    memcpy(dst->data, src->data, (size_t)tensor_numel(src) * sizeof(float));
    return dst;
}

/*
 * tensor_free  –  NO-OP in the static port.
 *
 * Memory is reclaimed en-masse by act_arena_reset() at the start of the
 * next model's forward pass.  Keeping the same function signature means
 * cnn_models_c.c does not need to be modified for this call.
 *
 * CGRA note: the no-op free means all intermediate tensors remain addressable
 * during the entire forward pass – useful for debugging memory access patterns
 * and verifying intermediate results against a software reference.
 */
void tensor_free(Tensor *tensor) {
    (void)tensor; /* intentional no-op */
}

int tensor_numel(const Tensor *tensor) {
    return tensor->n * tensor->c * tensor->h * tensor->w;
}

void tensor_fill_dummy(Tensor *tensor, float scale, int seed) {
    int total = tensor_numel(tensor);
    for (int i = 0; i < total; ++i) {
        tensor->data[i] = seeded_value(i, seed, scale);
    }
}

float tensor_checksum(const Tensor *tensor) {
    float acc = 0.0f;
    int total = tensor_numel(tensor);
    for (int i = 0; i < total; ++i) {
        acc += tensor->data[i] * (float)((i % 7) + 1);
    }
    return acc;
}

void tensor_print_values(const char *name, const Tensor *tensor, int max_values) {
#ifdef DEEPBINDI_ENABLE_LOGGING
    int total = tensor_numel(tensor);
    int limit = total < max_values ? total : max_values;
    DEEPBINDI_PRINTF("%s | shape=(%d,%d,%d,%d) | checksum=%0.6f | values=[",
           name, tensor->n, tensor->c, tensor->h, tensor->w,
           tensor_checksum(tensor));
    for (int i = 0; i < limit; ++i) {
        DEEPBINDI_PRINTF("%s%0.6f", i == 0 ? "" : ", ", tensor->data[i]);
    }
    if (limit < total) {
        DEEPBINDI_PRINTF(", ...");
    }
    DEEPBINDI_PRINTF("]\n");
#else
    (void)name; (void)tensor; (void)max_values;
#endif
}

/* ── Layer constructors ──────────────────────────────────────────────────── */

/*
 * conv2d_layer_create  –  Build a Conv2DLayer with weights from the weight pool.
 *
 * STATIC CHANGE: weight_alloc() replaces malloc().  The returned layer's
 * .weights and .bias pointers are offsets into g_weight_pool.
 *
 * CGRA note: all weights for the entire model sit in a contiguous region of
 * g_weight_pool.  The starting offset of each layer's weights can be computed
 * statically, enabling compile-time DMA descriptor generation.
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
) {
    Conv2DLayer layer;
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

    /* Allocate weight and bias storage from the static weight pool */
    layer.weights = weight_alloc(weight_count);
    layer.bias    = weight_alloc(out_channels);

    for (int i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value(i, seed, 0.04f);
    }
    for (int i = 0; i < out_channels; ++i) {
        layer.bias[i] = seeded_value(i, seed + 11, 0.01f);
    }
    return layer;
}

/*
 * conv2d_layer_free  –  NO-OP in the static port.
 * Weights remain valid in g_weight_pool for the lifetime of the program.
 */
void conv2d_layer_free(Conv2DLayer *layer) {
    (void)layer; /* intentional no-op */
}

/*
 * batchnorm_layer_create  –  Build a BatchNormLayer from the weight pool.
 *
 * STATIC CHANGE: four weight_alloc() calls replace four malloc() calls.
 * gamma, beta, mean, var all live contiguously inside g_weight_pool.
 */
BatchNormLayer batchnorm_layer_create(int num_features, float eps, int seed) {
    BatchNormLayer layer;
    layer.num_features = num_features;
    layer.eps          = eps;

    layer.gamma = weight_alloc(num_features);
    layer.beta  = weight_alloc(num_features);
    layer.mean  = weight_alloc(num_features);
    layer.var   = weight_alloc(num_features);

    for (int i = 0; i < num_features; ++i) {
        layer.gamma[i] = 1.0f + fabsf(seeded_value(i, seed,      0.08f));
        layer.beta[i]  =               seeded_value(i, seed + 3,  0.02f);
        layer.mean[i]  =               seeded_value(i, seed + 7,  0.03f);
        layer.var[i]   = 1.0f + fabsf(seeded_value(i, seed + 13, 0.05f));
    }
    return layer;
}

/* NO-OP: BN parameters persist in the weight pool. */
void batchnorm_layer_free(BatchNormLayer *layer) {
    (void)layer;
}

/*
 * dense_layer_create  –  Build a DenseLayer from the weight pool.
 *
 * STATIC CHANGE: weight_alloc() × 2 replace two malloc() calls.
 * For large FC layers (e.g. CNN_2D_v2 Dense1: 23296×32 = 745 472 floats ≈ 2.8 MB)
 * this is the dominant consumer of g_weight_pool.
 */
DenseLayer dense_layer_create(int in_features, int out_features, int seed) {
    DenseLayer layer;
    int weight_count = in_features * out_features;

    layer.in_features  = in_features;
    layer.out_features = out_features;

    layer.weights = weight_alloc(weight_count);
    layer.bias    = weight_alloc(out_features);

    for (int i = 0; i < weight_count; ++i) {
        layer.weights[i] = seeded_value(i, seed, 0.03f);
    }
    for (int i = 0; i < out_features; ++i) {
        layer.bias[i] = seeded_value(i, seed + 19, 0.02f);
    }
    return layer;
}

/* NO-OP: dense weights persist in the weight pool. */
void dense_layer_free(DenseLayer *layer) {
    (void)layer;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Forward-pass primitives  –  NUMERICALLY IDENTICAL TO THE DYNAMIC VERSION
 *
 * None of the functions below were changed; they still call tensor_create()
 * (which now goes to the arena) but otherwise operate on plain float arrays.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/*
 * conv2d_forward  –  PRIMARY CGRA KERNEL
 *
 * 2-D convolution with zero-padding and optional grouped / depthwise conv.
 *
 * Loop nest (7 levels):
 *   for n   in [0, batch)            – embarrassingly parallel per sample
 *     for oc  in [0, out_channels)   – output feature maps
 *       for oh  in [0, out_h)        – output rows
 *         for ow  in [0, out_w)      – output columns     ← spatial tile
 *           sum = bias[oc]
 *           for icg in [0, in_per_group) – input channels per group
 *             for kh  in [0, kernel_h)   – kernel row
 *               for kw  in [0, kernel_w) – kernel col
 *                 sum += input[n,ic,oh*s+kh,ow*s+kw] * weight[oc,icg,kh,kw]
 *           output[n,oc,oh,ow] = sum
 *
 * CGRA mapping hints:
 *   • (kh,kw) innermost loops → one MAC per iteration, no cross-iteration
 *     dependency → direct CGRA functional-unit (FU) chain.
 *   • (oh,ow) → spatial tile across CGRA rows/columns.
 *   • Depthwise (groups == in_channels): in_per_group == 1, icg loop collapses.
 *   • Static memory: weight base address = &g_weight_pool[layer.weights – g_weight_pool]
 *     known at link time → compile-time DMA descriptor.
 */
Tensor *conv2d_forward(const Tensor *input, const Conv2DLayer *layer) {
    int out_h = (input->h + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int out_w = (input->w + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;
    int in_per_group  = layer->in_channels  / layer->groups;
    int out_per_group = layer->out_channels / layer->groups;
    Tensor *output = tensor_create(input->n, layer->out_channels, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int oc = 0; oc < layer->out_channels; ++oc) {
            int group    = oc / out_per_group;
            int in_start = group * in_per_group;
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = layer->bias[oc];
                    for (int icg = 0; icg < in_per_group; ++icg) {
                        int ic = in_start + icg;
                        for (int kh = 0; kh < layer->kernel_h; ++kh) {
                            for (int kw = 0; kw < layer->kernel_w; ++kw) {
                                int ih = oh * layer->stride_h + kh - layer->pad_h;
                                int iw = ow * layer->stride_w + kw - layer->pad_w;
                                int weight_idx;
                                if (ih < 0 || ih >= input->h || iw < 0 || iw >= input->w) {
                                    continue;
                                }
                                weight_idx =
                                    ((((oc * in_per_group) + icg) * layer->kernel_h + kh) * layer->kernel_w) + kw;
                                /* *** MAC: CGRA FU target *** */
                                sum += input->data[tensor_index(input, n, ic, ih, iw)] * layer->weights[weight_idx];
                            }
                        }
                    }
                    output->data[tensor_index(output, n, oc, oh, ow)] = sum;
                }
            }
        }
    }
    return output;
}

/*
 * batchnorm_forward_inplace
 *
 * y = gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]
 * In-place; no new tensor allocation.
 * CGRA fusion target: fold into the conv2d output stage.
 */
void batchnorm_forward_inplace(Tensor *input, const BatchNormLayer *layer) {
    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            float gamma   = layer->gamma[c];
            float beta    = layer->beta[c];
            float mean    = layer->mean[c];
            float inv_std = 1.0f / sqrtf(layer->var[c] + layer->eps);
            for (int h = 0; h < input->h; ++h) {
                for (int w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    input->data[idx] = gamma * (input->data[idx] - mean) * inv_std + beta;
                }
            }
        }
    }
}

/*
 * maxpool2d_forward  –  Sliding-window max reduction.
 * CGRA: comparison-tree reduction over (kernel_h × kernel_w) window.
 */
Tensor *maxpool2d_forward(const Tensor *input, int kernel_h, int kernel_w,
                           int stride_h, int stride_w) {
    int out_h = (input->h - kernel_h) / stride_h + 1;
    int out_w = (input->w - kernel_w) / stride_w + 1;
    Tensor *output = tensor_create(input->n, input->c, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_value = -3.402823e+38f; /* -FLT_MAX (no math.h dep) */
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            float v = input->data[tensor_index(input, n, c,
                                                               oh * stride_h + kh,
                                                               ow * stride_w + kw)];
                            if (v > max_value) {
                                max_value = v;
                            }
                        }
                    }
                    output->data[tensor_index(output, n, c, oh, ow)] = max_value;
                }
            }
        }
    }
    return output;
}

/*
 * adaptive_avg_pool2d_forward  –  Global (or partial) average pooling.
 * Most common use: out_h=1, out_w=1 (global avg pool before classifier).
 */
Tensor *adaptive_avg_pool2d_forward(const Tensor *input, int out_h, int out_w) {
    Tensor *output = tensor_create(input->n, input->c, out_h, out_w);

    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                int h_start = (oh * input->h) / out_h;
                int h_end   = ((oh + 1) * input->h + out_h - 1) / out_h;
                for (int ow = 0; ow < out_w; ++ow) {
                    int w_start = (ow * input->w) / out_w;
                    int w_end   = ((ow + 1) * input->w + out_w - 1) / out_w;
                    float sum = 0.0f;
                    int count = 0;
                    for (int ih = h_start; ih < h_end; ++ih) {
                        for (int iw = w_start; iw < w_end; ++iw) {
                            sum += input->data[tensor_index(input, n, c, ih, iw)];
                            ++count;
                        }
                    }
                    output->data[tensor_index(output, n, c, oh, ow)] = sum / (float)count;
                }
            }
        }
    }
    return output;
}

/*
 * flatten_forward  –  Reshape (N,C,H,W) → (N, C*H*W, 1, 1).
 * Pure data movement; no arithmetic.
 */
Tensor *flatten_forward(const Tensor *input) {
    Tensor *output = tensor_create(input->n, tensor_numel(input) / input->n, 1, 1);
    memcpy(output->data, input->data, (size_t)tensor_numel(input) * sizeof(float));
    return output;
}

/*
 * dense_forward  –  SECONDARY CGRA KERNEL
 *
 * y[out] = bias[out] + Σ_in x[in] * W[out][in]
 *
 * Matrix-vector multiply: map `out` across CGRA rows, pipeline `in` inside.
 */
Tensor *dense_forward(const Tensor *input, const DenseLayer *layer) {
    int features = tensor_numel(input) / input->n;
    Tensor *output = tensor_create(input->n, layer->out_features, 1, 1);

    if (features != layer->in_features) {
        DEEPBINDI_LOG_ERROR("dense_forward: expected %d features, got %d\n",
                layer->in_features, features);
        DEEPBINDI_FATAL("dense_forward shape mismatch");
    }

    for (int n = 0; n < input->n; ++n) {
        const float *src = input->data + n * layer->in_features;
        for (int out = 0; out < layer->out_features; ++out) {
            float sum = layer->bias[out];
            for (int in = 0; in < layer->in_features; ++in) {
                sum += src[in] * layer->weights[out * layer->in_features + in];
            }
            output->data[n * layer->out_features + out] = sum;
        }
    }
    return output;
}

/*
 * concat_height  –  Concatenate two tensors along H (torch.cat dim=2).
 * Pure data movement.
 */
Tensor *concat_height(const Tensor *a, const Tensor *b) {
    Tensor *output;
    if (a->n != b->n || a->c != b->c || a->w != b->w) {
        DEEPBINDI_LOG_ERROR("concat_height: incompatible shapes\n");
        DEEPBINDI_FATAL("concat_height shape mismatch");
    }
    output = tensor_create(a->n, a->c, a->h + b->h, a->w);
    for (int n = 0; n < output->n; ++n) {
        for (int c = 0; c < output->c; ++c) {
            for (int h = 0; h < a->h; ++h) {
                for (int w = 0; w < output->w; ++w) {
                    output->data[tensor_index(output, n, c, h, w)] =
                        a->data[tensor_index(a, n, c, h, w)];
                }
            }
            for (int h = 0; h < b->h; ++h) {
                for (int w = 0; w < output->w; ++w) {
                    output->data[tensor_index(output, n, c, a->h + h, w)] =
                        b->data[tensor_index(b, n, c, h, w)];
                }
            }
        }
    }
    return output;
}

/*
 * add_forward  –  Element-wise addition (residual connection).
 * Trivially parallel across all elements.
 */
Tensor *add_forward(const Tensor *a, const Tensor *b) {
    int total;
    Tensor *output;
    if (a->n != b->n || a->c != b->c || a->h != b->h || a->w != b->w) {
        DEEPBINDI_LOG_ERROR("add_forward: incompatible shapes\n");
        DEEPBINDI_FATAL("add_forward shape mismatch");
    }
    output = tensor_create(a->n, a->c, a->h, a->w);
    total  = tensor_numel(a);
    for (int i = 0; i < total; ++i) {
        output->data[i] = a->data[i] + b->data[i];
    }
    return output;
}

/*
 * channel_scale_forward  –  SE channel re-weighting.
 * output[n,c,h,w] = input[n,c,h,w] * scale[n,c,0,0]
 */
Tensor *channel_scale_forward(const Tensor *input, const Tensor *scale) {
    Tensor *output;
    if (scale->n != input->n || scale->c != input->c ||
        scale->h != 1 || scale->w != 1) {
        DEEPBINDI_LOG_ERROR("channel_scale_forward: scale must be (n,c,1,1)\n");
        DEEPBINDI_FATAL("channel_scale_forward shape mismatch");
    }
    output = tensor_create(input->n, input->c, input->h, input->w);
    for (int n = 0; n < input->n; ++n) {
        for (int c = 0; c < input->c; ++c) {
            float factor = scale->data[tensor_index(scale, n, c, 0, 0)];
            for (int h = 0; h < input->h; ++h) {
                for (int w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    output->data[idx] = input->data[idx] * factor;
                }
            }
        }
    }
    return output;
}

/* ── Activation functions (in-place, unchanged from dynamic version) ─────── */

void relu_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        if (input->data[i] < 0.0f) {
            input->data[i] = 0.0f;
        }
    }
}

void sigmoid_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        input->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
}

void hardsigmoid_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        float v = (input->data[i] + 3.0f) / 6.0f;
        if (v < 0.0f) { v = 0.0f; }
        if (v > 1.0f) { v = 1.0f; }
        input->data[i] = v;
    }
}

void hardswish_inplace(Tensor *input) {
    int total = tensor_numel(input);
    for (int i = 0; i < total; ++i) {
        float gate = (input->data[i] + 3.0f) / 6.0f;
        if (gate < 0.0f) { gate = 0.0f; }
        if (gate > 1.0f) { gate = 1.0f; }
        input->data[i] *= gate;
    }
}

void softmax_inplace(Tensor *input) {
    int total = tensor_numel(input);
    float max_value = input->data[0];
    float sum = 0.0f;
    for (int i = 1; i < total; ++i) {
        if (input->data[i] > max_value) {
            max_value = input->data[i];
        }
    }
    for (int i = 0; i < total; ++i) {
        input->data[i] = expf(input->data[i] - max_value);
        sum += input->data[i];
    }
    for (int i = 0; i < total; ++i) {
        input->data[i] /= sum;
    }
}
