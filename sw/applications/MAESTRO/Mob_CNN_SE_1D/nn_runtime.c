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
// Metrics
#  include "csr.h"

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

    unsigned int cycles;
    CSR_WRITE(CSR_REG_MCYCLE, 0);

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
    CSR_READ(CSR_REG_MCYCLE, &cycles);
    DEEPBINDI_TRACE(
        "DBG: conv dims n=%d ic=%d oc=%d ih=%d iw=%d oh=%d ow=%d kh=%d kw=%d strideh=%d stridew=%d\r\nCycles: %u\r\n",
        input->n, layer->in_channels, layer->out_channels,
        input->h, input->w, out_h, out_w, layer->kernel_h, layer->kernel_w, layer->stride_h, layer->stride_w,
        cycles);

    return output;
}

// ============================================================================
// CODIGO PARA EL CGRA
// ============================================================================

int32_t conv2d_acumular_remanente_cpu(
    const Tensor *input,
    const Conv2DLayer *layer,
    int n, int oc, int oh, int ow, int in_start, int in_per_group
) {
    // 1. El cálculo en la CPU comienza cargando el bias de este canal de salida
    int32_t sum_remanente = layer->bias[oc];

    // 2. Determinamos dónde se quedó el CGRA (el mayor múltiplo de 16)
    int icg_start = (in_per_group / 16) * 16;
    
    // 3. Procesamos los canales que quedan sueltos
    for (int icg = icg_start; icg < in_per_group; ++icg) {
        int ic = in_start + icg;
        
        for (int kh = 0; kh < layer->kernel_h; ++kh) {
            for (int kw = 0; kw < layer->kernel_w; ++kw) {
                
                int ih = oh * layer->stride_h + kh - layer->pad_h;
                int iw = ow * layer->stride_w + kw - layer->pad_w;

                // Al estar en la CPU, verificamos los límites para el padding de ceros
                if (ih < 0 || ih >= input->h || iw < 0 || iw >= input->w) {
                    continue; 
                }

                int weight_idx = ((oc * in_per_group + icg) * layer->kernel_h + kh) 
                                 * layer->kernel_w + kw;
                
                sum_remanente += input->data[tensor_index(input, n, ic, ih, iw)] 
                                 * layer->weights[weight_idx];
            }
        }
    }

    return sum_remanente;
}

int32_t cgra_kernel_3loops(
    const int32_t *ptr_in_lider,
    const int32_t *ptr_w,
    int channels_per_group,  // Múltiplos de 16 gestionados en paralelo
    int kernel_h,
    int kernel_w,
    int input_w,             // Ancho real de la imagen de entrada (sin padding)
    int tam_canal_input      // Tamaño real de un canal de entrada (input_h * input_w)
) {
    // La matriz empieza su acumulación limpia en 0 (el bias lo pone la CPU)
    int32_t sum = 0;

    // PRECALCULO DE CONSTANTES DE SALTO
    // salto_fila_filtro: Al terminar una fila del filtro (kw), cuántos píxeles reales 
    // hay que saltar en la entrada para empezar la siguiente fila justo abajo.
    int salto_fila_filtro = input_w - kernel_w;

    // salto_siguiente_canal: Al terminar el filtro 2D (kh y kw), cuántos píxeles saltamos
    // para posicionarnos en el mismo origen espacial (oh, ow) pero del siguiente canal de entrada.
    int salto_siguiente_canal = tam_canal_input - (kernel_h * input_w);

    // --- LOS 3 BUCLES EJECUTADOS POR LOS PEs ---
    for (int c = 0; c < channels_per_group; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {

                // Multiplicación y acumulación interna en los PEs de datos
                sum += (*ptr_in_lider) * (*ptr_w);

                // Modificación explícita de las direcciones físicas de memoria (+4 bytes)
                ptr_in_lider = (int32_t *)((int32_t)ptr_in_lider + (int32_t)4);
                ptr_w        = (int32_t *)((int32_t)ptr_w        + (int32_t)4);
            }
            // Fin kw: Salto de línea en el filtro sumando el offset en bytes
            ptr_in_lider = (int32_t *)((int32_t)ptr_in_lider + (int32_t)salto_fila_filtro * 4);
        }
        // Fin kh: Salto al siguiente canal sumando el offset en bytes
        ptr_in_lider = (int32_t *)((int32_t)ptr_in_lider + (int32_t)salto_siguiente_canal * 4);
    }

    return sum; 
}

Tensor *conv2d_forward_oe_cgra(const Tensor *input, const Conv2DLayer *layer) {
    int out_h = (input->h + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int out_w = (input->w + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;
    int in_per_group  = layer->in_channels  / layer->groups;
    int out_per_group = layer->out_channels / layer->groups;
    int n, oc, oh, ow, icg, kh, kw;
    Tensor *output = tensor_create(input->n, layer->out_channels, out_h, out_w);

    DEEPBINDI_TRACE(
        "DBG: conv dims n=%d ic=%d oc=%d ih=%d iw=%d oh=%d ow=%d kh=%d kw=%d strideh=%d stridew=%d\r\n",
        input->n, layer->in_channels, layer->out_channels,
        input->h, input->w, out_h, out_w, layer->kernel_h, layer->kernel_w, layer->stride_h, layer->stride_w);

    for (n = 0; n < input->n; ++n) {
        for (oc = 0; oc < layer->out_channels; ++oc) {
            int group    = oc / out_per_group;
            int in_start = group * in_per_group;

            // ====================================================================
            // 1. CÁLCULO ESTRICTO DE LÍMITES SEGUROS
            // ====================================================================
            int oh_seguro_inicio = (layer->pad_h + layer->stride_h - 1) / layer->stride_h;
            int oh_seguro_fin = (input->h + layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
            if (oh_seguro_fin < oh_seguro_inicio) oh_seguro_fin = oh_seguro_inicio;

            int ow_seguro_inicio = (layer->pad_w + layer->stride_w - 1) / layer->stride_w;
            int ow_seguro_fin = (input->w + layer->pad_w - layer->kernel_w) / layer->stride_w + 1;
            if (ow_seguro_fin < ow_seguro_inicio) ow_seguro_fin = ow_seguro_inicio;

            // ====================================================================
            // 2. RECORRIDO DE LA IMAGEN DE SALIDA
            // ====================================================================
            for (oh = 0; oh < out_h; ++oh) {
                for (ow = 0; ow < out_w; ++ow) {

                    // ¿Estamos en la zona segura (dentro de la imagen)?
                    if (oh >= oh_seguro_inicio && oh < oh_seguro_fin &&
                        ow >= ow_seguro_inicio && ow < ow_seguro_fin) {
                        
                        // --------------------------------------------------------
                        // ZONA CGRA: Cálculo del centro en paralelo por canales
                        // --------------------------------------------------------
                        int32_t *ptr_in_base = input->data 
                                            + (n * layer->in_channels * input->h * input->w)
                                            + (in_start * input->h * input->w)
                                            + (oh * layer->stride_h * input->w)
                                            + (ow * layer->stride_w);

                        int tam_filtro_3d = in_per_group * layer->kernel_h * layer->kernel_w;
                        const int32_t *ptr_w_base = &(layer->weights[oc * tam_filtro_3d]);

                        // Filtramos para que el CGRA ejecute SOLO los múltiplos de 16
                        int canales_cgra = (in_per_group / 16) * 16;
                        int32_t resultado_cgra = 0;

                        if (canales_cgra > 0) {
                            resultado_cgra = cgra_kernel_3loops(
                                ptr_in_base, ptr_w_base,
                                canales_cgra, layer->kernel_h, layer->kernel_w,
                                input->w, (input->h * input->w)
                            );
                        }

                        // La CPU calcula el sobrante (canales restantes) y añade el bias
                        int32_t resultado_remanente_cpu = conv2d_acumular_remanente_cpu(
                            input, layer, n, oc, oh, ow, in_start, in_per_group
                        );

                        // Fusión final: Se suma el cómputo del CGRA con el de la CPU
                        output->data[tensor_index(output, n, oc, oh, ow)] = resultado_cgra + resultado_remanente_cpu;

                    } else {
                        
                        // --------------------------------------------------------
                        // ZONA CPU: Cómputo secuencial tradicional para los bordes
                        // --------------------------------------------------------
                        int32_t sum = layer->bias[oc];
                        for (icg = 0; icg < in_per_group; ++icg) {
                            int ic = in_start + icg;
                            for (kh = 0; kh < layer->kernel_h; ++kh) {
                                for (kw = 0; kw < layer->kernel_w; ++kw) {
                                    int ih = oh * layer->stride_h + kh - layer->pad_h;
                                    int iw = ow * layer->stride_w + kw - layer->pad_w;

                                    if (ih < 0 || ih >= input->h || iw < 0 || iw >= input->w) {
                                        continue; 
                                    }
                                    int weight_idx = ((oc * in_per_group + icg) * layer->kernel_h + kh) * layer->kernel_w + kw;
                                    sum += input->data[tensor_index(input, n, ic, ih, iw)] * layer->weights[weight_idx];
                                }
                            }
                        }
                        output->data[tensor_index(output, n, oc, oh, ow)] = sum;
                    }

                }
            }
        }
    }
    return output;
}

// ============================================================================
// END -------------- CODIGO PARA EL CGRA
// ============================================================================



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
    unsigned int cycles;
    CSR_WRITE(CSR_REG_MCYCLE, 0);
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
    CSR_READ(CSR_REG_MCYCLE, &cycles);
    DEEPBINDI_TRACE(
        "DBG: norm dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        input->n, input->c, input->h, input->w, cycles);
}

/*
 * batchnorm_rshift_inplace  --  BN + post-shift fused in a single int64_t pass.
 *
 * Avoids int32_t overflow when the BN output (before shifting) exceeds INT32_MAX.
 * Applies to BN_PW1 in both mobile models where large BN scales push the
 * intermediate above 2^31-1.
 *
 * Formula: y = (((int64_t)x * scale) >> 7 + offset) >> post_shift
 */
void batchnorm_rshift_inplace(Tensor *input, const BatchNormLayer *layer, int post_shift) {
    int n, c, h, w;
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (n = 0; n < input->n; ++n) {
        for (c = 0; c < input->c; ++c) {
            int64_t scale  = (int64_t)layer->scale[c];
            int64_t offset = (int64_t)layer->offset[c];
            for (h = 0; h < input->h; ++h) {
                for (w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    int64_t val = (((int64_t)input->data[idx] * scale) >> 7) + offset;
                    val >>= post_shift;
                    input->data[idx] = (int32_t)val;
                }
            }
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: norm_rshift post_shift=%d dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        post_shift, input->n, input->c, input->h, input->w, cycles);
}

/*
 * batchnorm_rshift_perchannel  --  BN + per-channel post-shift in int64_t.
 *
 * shifts[c] is the post-shift for channel c, calibrated so that
 * the channel's maximum possible output fits in int32_t without
 * destroying signal in channels with small BN scales.
 */
void batchnorm_rshift_perchannel(Tensor *input, const BatchNormLayer *layer,
                                  const int32_t *shifts) {
    int n, c, h, w;
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (n = 0; n < input->n; ++n) {
        for (c = 0; c < input->c; ++c) {
            int64_t scale  = (int64_t)layer->scale[c];
            int64_t offset = (int64_t)layer->offset[c];
            int     shift  = (int)shifts[c];
            for (h = 0; h < input->h; ++h) {
                for (w = 0; w < input->w; ++w) {
                    int idx = tensor_index(input, n, c, h, w);
                    int64_t val = (((int64_t)input->data[idx] * scale) >> 7) + offset;
                    val >>= shift;
                    input->data[idx] = (int32_t)val;
                }
            }
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: norm_rshift_perchannel dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        input->n, input->c, input->h, input->w, cycles);
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
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
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
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: maxpool kernel=[%dx%d] stride=[%dx%d] in_dims n=%d c=%d h=%d w=%d out_dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        kernel_h, kernel_w, stride_h, stride_w,
        input->n, input->c, input->h, input->w,
        output->n, output->c, output->h, output->w,
        cycles);

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
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (i = 0; i < total; ++i) {
        output->data[i] = input->data[i];
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: flatten total=%d in_dims n=%d c=%d h=%d w=%d out_dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        total,
        input->n, input->c, input->h, input->w,
        output->n, output->c, output->h, output->w,
        cycles);

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
    unsigned int cycles;

    if (features != layer->in_features) {
        DEEPBINDI_LOG_ERROR("dense_forward: shape mismatch (expected %d, got %d)\r\n",
                layer->in_features, features);
        DEEPBINDI_FATAL("dense_forward shape mismatch");
    }

    CSR_WRITE(CSR_REG_MCYCLE, 0);
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
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: dense in_features=%d out_features=%d batch_n=%d dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        layer->in_features, layer->out_features, input->n,
        output->n, output->c, output->h, output->w,
        cycles);

    return output;
}
/* ---- Activations (in-place) ------------------------------------------- */

void relu_inplace(Tensor *input) {
    int i;
    int total = tensor_numel(input);
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (i = 0; i < total; ++i) {
        if (input->data[i] < 0) {
            input->data[i] = 0;
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: relu dim=%d\r\nCycles: %u\r\n",
        total, cycles);
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
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (i = 0; i < total; ++i) {
        input->data[i] = (input->data[i] > 0) ? 1 : 0;
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: sigmoid total=%d dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        total, input->n, input->c, input->h, input->w, cycles);
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
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (i = 0; i < total; ++i) {
        input->data[i] >>= shift;
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: rshift total=%d\r\nCycles: %u\r\n",
        total, cycles);
}

/* ---- SE (Squeeze-and-Excitation) primitives ------------------------------- */

/*
 * globalavgpool_forward  --  reduce (N,C,H,W) to (N,C,1,1) by spatial mean.
 *
 * int64_t accumulator prevents overflow for large H*W or large values.
 * For the MobileCNN-SE-1D use case (H=1, W=4), overflow is impossible with
 * int32_t values, but the 64-bit path is retained for generality.
 */
Tensor *globalavgpool_forward(const Tensor *input) {
    int n, c, h, iw;
    int spatial = input->h * input->w;
    Tensor *output = tensor_create(input->n, input->c, 1, 1);
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (n = 0; n < input->n; ++n) {
        for (c = 0; c < input->c; ++c) {
            int64_t sum = 0;
            for (h = 0; h < input->h; ++h) {
                for (iw = 0; iw < input->w; ++iw) {
                    sum += (int64_t)input->data[tensor_index(input, n, c, h, iw)];
                }
            }
            output->data[n * input->c + c] = (int32_t)(sum / (int64_t)spatial);
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: globalavgpool spatial_size=%d in_dims n=%d c=%d h=%d w=%d out_dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        spatial,
        input->n, input->c, input->h, input->w,
        output->n, output->c, 1, 1,
        cycles);

    return output;
}

/*
 * hardsigmoid_se_inplace  --  integer hard-sigmoid for SE channel weights.
 *
 * Formula: y = clip((x >> shift) + 64, 0, 128)
 * Output range: [0, 128] where 128 == 1.0 in Q7.
 *
 * The shift parameter places the input x in a range such that the
 * linear region of hard-sigmoid is meaningful.  It is calibrated by
 * export_mobile_weights.py from the actual trained SE Dense2 output range.
 *
 * Example (shift=23): max(|x|) ~ 350M -> 350M >> 23 = 41 -> y in [23, 105]
 *
 * In the dummy path, shift=0 and values are small (< 1000) -> y near 64
 * (all channels equally weighted at 0.5), which is a valid no-op for benchmarking.
 */
void hardsigmoid_se_inplace(Tensor *input, int shift) {
    int i;
    int total = tensor_numel(input);
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (i = 0; i < total; ++i) {
        int32_t y = (input->data[i] >> shift) + 64;
        if (y < 0)   y = 0;
        if (y > 128) y = 128;
        input->data[i] = y;
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: hardsigmoid_se shift=%d total=%d dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        shift, total, input->n, input->c, input->h, input->w, cycles);
}

/*
 * se_channel_scale_inplace  --  SE feature recalibration.
 *
 * Scales each channel c of `feat` by the Q7 excitation weight se_weights[c]:
 * feat[n,c,h,w] = ((int64_t)feat[n,c,h,w] * se_weights->data[c]) >> 7
 *
 * int64_t intermediate prevents overflow when feat values are near INT32_MAX
 * (which can happen after large BN scales).  The >>7 undoes the Q7 scale.
 */
void se_channel_scale_inplace(Tensor *feat, const Tensor *se_weights) {
    int n, c, h, iw;
    unsigned int cycles;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (n = 0; n < feat->n; ++n) {
        for (c = 0; c < feat->c; ++c) {
            int32_t w_q7 = se_weights->data[c]; /* Q7: 128 = 1.0 */
            for (h = 0; h < feat->h; ++h) {
                for (iw = 0; iw < feat->w; ++iw) {
                    int idx = tensor_index(feat, n, c, h, iw);
                    feat->data[idx] =
                        (int32_t)(((int64_t)feat->data[idx] * (int64_t)w_q7) >> 7);
                }
            }
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &cycles);

    DEEPBINDI_TRACE(
        "DBG: se_channel_scale dims n=%d c=%d h=%d w=%d\r\nCycles: %u\r\n",
        feat->n, feat->c, feat->h, feat->w, cycles);
}
