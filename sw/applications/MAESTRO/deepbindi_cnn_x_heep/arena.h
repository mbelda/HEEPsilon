/**
 * arena.h  --  Static memory pools, sized for CNN_1D_v2 (PYE) on X-HEEP.
 *              int32_t variant: no floating-point, no math.h.
 *
 * Two global pools replace all heap allocations:
 *
 *   g_weight_pool[]  Holds weights, biases, and BatchNorm parameters for
 *                    one CNN_1D_v2 forward pass.  Allocated once at startup;
 *                    never freed.  Replace with const arrays in flash (.rodata)
 *                    once trained weights are available (see weights_cnn_1d_v2.h).
 *
 *   g_act_arena[]    Scratch buffer for intermediate activation tensors.
 *                    Bump-allocated; reset by act_arena_reset() at the start of
 *                    each forward pass.  All tensor_free() calls are no-ops.
 *
 *   g_tensor_pool[]  Pool of Tensor descriptor structs (n,c,h,w,*data).
 *
 * Pool sizes for CNN_1D_v2 (one forward pass, 32-bit int, dummy weights):
 *
 *   Weight pool (int32_t element counts):
 *     Conv1 weights : 57 * 32 * 1 * 5  =  9 120
 *     Conv1 bias    :              32
 *     BN1 scale+off :          2 * 32  =     64
 *     Conv2 weights : 32 * 64 * 1 * 5  = 10 240
 *     Conv2 bias    :              64
 *     BN2 scale+off :          2 * 64  =    128
 *     FC    weights :      64 * 1      =     64
 *     FC    bias    :               1
 *     Total                              19 713  -->  WEIGHT_POOL_WORDS = 24 000
 *
 *   Activation arena (int32_t element counts):
 *     input         :  1*57*1*10 =  570
 *     conv1 out     :  1*32*1*8  =  256   out_w = (10+2*1-5)/1+1 = 8
 *     pool1 out     :  1*32*1*4  =  128   out_w = (8-2)/2+1 = 4
 *     conv2 out     :  1*64*1*2  =  128   out_w = (4+2*1-5)/1+1 = 2
 *     pool2 out     :  1*64*1*1  =   64   out_w = (2-2)/2+1 = 1
 *     flatten       :  1*64*1*1  =   64
 *     fc out        :  1*1*1*1   =    1
 *     Total                        1 211  -->  ACT_ARENA_WORDS = 2 048
 *
 *   Tensor descriptors: 7 live  -->  MAX_LIVE_TENSORS = 16
 *
 * Total SRAM footprint:
 *   Weight pool  : 24 000 * 4 B =  93.75 KB  (move to flash for production)
 *   Act arena    :  2 048 * 4 B =   8.00 KB
 *   Tensor pool  :     16 * 24 B =  0.375 KB
 *   Total                         ~102 KB
 */

#ifndef ARENA_H
#define ARENA_H

#include <stdint.h>
#include "nn_runtime.h"

/* ---- Pool sizing ------------------------------------------------------- */

#ifdef DEEPBINDI_REAL_WEIGHTS
#define WEIGHT_POOL_WORDS   1
#else
#define WEIGHT_POOL_WORDS   24000
#endif
#define ACT_ARENA_WORDS     2048
#define MAX_LIVE_TENSORS    16

/* ---- Global pool declarations ----------------------------------------- */

extern int32_t g_weight_pool[WEIGHT_POOL_WORDS];
extern int     g_weight_top;

extern int32_t g_act_arena[ACT_ARENA_WORDS];
extern int     g_act_top;

extern Tensor  g_tensor_pool[MAX_LIVE_TENSORS];
extern int     g_tensor_top;

/* ---- Allocator functions ----------------------------------------------- */

/** Bump-allocate n int32_t slots from the weight pool (never freed in production). */
int32_t *weight_alloc(int n);

/** Bump-allocate n int32_t slots from the activation arena. */
int32_t *act_alloc(int n);

/** Reset the activation arena and tensor-struct pool between forward passes. */
void act_arena_reset(void);

/**
 * Reset the weight pool to allow re-loading dummy weights on the next forward pass.
 * Only valid when dummy weights are used (seeded deterministic generation).
 * NOT safe to call if real trained weights have been loaded from const arrays,
 * because those weights live outside the pool and are not regenerated.
 */
void weight_arena_reset(void);

/** Print current pool usage (no %f). */
void arena_stats(void);

#endif /* ARENA_H */
