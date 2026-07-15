/**
 * arena.h  --  Static memory pools for DeepBindi CNN inference on X-HEEP.
 *              int32_t variant: no floating-point, no math.h.
 *
 * Two global pools replace all heap allocations:
 *
 *   g_weight_pool[]  Dummy weights for benchmarking (seeded generation).
 *                    Unused in real-weights build (weights live in .rodata).
 *
 *   g_act_arena[]    Scratch buffer for intermediate activation tensors.
 *                    Bump-allocated; reset by act_arena_reset() at the start
 *                    of each forward pass.  tensor_free() is a no-op.
 *
 *   g_tensor_pool[]  Pool of Tensor descriptor structs (n,c,h,w,*data).
 *
 * Model selection: compile with -DDEEPBINDI_MODEL=N
 *   0 (default): CNN_1D_v2    (baseline, 19713 weight words, 1211 act words)
 *   1:           MobileCNN-1D (DW+PW,    ~4937 weight words, 1731 act words)
 *   2:           MobileCNN-SE (DW+PW+SE, ~5489 weight words, 1803 act words)
 *
 * Weight pool (dummy build, worst-case per model):
 *
 *   CNN_1D_v2 (MODEL 0):
 *     Conv1(57*32*5)+bias+BN : 9120+32+64    =  9 216
 *     Conv2(32*64*5)+bias+BN : 10240+64+128  = 10 432
 *     FC(64)+bias             : 64+1          =     65
 *     Total                                    19 713  --> 24 000 words
 *
 *   MobileCNN-1D (MODEL 1):
 *     DW1(57*5)+bias+BN(57)   :  285+57+114  =    456
 *     PW1(57*32)+bias+BN(32)  : 1824+32+64   =  1 920
 *     DW2(32*5)+bias+BN(32)   :  160+32+64   =    256
 *     PW2(32*64)+bias+BN(64)  : 2048+64+128  =  2 240
 *     FC(64)+bias              :   64+1       =     65
 *     Total                                     4 937  -->  6 000 words
 *
 *   MobileCNN-SE-1D (MODEL 2):
 *     Same as MODEL 1, plus:
 *     SE_Dense1(32*8)+bias     :  256+8       =    264
 *     SE_Dense2(8*32)+bias     :  256+32      =    288
 *     Total                                     5 489  -->  6 000 words
 *
 * Activation arena (all models, int32_t elements):
 *   CNN_1D_v2:       1211 words peak
 *   MobileCNN-1D:    1731 words peak
 *   MobileCNN-SE-1D: 1803 words peak
 *   --> ACT_ARENA_WORDS = 2048  (covers all models)
 *
 * Tensor pool: max 12 simultaneous descriptors (MobileCNN-SE-1D)
 *   --> MAX_LIVE_TENSORS = 16  (covers all models)
 *
 * Total SRAM footprint (dummy build, largest model CNN_1D_v2):
 *   Weight pool  : 24 000 * 4 B =  93.75 KB  (move to flash for production)
 *   Act arena    :  2 048 * 4 B =   8.00 KB
 *   Tensor pool  :     16 * 24 B =  0.38 KB
 *   Total                         ~102 KB
 */

#ifndef ARENA_H
#define ARENA_H

#include <stdint.h>
#include "nn_runtime.h"

/* ---- Pool sizing  (model-aware) ---------------------------------------- */

#ifndef DEEPBINDI_MODEL
#  define DEEPBINDI_MODEL 0   /* default: CNN_1D_v2 */
#endif

/* Real-weights build: weight pool is unused (weights in .rodata).
 * A 1-word stub avoids a zero-size array (C99 UB). */
#if defined(DEEPBINDI_REAL_WEIGHTS)
#  define WEIGHT_POOL_WORDS   1
#elif DEEPBINDI_MODEL == 1 || DEEPBINDI_MODEL == 2
#  define WEIGHT_POOL_WORDS   6000   /* MobileCNN-1D / SE (max ~5489 dummy words) */
#else
#  define WEIGHT_POOL_WORDS   24000  /* CNN_1D_v2 baseline (max ~19713 dummy words) */
#endif

#define ACT_ARENA_WORDS     2048   /* covers all models (max 1803 words for SE) */
#define MAX_LIVE_TENSORS    16     /* covers all models (max 12 for SE model) */

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
