/**
 * arena.h  –  Static memory pools for the no-malloc CNN inference port.
 *
 * Two global pools replace all heap allocations:
 *
 *   g_weight_pool[]   Holds every layer's weights, biases, and BatchNorm
 *                     parameters for all 10 models run sequentially.
 *                     Allocated once during the model forward pass; never freed.
 *                     In a real deployment you would compile only ONE model,
 *                     dramatically reducing this size.
 *
 *   g_act_arena[]     Scratch buffer for intermediate activation tensors.
 *                     Allocated with a bump allocator; reset at the start of
 *                     each model's forward pass (act_arena_reset()).
 *
 *   g_tensor_pool[]   Pool of Tensor descriptor structs (n,c,h,w,*data).
 *                     Pointers into this pool replace heap-allocated structs.
 *                     Reset together with g_act_arena.
 *
 * CGRA note
 * ---------
 * This layout gives the accelerator exactly the memory model it needs:
 *  • Weights  → read-only region, can be pre-loaded into CGRA local memory.
 *  • Activations → double-buffer or ping-pong from g_act_arena; the arena
 *    base address and stride are fixed at compile time.
 *  • No pointer chasing, no TLB misses caused by malloc scatter.
 *
 * Sizing (all 10 models run once sequentially, worst-case per forward pass):
 *  WEIGHT_POOL_FLOATS = 2 000 000   ≈ 7.6 MB  (dominated by FC layers in PYB)
 *  ACT_ARENA_FLOATS   =   500 000   ≈ 1.9 MB  (dominated by PYB parallel branches)
 *  MAX_LIVE_TENSORS   =       128             (peak: MobileNetV3 ≈ 82 tensors)
 */

#ifndef ARENA_H
#define ARENA_H

#include "nn_runtime.h"

/* ── Pool sizing ─────────────────────────────────────────────────────────── */

/* Weight pool: holds weights for all 10 models run once each.
 * For a single-model deployment reduce to the model's actual weight count. */
//#define WEIGHT_POOL_FLOATS 2000000
//#define WEIGHT_POOL_FLOATS 333121 // CNN_2D_v1
#define WEIGHT_POOL_FLOATS 18817 // CNN_1D_v1

/* Activation arena: worst case is CNN_2D_v2 with parallel branches (~368 k).
 * Increase if you add larger models; decrease for single-model deployments. */
//#define ACT_ARENA_FLOATS 500000
//#define ACT_ARENA_FLOATS 137810 // CNN_2D_v1
#define ACT_ARENA_FLOATS 1595 // CNN_1D_v1

/* Maximum number of live Tensor descriptors during any single forward pass.
 * MobileNetV3 with all SE blocks live simultaneously needs ≈ 82 structs. */
//#define MAX_LIVE_TENSORS 128
#define MAX_LIVE_TENSORS 8 // CNN_2D_v1
#define MAX_LIVE_TENSORS 5 // CNN_1D_v1

/* ── Global pool declarations ────────────────────────────────────────────── */

/** Weight pool: layer parameters (weights, biases, BN params).
 *  Defined in arena.c.  Point-of-view of CGRA: treat as ROM after init. */
extern float g_weight_pool[WEIGHT_POOL_FLOATS];

/** Current allocation index into g_weight_pool (bump allocator). */
extern int   g_weight_top;

/** Activation arena: intermediate feature-map storage. */
extern float g_act_arena[ACT_ARENA_FLOATS];

/** Current allocation index into g_act_arena. */
extern int   g_act_top;

/** Pool of Tensor descriptors (struct { n,c,h,w,*data }).
 *  data pointers inside each struct point into g_act_arena. */
extern Tensor g_tensor_pool[MAX_LIVE_TENSORS];

/** Current allocation index into g_tensor_pool. */
extern int    g_tensor_top;

/* ── Allocator functions ─────────────────────────────────────────────────── */

/**
 * weight_alloc  –  Bump-allocate n floats from the weight pool.
 *
 * Called by layer_create functions (conv2d_layer_create, etc.) to obtain
 * storage for weights, biases, and BN parameters.  Never freed; the weight
 * pool is reset implicitly by a full program restart.
 *
 * Aborts with an error message if the pool is exhausted.
 */
float *weight_alloc(int n);

/**
 * act_alloc  –  Bump-allocate n floats from the activation arena.
 *
 * Called by tensor_create() to obtain the data buffer for a new activation
 * tensor.  Not freed individually; call act_arena_reset() between models.
 */
float *act_alloc(int n);

/**
 * act_arena_reset  –  Reset the activation arena and tensor-struct pool.
 *
 * Call this at the start of each model's forward pass.  All Tensor pointers
 * from previous calls become invalid after this call.
 *
 * Note: g_weight_pool is NOT reset; weights accumulate for the entire run.
 */
void act_arena_reset(void);

/**
 * arena_stats  –  Print current pool usage to stdout.
 *
 * Useful for verifying that pool sizes are sufficient and for right-sizing
 * them before a CGRA deployment.
 */
void arena_stats(void);

#endif /* ARENA_H */
