/**
 * arena.c  –  Static memory pool definitions and allocators.
 *
 * This is the ONLY file that defines static storage.  Everything else
 * (nn_runtime.c, cnn_models_c.c) simply calls weight_alloc() / act_alloc()
 * and never touches malloc/free.
 *
 * Memory map (all in BSS/data segment, no heap):
 *
 *   g_weight_pool  [WEIGHT_POOL_FLOATS * 4 bytes]  ← layer weights (ROM-like)
 *   g_act_arena    [ACT_ARENA_FLOATS   * 4 bytes]  ← activation scratch
 *   g_tensor_pool  [MAX_LIVE_TENSORS   * sizeof(Tensor)]  ← Tensor descriptors
 *
 * Total static footprint ≈ 2 000 000*4 + 500 000*4 + 128*20 ≈ 10 MB.
 * For a single-model CGRA deployment, reduce WEIGHT_POOL_FLOATS and
 * ACT_ARENA_FLOATS to the values printed by arena_stats() after one run.
 */
#include "deepbindi_config.h"  /* logging + fatal error macros – include first */
#include "arena.h"

#include <string.h>

/* ── Global static pools ─────────────────────────────────────────────────── */

/* Weight pool – treat as ROM after weights_init().
 * In an FPGA/CGRA deployment this would be loaded into on-chip BRAM or
 * a dedicated weight buffer with DMA access. */
float g_weight_pool[WEIGHT_POOL_FLOATS];
int   g_weight_top = 0;

/* Activation arena – scratch SRAM for intermediate feature maps.
 * A CGRA ping-pong scheme would split this into two halves and alternate. */
float g_act_arena[ACT_ARENA_FLOATS];
int   g_act_top = 0;

/* Tensor descriptor structs – small (5 ints + pointer = ~24 bytes each). */
Tensor g_tensor_pool[MAX_LIVE_TENSORS];
int    g_tensor_top = 0;

/* ── High-water marks (for sizing guidance) ──────────────────────────────── */
static int g_weight_hwm = 0;  /* max weight_top seen across all runs */
static int g_act_hwm    = 0;  /* max act_top seen during a single model run */

/* ── Allocators ──────────────────────────────────────────────────────────── */

float *weight_alloc(int n) {
    float *ptr;
    if (g_weight_top + n > WEIGHT_POOL_FLOATS) {
        DEEPBINDI_LOG_ERROR(
                "weight_alloc: pool exhausted (need %d, have %d, used %d)\n"
                "  Increase WEIGHT_POOL_FLOATS in arena.h\n",
                n, WEIGHT_POOL_FLOATS - g_weight_top, g_weight_top);
        DEEPBINDI_FATAL("weight pool exhausted");
    }
    ptr = g_weight_pool + g_weight_top;
    g_weight_top += n;
    if (g_weight_top > g_weight_hwm) {
        g_weight_hwm = g_weight_top;
    }
    /* Zero-initialise so the caller can fill selectively */
    memset(ptr, 0, (size_t)n * sizeof(float));
    return ptr;
}

float *act_alloc(int n) {
    float *ptr;
    if (g_act_top + n > ACT_ARENA_FLOATS) {
        DEEPBINDI_LOG_ERROR(
                "act_alloc: arena exhausted (need %d, have %d, used %d)\n"
                "  Increase ACT_ARENA_FLOATS in arena.h\n",
                n, ACT_ARENA_FLOATS - g_act_top, g_act_top);
        DEEPBINDI_FATAL("activation arena exhausted");
    }
    ptr = g_act_arena + g_act_top;
    g_act_top += n;
    if (g_act_top > g_act_hwm) {
        g_act_hwm = g_act_top;
    }
    /* Zero-initialise to match calloc behaviour of the dynamic version */
    memset(ptr, 0, (size_t)n * sizeof(float));
    return ptr;
}

void act_arena_reset(void) {
    /* Discard all activation tensors from the previous forward pass.
     * Weight pool (g_weight_pool) is intentionally NOT reset. */
    g_act_top    = 0;
    g_tensor_top = 0;
}

void arena_stats(void) {
    DEEPBINDI_PRINTF("── Arena usage ─────────────────────────────────────────────\n");
    DEEPBINDI_PRINTF("  Weight pool : %d / %d floats used  (%.1f / %.1f KB)\n",
           g_weight_hwm, WEIGHT_POOL_FLOATS,
           g_weight_hwm * 4.0f / 1024.0f,
           WEIGHT_POOL_FLOATS * 4.0f / 1024.0f);
    DEEPBINDI_PRINTF("  Act arena   : %d / %d floats HWM   (%.1f / %.1f KB)\n",
           g_act_hwm, ACT_ARENA_FLOATS,
           g_act_hwm * 4.0f / 1024.0f,
           ACT_ARENA_FLOATS * 4.0f / 1024.0f);
    /* Use (unsigned) cast + %u to avoid %zu, which is unsupported by newlib-nano */
    DEEPBINDI_PRINTF("  Tensor pool : %d / %d structs used (%u / %u bytes)\n",
           g_tensor_top, MAX_LIVE_TENSORS,
           (unsigned)g_tensor_top * (unsigned)sizeof(Tensor),
           (unsigned)MAX_LIVE_TENSORS * (unsigned)sizeof(Tensor));
    DEEPBINDI_PRINTF("────────────────────────────────────────────────────────────\n");
}
