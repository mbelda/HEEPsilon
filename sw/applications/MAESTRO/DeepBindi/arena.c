/**
 * arena.c  --  Static memory pool definitions and allocators.
 *              X-HEEP / int32 variant: no memset, no memcpy, no math.h.
 */
#include "deepbindi_config.h"
#include "arena.h"

/* ---- Global static pools ---------------------------------------------- */

int32_t g_weight_pool[WEIGHT_POOL_WORDS];
int     g_weight_top = 0;

int32_t g_act_arena[ACT_ARENA_WORDS];
int     g_act_top = 0;

Tensor  g_tensor_pool[MAX_LIVE_TENSORS];
int     g_tensor_top = 0;

static int g_weight_hwm = 0;
static int g_act_hwm    = 0;

/* ---- Allocators -------------------------------------------------------- */

int32_t *weight_alloc(int n) {
    int32_t *ptr;
    int i;
    if (g_weight_top + n > WEIGHT_POOL_WORDS) {
        DEEPBINDI_LOG_ERROR(
            "weight_alloc: pool exhausted (need %d, used %d, cap %d)\r\n",
            n, g_weight_top, WEIGHT_POOL_WORDS);
        DEEPBINDI_FATAL("weight pool exhausted");
    }
    ptr = g_weight_pool + g_weight_top;
    g_weight_top += n;
    if (g_weight_top > g_weight_hwm) {
        g_weight_hwm = g_weight_top;
    }
    for (i = 0; i < n; ++i) {
        ptr[i] = 0;
    }
    return ptr;
}

int32_t *act_alloc(int n) {
    int32_t *ptr;
    int i;
    if (g_act_top + n > ACT_ARENA_WORDS) {
        DEEPBINDI_LOG_ERROR(
            "act_alloc: arena exhausted (need %d, used %d, cap %d)\r\n",
            n, g_act_top, ACT_ARENA_WORDS);
        DEEPBINDI_FATAL("activation arena exhausted");
    }
    ptr = g_act_arena + g_act_top;
    g_act_top += n;
    if (g_act_top > g_act_hwm) {
        g_act_hwm = g_act_top;
    }
    for (i = 0; i < n; ++i) {
        ptr[i] = 0;
    }
    return ptr;
}

void act_arena_reset(void) {
    g_act_top    = 0;
    g_tensor_top = 0;
}

void weight_arena_reset(void) {
    g_weight_top = 0;
    /* NOTE: only safe when all weights are dummy (seeded generation).
     * Do not call this after loading trained const weights -- those live
     * outside the pool and will not be regenerated. */
}

void arena_stats(void) {
    DEEPBINDI_PRINTF("-- Arena usage --\r\n");
    DEEPBINDI_PRINTF("  Weight pool : %d / %d words  (%d / %d KB)\r\n",
           g_weight_hwm, WEIGHT_POOL_WORDS,
           (g_weight_hwm * 4) / 1024,
           (WEIGHT_POOL_WORDS * 4) / 1024);
    DEEPBINDI_PRINTF("  Act arena   : %d / %d words  (%d / %d KB)\r\n",
           g_act_hwm, ACT_ARENA_WORDS,
           (g_act_hwm * 4) / 1024,
           (ACT_ARENA_WORDS * 4) / 1024);
    DEEPBINDI_PRINTF("  Tensor pool : %d / %d structs\r\n",
           g_tensor_top, MAX_LIVE_TENSORS);
    DEEPBINDI_PRINTF("-----------------\r\n");
}
