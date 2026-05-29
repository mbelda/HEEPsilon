/**
 * main.c  –  Demo driver for the static CNN inference port.
 *            STATIC VERSION: no malloc/free anywhere in the program.
 *
 * This driver:
 *   1. Runs all 10 model forward passes sequentially.
 *   2. Prints output shape, checksum, and first values for each model
 *      (same format as c_port/main.c – checksums must match).
 *   3. After all models, prints memory usage statistics so you can
 *      right-size the static pools for a single-model CGRA deployment.
 *
 * Memory model
 * ────────────
 * The program has exactly three static regions (no heap):
 *
 *   g_weight_pool  [2 000 000 × 4 B = 7.6 MB]  weights for all 10 models
 *   g_act_arena    [  500 000 × 4 B = 1.9 MB]  activation scratch (per model)
 *   g_tensor_pool  [      128 × ~24 B ≈ 3 kB]  Tensor descriptors (per model)
 *
 * For a single-model CGRA deployment, read the "Arena usage" table printed
 * at the end and reduce WEIGHT_POOL_FLOATS / ACT_ARENA_FLOATS accordingly.
 * For example, deploying only CNN_1D_v1 (PYD) needs < 100 kB of static RAM.
 *
 * Validation
 * ──────────
 * The checksums printed here must match those from c_port/deepbindi_c_demo.
 * Use this as the software reference when validating a CGRA implementation:
 *   make run > static_out.txt
 *   cd ../c_port && make run > dyn_out.txt
 *   diff static_out.txt dyn_out.txt   # should be identical
 */

#include "deepbindi_config.h"  /* logging + fatal error macros – include first */
#include "cnn_models_c.h"
#include "arena.h"
#include "csr.h"

int main(void) {
    Tensor *out;

    // Enable cycle counting
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);
    // Reset cycle counter
    CSR_WRITE(CSR_REG_MCYCLE, 0);

    DEEPBINDI_PRINTF("DeepBindi static CNN inference demo (no malloc)\n");
    DEEPBINDI_PRINTF("================================================\n");

    /* Each run_*() begins with act_arena_reset() internally, so the order
     * of calls does not matter for correctness. */
/*
    out = run_cnn_2d_v1();
    tensor_print_values("CNN_2D_v1              ", out, 4);

    out = run_cnn_2d_v2();
    tensor_print_values("CNN_2D_v2              ", out, 4);

    out = run_cnn_2d_v3();
    tensor_print_values("CNN_2D_v3              ", out, 4);
*/
    out = run_cnn_1d_v1();
    tensor_print_values("CNN_1D_v1              ", out, 4);
/*
    out = run_cnn_1d_v2();
    tensor_print_values("CNN_1D_v2              ", out, 4);

    out = run_cnn_1d_v3();
    tensor_print_values("CNN_1D_v3              ", out, 4);

    out = run_mobilenet_v3_custom();
    tensor_print_values("MobileNetV3Custom      ", out, 4);

    out = run_cnn_1d_tensorflow_sigmoid();
    tensor_print_values("CNN_1d_tf_sigmoid      ", out, 4);

    out = run_cnn_1d_tensorflow_softmax();
    tensor_print_values("CNN_1d_tf_softmax      ", out, 4);

    out = run_cnn_2d_tensorflow_softmax();
    tensor_print_values("CNN_2d_tf_softmax      ", out, 4);
    */

    DEEPBINDI_PRINTF("\n");
    /* Print pool usage so users can right-size for deployment */
    arena_stats();
    DEEPBINDI_PRINTF("\nNote: act arena and tensor pool are reused between models.\n");
    DEEPBINDI_PRINTF("      The weight pool accumulates across all 10 model runs.\n");
    DEEPBINDI_PRINTF("      For single-model deployment, only that model's weight\n");
    DEEPBINDI_PRINTF("      count and its act arena high-water mark are needed.\n");

    return 0;
}
