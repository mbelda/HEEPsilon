// Copyright 2024 DeepBindi / EPFL / UPM
// SPDX-License-Identifier: Apache-2.0

/**
 * main.c  --  DeepBindi CNN_1D_v2 inference on X-HEEP (int32 variant).
 *
 * Model: CNN_1D_v2 (PYE / Model 4 in the paper)
 *        "DeepBindi: An End-to-End Fear Detection System
 *         Optimized for Extreme-Edge Deployment"
 *
 * Execution flow:
 *   1. (Optional) Enable FPU -- only if DEEPBINDI_ENABLE_FPU is defined.
 *      Not required for this int32 port.
 *   2. Reset mcycle counter.
 *   3. Run CNN_1D_v2 forward pass on two test samples from test_input.h:
 *        sample 0: index   0, label 0 (NO_FEAR)
 *        sample 1: index  49, label 1 (FEAR)
 *   4. Read cycle count per pass, print output (0 or 1) and timing.
 *   5. Print arena usage statistics.
 *
 * Build targets:
 *   X-HEEP : standard CMake flow; include csr.h and x-heep.h automatically.
 *   PC     : make -f Makefile  (adds -DTARGET_PC; stubs CSR macros).
 *
 * Static SRAM footprint:
 *   Weight pool  : 24000 * 4 B =  93.75 KB  (move to flash for production)
 *   Act arena    :  2048 * 4 B =   8.00 KB
 *   Tensor pool  :    16 structs =  0.38 KB
 *
 * TODO (production):
 *   1. Extract trained int8 weights with extract_weights.py -> weights_cnn_1d_v2.h.
 *   2. Replace seeded_value_int32() in nn_runtime.c with reads from those const arrays.
 *   3. Feed real PPG/EDA/skin-temp feature vector instead of test_input_0[].
 */

#include <stdio.h>
#include "deepbindi_config.h"

/* X-HEEP hardware headers: not available on PC. */

#include "csr.h"
#include "x-heep.h"


#include "cnn_models_c.h"
#include "arena.h"
#include "test_input.h"

/* Enable FPU: FS field of mstatus must be != 0 before any FP instruction.
 * Not needed for this int32 port; define DEEPBINDI_ENABLE_FPU only if you
 * add floating-point operations. */
#define FS_INITIAL 0x01

/* PRINTF macro -- active on FPGA and simulation unless overridden. */
#ifndef PRINTF_IN_FPGA
#define PRINTF_IN_FPGA  1
#endif

#ifndef PRINTF_IN_SIM
#define PRINTF_IN_SIM   1
#endif

#if defined(TARGET_PC)
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif defined(TARGET_SIM) && PRINTF_IN_SIM
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif PRINTF_IN_FPGA && !defined(TARGET_SIM)
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
    #define PRINTF(...)
#endif

int main(void)
{
    Tensor      *out;
    unsigned int cycles;

    /* ---- Hardware initialisation --------------------------------------- */

#ifdef DEEPBINDI_ENABLE_FPU
    /* Enable floating-point unit (only needed if FP operations are used). */
    CSR_SET_BITS(CSR_REG_MSTATUS, (FS_INITIAL << 13));
#endif

    /* Enable and zero the hardware cycle counter. */
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);
    CSR_WRITE(CSR_REG_MCYCLE, 0);

    PRINTF("DeepBindi CNN_1D_v2 on X-HEEP (int32)\r\n");

    /* ---- Sample 0: NO_FEAR (expected label = %d) ----------------------- */

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    PRINTF("--- Sample 0 (expected label=%d / NO_FEAR) ---\r\n", TEST_INPUT_LABEL_0);

    out = run_cnn_1d_v2(test_input_0);

    CSR_READ(CSR_REG_MCYCLE, &cycles);

    PRINTF("Output : %d (%s)\r\n",
           (int)out->data[0],
           out->data[0] ? "FEAR" : "NO_FEAR");
    PRINTF("Cycles : %u\r\n", cycles);

#ifndef DEEPBINDI_SINGLE_SAMPLE
    /* ---- Sample 1: FEAR (expected label = %d) -------------------------- */

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    PRINTF("--- Sample 1 (expected label=%d / FEAR) ---\r\n", TEST_INPUT_LABEL_1);

    out = run_cnn_1d_v2(test_input_1);

    CSR_READ(CSR_REG_MCYCLE, &cycles);

    PRINTF("Output : %d (%s)\r\n",
           (int)out->data[0],
           out->data[0] ? "FEAR" : "NO_FEAR");
    PRINTF("Cycles : %u\r\n", cycles);
#endif

    /* ---- Arena stats --------------------------------------------------- */

    arena_stats();

    return 0;
}
