// Copyright 2024 DeepBindi / EPFL / UPM
// SPDX-License-Identifier: Apache-2.0

/**
 * main.c  --  DeepBindi CNN inference on X-HEEP (int32 variant).
 *
 * Supports three models selected at compile time via -DDEEPBINDI_MODEL=N:
 *   0 (default): CNN_1D_v2     -- baseline, standard 1D convolutions
 *   1:           MobileCNN-1D  -- depthwise-separable convolutions (Option A)
 *   2:           MobileCNN-SE  -- DW+PW + SE channel attention (Option B)
 *
 * All models:
 *   - Input  : (1, 57, 1, 10)  -- 57 features x 10 time frames (WEMAC)
 *   - Output : 0 (NO_FEAR) or 1 (FEAR)
 *   - Binary : no float, no FPU, no heap, no math.h
 *
 * Build targets (see Makefile):
 *   make                  -> CNN_1D_v2     dummy
 *   make run-real         -> CNN_1D_v2     real weights
 *   make mobile1d         -> MobileCNN-1D  dummy
 *   make mobile1d-real    -> MobileCNN-1D  real weights
 *   make mobile_se        -> MobileCNN-SE  dummy
 *   make mobile_se-real   -> MobileCNN-SE  real weights
 *
 * Real weights require:
 *   - python train_mobilecnn_1d.py --model mobile1d   (for models 1, 2)
 *   - python export_mobile_weights.py --model mobile1d (model 1)
 *   - python export_mobile_weights.py --model mobile_se (model 2)
 *   - python export_pytorch_weights.py  (model 0)
 *   - python extract_test_inputs.py     (for test_input.h, all models)
 */

#include <stdio.h>
#include "deepbindi_config.h"

#ifndef TARGET_PC
#  include "csr.h"
#  include "x-heep.h"
#endif

#ifndef DEEPBINDI_MODEL
#  define DEEPBINDI_MODEL 0
#endif

/* Include the correct model header */
#if DEEPBINDI_MODEL == 1 || DEEPBINDI_MODEL == 2
#  include "mobile_models_c.h"
#else
#  include "cnn_models_c.h"
#endif

#include "arena.h"
#include "test_input.h"

#define FS_INITIAL      0x01
#define PRINTF_IN_FPGA  1
#define PRINTF_IN_SIM   0

#if defined(TARGET_PC)
#  define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif defined(TARGET_SIM) && PRINTF_IN_SIM
#  define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif PRINTF_IN_FPGA && !defined(TARGET_SIM)
#  define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
#  define PRINTF(...)
#endif

/* Model name string for the header line */
#if DEEPBINDI_MODEL == 2
#  define MODEL_NAME "MobileCNN-SE-1D"
#elif DEEPBINDI_MODEL == 1
#  define MODEL_NAME "MobileCNN-1D"
#else
#  define MODEL_NAME "CNN_1D_v2"
#endif

/* Forward-pass dispatch */
static Tensor *run_model(const int32_t *input_data) {
#if DEEPBINDI_MODEL == 2
    return run_mobilecnn_se_1d(input_data);
#elif DEEPBINDI_MODEL == 1
    return run_mobilecnn_1d(input_data);
#else
    return run_cnn_1d_v2(input_data);
#endif
}

int main(void)
{
    Tensor      *out;
    unsigned int cycles;

#ifdef DEEPBINDI_ENABLE_FPU
    CSR_SET_BITS(CSR_REG_MSTATUS, (FS_INITIAL << 13));
#endif

    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);
    CSR_WRITE(CSR_REG_MCYCLE, 0);

    PRINTF("DeepBindi %s on X-HEEP (int32)\r\n", MODEL_NAME);

    /* ---- Sample 0: NO_FEAR ------------------------------------------- */

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    PRINTF("--- Sample 0 (expected label=%d / NO_FEAR) ---\r\n",
           TEST_INPUT_LABEL_0);

    out = run_model(test_input_0);

    CSR_READ(CSR_REG_MCYCLE, &cycles);
    PRINTF("Output : %d (%s)\r\n",
           (int)out->data[0], out->data[0] ? "FEAR" : "NO_FEAR");
    PRINTF("Cycles : %u\r\n", cycles);

    /* ---- Sample 1: FEAR ---------------------------------------------- */

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    PRINTF("--- Sample 1 (expected label=%d / FEAR) ---\r\n",
           TEST_INPUT_LABEL_1);

    out = run_model(test_input_1);

    CSR_READ(CSR_REG_MCYCLE, &cycles);
    PRINTF("Output : %d (%s)\r\n",
           (int)out->data[0], out->data[0] ? "FEAR" : "NO_FEAR");
    PRINTF("Cycles : %u\r\n", cycles);

    /* ---- Arena stats -------------------------------------------------- */

    arena_stats();

    return 0;
}
