/**
 * cnn_models_c.h  --  Forward-pass entry point for CNN_1D_v2 (PYE).
 *                     int32 variant: no float, no math.h.
 *
 * CNN_1D_v2 architecture (Model 4 in the DeepBindi paper):
 *   Input    : (1, 57, 1, 10)  --  57 features * 10 time frames
 *   Conv1    : Conv1d(57->32, k=5, pad=1) -> BN -> ReLU -> MaxPool1d(2)
 *   Conv2    : Conv1d(32->64, k=5, pad=1) -> BN -> ReLU -> MaxPool1d(2)
 *   Classify : Flatten -> Dense(64->1) -> Threshold(0)
 *   Output   : (1, 1, 1, 1)  --  0 (NO_FEAR) or 1 (FEAR)
 *
 * Note: FC input is 64 (not 128) because with num_frames=10 and pad=1 the
 * spatial width after pool2 is 1, giving 64*1=64 features.  The paper's 128
 * corresponds to num_frames=14.
 *
 * Parameter count (32-bit int, dummy weights):
 *   Conv1 weights + bias : 9120 + 32            = 9 152 words = 35.75 KB
 *   BN1 scale + offset   :   32 + 32            =    64 words =  0.25 KB
 *   Conv2 weights + bias : 10240 + 64           = 10 304 words = 40.25 KB
 *   BN2 scale + offset   :   64 + 64            =   128 words =  0.50 KB
 *   FC  weights + bias   :   64 + 1             =    65 words =  0.25 KB
 *   Total                                        19 713 words = ~77 KB
 */

#ifndef CNN_MODELS_C_H
#define CNN_MODELS_C_H

#include <stdint.h>
#include "nn_runtime.h"

/**
 * run_cnn_1d_v2  --  Run one forward pass of CNN_1D_v2.
 *
 * @param input_data  Pointer to 570 int32_t input values in channel-first
 *                    layout: data[ch * 10 + t] for channel ch, time step t.
 *                    Pass NULL to use internally generated dummy data.
 *
 * Internally calls act_arena_reset(), then allocates all intermediate
 * tensors from g_act_arena.  The returned Tensor pointer is valid until
 * the next call to act_arena_reset().
 *
 * Returns: pointer to output tensor (1, 1, 1, 1);
 *          data[0] == 1 => FEAR, data[0] == 0 => NO_FEAR.
 */
Tensor *run_cnn_1d_v2(const int32_t *input_data);

#endif /* CNN_MODELS_C_H */
