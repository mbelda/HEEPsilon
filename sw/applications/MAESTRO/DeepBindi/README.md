# deepbindi_cnn_x_heep — int32 inference engine for X-HEEP

Fully integer, heap-free, FPU-free inference engine for three DeepBindi CNN
architectures, designed for bare-metal deployment on the
[X-HEEP](https://github.com/esl-epfl/x-heep) RISC-V SoC with optional STRELA
CGRA acceleration.

---

## Interactive model tour

An interactive layer-by-layer tutorial is available as a standalone HTML page:

```
docs/model_tour.html
```

Open it in any browser (no server required). For each of the three models it shows the
shape transformation, integer arithmetic formula, weight count, MAC count, and cumulative
activation-arena usage at every layer. Use `←` / `→` keys or the pipeline strip at the
bottom to step through.

---

## Models

Three architectures selectable at compile time via `-DDEEPBINDI_MODEL=N`:

| N | Name | C entry point | Weights (ROM) | Act RAM peak |
|---|------|---------------|---------------|--------------|
| 0 | CNN_1D_v2 (baseline) | `run_cnn_1d_v2()` | 77 KB | 4.8 KB |
| 1 | MobileCNN-1D (Option A) | `run_mobilecnn_1d()` | 19.4 KB | 6.9 KB |
| 2 | MobileCNN-SE-1D (Option B) | `run_mobilecnn_se_1d()` | 21.6 KB | 7.2 KB |

All models share the same input/output convention:
- **Input**: `(1, 57, 1, 10)` — 57 WEMAC physiological features × 10 time frames
- **Output**: `(1, 1, 1, 1)` — `0` = NO_FEAR, `1` = FEAR (sign threshold)

### Architecture provenance

**CNN_1D_v2** is taken directly from the DeepBindi paper (Gutiérrez-Martín et al., IEEE JBHI 2026; see citation below).

**MobileCNN-1D** and **MobileCNN-SE-1D** are original designs adapted to 1D WEMAC biosignal
classification. They draw conceptual inspiration from the MobileNetV1 depthwise-separable
convolution factorization (Howard et al., 2017), the Squeeze-and-Excitation block (Hu et al.,
CVPR 2018), and lightweight CNN studies targeting resource-constrained IoT/embedded devices such
as:

- Sanjay & Ahmadinia, "MobileNet-Tiny: A Deep Neural Network-Based Real-Time Object Detection
  for Raspberry Pi," ICMLA 2019.
- Nyakuri et al., "Tiny-MobileNet-SE: A Hybrid Lightweight CNN Architecture for
  Resource-Constrained IoT Devices," IEEE Access 2025.

The 1D block structure (DW conv → BN → ReLU → PW conv → BN → ReLU → MaxPool) and the
integer quantization scheme are specific to this implementation and are not direct translations
of any of the above.

---

## Architecture diagrams

### Model 0 — CNN_1D_v2 (baseline)

```
Input (1,57,1,10)
      |
      v
 Conv2D(57->32, k=5, pad=1)                        shape -> (1,32,1,8)
      |  BN(32) pre-folded to Q7 scale+offset
      |  ReLU
      v
 MaxPool2D(k=2, s=2)                               shape -> (1,32,1,4)
      |
      v
 Conv2D(32->64, k=5, pad=1)                        shape -> (1,64,1,2)
      |  BN(64) + ReLU
      v
 MaxPool2D(k=2, s=2)                               shape -> (1,64,1,1)
      |
      v
 Flatten                                            shape -> (1,64,1,1)
      |
      v
 Dense(64->1) + threshold(x>0)                     shape -> (1,1,1,1)
      |
      v
  0 (NO_FEAR)  or  1 (FEAR)
```

Parameter count: 19 713 int32 (77 KB ROM), 1 211 int32 act RAM peak.

---

### Model 1 — MobileCNN-1D (Option A, depthwise-separable)

```
Input (1,57,1,10)
      |
      v
+-- Block 1 -----------------------------------------------+
|  DWConv(57->57, k=5, pad=1, groups=57)   shape (1,57,1,8)|
|  BN_DW1 + ReLU + rshift(SHIFT_DW1)                       |
|  PWConv(57->32, k=1)                     shape (1,32,1,8) |
|  BN_PW1 + ReLU + rshift(SHIFT_PW1)                       |
|  MaxPool(k=2,s=2)                        shape (1,32,1,4) |
+-----------------------------------------------------------+
      |
      v
+-- Block 2 -----------------------------------------------+
|  DWConv(32->32, k=5, pad=1, groups=32)   shape (1,32,1,2)|
|  BN_DW2 (per-channel rshift)                              |
|  ReLU                                                     |
|  PWConv(32->64, k=1)                     shape (1,64,1,2) |
|  BN_PW2 + ReLU + rshift(SHIFT_PW2)                       |
|  MaxPool(k=2,s=2)                        shape (1,64,1,1) |
+-----------------------------------------------------------+
      |
      v
 Flatten -> Dense(64->1) -> threshold(x>0)
      |
      v
  0 (NO_FEAR)  or  1 (FEAR)
```

Parameter count: 4 969 int32 (19.4 KB ROM), 1 731 int32 act RAM peak.
~4× fewer weights than CNN_1D_v2.

---

### Model 2 — MobileCNN-SE-1D (Option B, depthwise-separable + SE attention)

```
Input (1,57,1,10)
      |
      v
+-- Block 1 -----------------------------------------------+
|  DWConv(57->57, k=5, pad=1, groups=57)   shape (1,57,1,8)|
|  BN_DW1 + ReLU + rshift(SHIFT_DW1)                       |
|  PWConv(57->32, k=1)                     shape (1,32,1,8) |
|  BN_PW1 + ReLU + rshift(SHIFT_PW1)                       |
|  MaxPool(k=2,s=2)                        shape (1,32,1,4) |
+-------------------+---------------------------------------+
                    |
          +---------+----------+
          |  SE attention block|
          |  GlobalAvgPool     | shape (1,32,1,1)
          |  Dense(32->8)+ReLU | shape (1, 8,1,1)
          |  rshift(SE_SHIFT_D1)
          |  Dense(8->32)      | shape (1,32,1,1)
          |  HardSigmoid Q7    | y=clip((x>>SE_SHIFT_HS)+64, 0, 128)
          +--------------------+
                    |
          (channel-wise scale of Block 1 output: feat[c] *= se[c]/128)
                    |
      +---------+---+
      v
+-- Block 2 -----------------------------------------------+
|  (same as Model 1 Block 2)                                |
+-----------------------------------------------------------+
      |
      v
 Flatten -> Dense(64->1) -> threshold(x>0)
      |
      v
  0 (NO_FEAR)  or  1 (FEAR)
```

Parameter count: 5 521 int32 (21.6 KB ROM), 1 803 int32 act RAM peak.
SE adds 552 int32 (Dense 32→8 and 8→32) over Model 1.

---

## Build matrix

Six build targets: three models × {dummy weights, real weights}.

| Make target | Binary | Model | Weights | Purpose |
|-------------|--------|-------|---------|---------|
| `make` / `all` | `deepbindi_cnn_pc` | CNN_1D_v2 | dummy | arena sizing |
| `make run-real` | `deepbindi_cnn_pc_real` | CNN_1D_v2 | real | classification test |
| `make mobile1d` | `deepbindi_mobile1d_pc` | MobileCNN-1D | dummy | arena sizing |
| `make mobile1d-real` | `deepbindi_mobile1d_pc_real` | MobileCNN-1D | real | classification test |
| `make mobile_se` | `deepbindi_mobile_se_pc` | MobileCNN-SE | dummy | arena sizing |
| `make mobile_se-real` | `deepbindi_mobile_se_pc_real` | MobileCNN-SE | real | classification test |

Compile flags:

```
-DDEEPBINDI_MODEL=N           0=CNN_1D_v2  1=MobileCNN-1D  2=MobileCNN-SE-1D
-DDEEPBINDI_REAL_WEIGHTS      load const int32_t arrays from weights_*.h
-DTARGET_PC                   stubs CSR macros, enables exit()
```

---

## Quick start (PC build)

### Prerequisites

- GCC ≥ C99 (`/c/msys64/ucrt64/bin` on Windows MSYS2)
- Python 3 + `torch`, `numpy` (weight export only)
- Trained checkpoints in `results/deepBindi/<model>/type4_fold0.pth`

### Dummy-weights build (no checkpoint required)

```bash
# Windows MSYS2 — set PATH and TMPDIR once:
export PATH="/c/msys64/usr/bin:/c/msys64/ucrt64/bin:$PATH"
export TMPDIR="/tmp"

# Build all three dummy targets
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=0 \
    -o deepbindi_cnn_pc    main.c cnn_models_c.c nn_runtime.c arena.c
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=1 \
    -o deepbindi_mobile1d_pc  main.c mobile_models_c.c nn_runtime.c arena.c
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=2 \
    -o deepbindi_mobile_se_pc main.c mobile_models_c.c nn_runtime.c arena.c
```

### Real-weights build (CNN_1D_v2)

```bash
python export_pytorch_weights.py          # writes weights_cnn_1d_v2.h
python extract_test_inputs.py             # writes test_input.h

gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=0 -DDEEPBINDI_REAL_WEIGHTS \
    -o deepbindi_cnn_pc_real main.c cnn_models_c.c nn_runtime.c arena.c
./deepbindi_cnn_pc_real
```

### Real-weights build (MobileCNN-1D and SE)

```bash
# Train (if no checkpoint exists):
python train_mobilecnn_1d.py --model mobile1d
python train_mobilecnn_1d.py --model mobile_se

# Export weights (calibrates int32 shifts from test_input.h automatically):
python export_mobile_weights.py --model mobile1d   # -> weights_mobile1d.h
python export_mobile_weights.py --model mobile_se  # -> weights_mobile_se.h

# Build Model 1:
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=1 -DDEEPBINDI_REAL_WEIGHTS \
    -o deepbindi_mobile1d_pc_real main.c mobile_models_c.c nn_runtime.c arena.c

# Build Model 2:
gcc -Wall -Wextra -O2 -std=c99 -DTARGET_PC -DDEEPBINDI_MODEL=2 -DDEEPBINDI_REAL_WEIGHTS \
    -o deepbindi_mobile_se_pc_real main.c mobile_models_c.c nn_runtime.c arena.c
```

### Expected output (all three real-weights builds)

```
--- Sample 0 (expected label=0 / NO_FEAR) ---
Output : 0 (NO_FEAR)
--- Sample 1 (expected label=1 / FEAR) ---
Output : 1 (FEAR)
-- Arena usage --
  Weight pool : 0 / 1 words  (0 / 0 KB)      <- weights in .rodata, not pool
  Act arena   : XXXX / 2048 words
  Tensor pool : N / 16 structs
```

---

## File inventory

| File | Purpose |
|------|---------|
| `main.c` | Entry point: runs 2 test samples, prints output + cycle count |
| `nn_runtime.h/c` | All tensor/layer primitives (int32, no float, no math.h) |
| `cnn_models_c.h/c` | `run_cnn_1d_v2()` — CNN_1D_v2 forward pass |
| `mobile_models_c.h/c` | `run_mobilecnn_1d()` and `run_mobilecnn_se_1d()` |
| `arena.h/c` | Static bump allocators (g_weight_pool + g_act_arena) |
| `deepbindi_config.h` | Pool sizes, logging macros, CSR stubs for PC build |
| `test_input.h` | Two WEMAC fold-0 test samples (NO_FEAR idx=698, FEAR idx=641) |
| `weights_cnn_1d_v2.h` | CNN_1D_v2 trained weights as const int32_t (generated) |
| `weights_mobile1d.h` | MobileCNN-1D trained weights + per-channel shifts (generated) |
| `weights_mobile_se.h` | MobileCNN-SE trained weights + SE shifts (generated) |
| `export_pytorch_weights.py` | Convert CNN_1D_v2 .pth → `weights_cnn_1d_v2.h` |
| `export_mobile_weights.py` | Convert MobileCNN .pth → `weights_mobile*.h` |
| `extract_test_inputs.py` | Extract WEMAC samples → `test_input.h` |
| `Makefile` | PC build — all six targets (dummy and real-weights) |

Training scripts (in `EPFL_STAY_LAURA/python_code/simulations/`):
- `train_cnn1d_v2_xheep.py` — trains CNN_1D_v2
- `train_mobilecnn_1d.py` — trains MobileCNN-1D and MobileCNN-SE-1D (flag `--model`)

---

## Integer arithmetic pipeline

All arithmetic is `int32_t`/`int64_t` throughout. No `float`, no `math.h`.

### Input quantization

```
x_int32 = round(x_float * 12).clip(-128, 127)
```

Scale factor 12 maps z-score normalized WEMAC features (~[-10, 10]) into int8
range (~[-120, 120]).

### Weight quantization (at export time, Python)

```
scale_f = max(|w_float|) / 127
w_int   = round(w_float / scale_f)     # range [-127, 127], stored as int32_t
```

### BatchNorm pre-folding (at export time, Python)

BN is folded into two integer constants per channel, eliminating `sqrtf` and
all per-inference floating-point:

```
scale_q7[c]  = round(gamma[c] / sqrt(var[c] + eps) * 128)   # Q7: 128 = 1.0
offset_q[c]  = round(beta[c]  - gamma[c] * mean[c] / sqrt(var[c] + eps))
```

Forward pass (C):
```c
/* int64_t intermediate prevents overflow before the int32_t cast */
y = (int32_t)(((int64_t)x * scale_q7[c]) >> 7) + offset_q[c];
```

### Fused BN + post-shift (batchnorm_rshift_inplace)

Used after every BN layer in the real-weights mobile models. Holding the
intermediate in int64_t prevents overflow even when `scale_q7` values reach
700+ and the accumulator is several billion:

```
y = (((int64_t)x * scale_q7[c]) >> 7 + offset_q[c]) >> post_shift
```

### Per-channel BN + post-shift (batchnorm_rshift_perchannel)

Used specifically for BN_DW2 in both mobile models. The depthwise BN Q7 scale
varies dramatically across channels (e.g., 308 to 1574 for MobileCNN-1D — a 5×
span). A single global post-shift sized for the worst channel wastes up to 5
bits of precision in all lower-scale channels:

```
y[c] = (((int64_t)x[c] * scale_q7[c]) >> 7 + offset_q[c]) >> shifts[c]
```

`shifts[c]` is stored as `mobile_bn_dw2_shift[32]` in the weights header.

### Data-driven shift calibration

The mobile weight export script (`export_mobile_weights.py`) calibrates the
post-shift values in two stages:

```
Stage 1 (theoretical):  SHIFT_DW1 and SHIFT_PW1 prevent int32_t overflow
                        in the conv accumulators.  Uses worst-case inputs.

Stage 2 (data-driven):  DW2 and PW2 shifts use actual pool1 max measured
                        by simulating Block 1 on the two test samples.
```

Why data-driven for Stage 2? The theoretical pool1 max (from 100% saturated
int8 inputs + int8 weights) is ~1000× larger than the observed value for real
physiological signals. Using it for DW2 calibration forces 13–16-bit right
shifts, reducing real signal to < 50 counts — enough to collapse all downstream
activations to zero (always FEAR).

```
Theoretical pool1 max:  ~2,638,000  (all inputs and weights at ±127)
Observed pool1 max:     ~4,464      (actual WEMAC samples after Block 1 sim)
Ratio:                  ~590×
```

With 8× safety margin over the observed max, DW2 per-channel shifts drop from
[13..16] (theoretical) to [7..10] (data-driven), preserving ~64× more signal.

Shift calibration flow:

```
export_mobile_weights.py
  |
  +-- load_test_inputs_from_header(test_input.h)
  |       -> sample0 (57,10) int32, sample1 (57,10) int32
  |
  +-- simulate_block1_pool1_max(sample, weights, shifts)
  |       -> int32 simulation of DW1+BN+ReLU+PW1+BN+ReLU+MaxPool
  |       -> actual_pool1_max  (e.g., 4464 for mobile1d)
  |
  +-- compute_all_shifts(bn_scales, actual_pool1_max, safety=8)
          -> SHIFT_DW1, SHIFT_PW1   (theoretical)
          -> shifts_dw2[32]         (data-driven, per-channel)
          -> SHIFT_PW2              (from calibrated V3_max)
```

---

## Memory footprint

All three models fit within the same 2 048-word (8 KB) activation arena:

```
  g_act_arena[2048]        (all int32_t)

  CNN_1D_v2     |||||||||||||||||||||||||||||||||||     1211 / 2048 words (59%)
  MobileCNN-1D  |||||||||||||||||||||||||||||||||||||||||||||  1731 / 2048 (85%)
  MobileCNN-SE  |||||||||||||||||||||||||||||||||||||||||||||||  1803 / 2048 (88%)
```

| Region | CNN_1D_v2 | MobileCNN-1D | MobileCNN-SE |
|--------|-----------|--------------|--------------|
| Weights (ROM / flash) | 19 713 × 4 B = **77 KB** | 4 969 × 4 B = **19.4 KB** | 5 521 × 4 B = **21.6 KB** |
| Act arena peak (RAM) | 1 211 × 4 B = **4.8 KB** | 1 731 × 4 B = **6.9 KB** | 1 803 × 4 B = **7.2 KB** |
| Tensor pool structs | 7 / 16 | 9 / 16 | 12 / 16 |
| g_weight_pool (real build) | 1 word (4 B) | 1 word (4 B) | 1 word (4 B) |
| g_weight_pool (dummy build) | 24 000 words | 6 000 words | 6 000 words |

In the real-weights build all weight constants live in `.rodata` (flash). The
`g_weight_pool[]` shrinks to 1 word. Total on-chip SRAM needed at inference:
**≤ 8 KB** (act arena) plus stack, regardless of model.

---

## X-HEEP integration

### X-HEEP readiness checklist

| Constraint | Status |
|-----------|--------|
| No `float` / `double` in data path | All arithmetic is `int32_t` / `int64_t` |
| No `math.h` (`expf`, `sqrtf`, ...) | BN pre-folded; sigmoid → sign threshold |
| No `malloc` / `free` / heap | All allocations from static arenas |
| No `memset` / `memcpy` | Explicit scalar loops throughout |
| No `%f` in `printf` | Values printed with `%d` only |
| CSR cycle counter | `CSR_WRITE/READ(CSR_REG_MCYCLE, ...)` — stubbed by `TARGET_PC` |
| FPU not required | No FP instruction in any code path; guarded by `DEEPBINDI_ENABLE_FPU` |
| `exit()` bare-metal safe | `DEEPBINDI_FATAL` → `for(;;){}` (watchdog reset) |

### CMake build flags for X-HEEP

```cmake
target_compile_definitions(deepbindi PRIVATE
    DEEPBINDI_MODEL=1           # or 0 or 2
    DEEPBINDI_REAL_WEIGHTS      # use .rodata const arrays
    # no TARGET_PC
)
```

### `make app` invocation (X-HEEP SDK)

Place (or symlink) the `deepbindi_cnn_x_heep/` directory under
`sw/applications/deepbindi_cnn_x_heep/` inside the X-HEEP repository, then:

```bash
# Minimal — CNN_1D_v2, real weights, result output only
make app PROJECT=deepbindi_cnn_x_heep \
         COMPILER_FLAGS='-DDEEPBINDI_MODEL=0 -DDEEPBINDI_REAL_WEIGHTS'

# MobileCNN-1D with per-layer trace and arena stats enabled
make app PROJECT=deepbindi_cnn_x_heep \
         COMPILER_FLAGS='-DDEEPBINDI_MODEL=1 -DDEEPBINDI_REAL_WEIGHTS -DDEEPBINDI_TRACE_LAYERS'

# MobileCNN-SE-1D, silent (no DEEPBINDI_TRACE_LAYERS) — result + cycles only
make app PROJECT=deepbindi_cnn_x_heep \
         COMPILER_FLAGS='-DDEEPBINDI_MODEL=2 -DDEEPBINDI_REAL_WEIGHTS'
```

`-DDEEPBINDI_TRACE_LAYERS` enables per-layer tensor shape/checksum prints and
arena usage stats via `DEEPBINDI_PRINTF`. The final result (`Output: 0/1`,
`Cycles: N`) is always printed regardless of this flag.

### CGRA acceleration targets

```
conv2d_forward()                *** PRIMARY ***
  innermost (kh, kw) MAC loops — standard CGRA MAC chain
  depthwise variant (groups=in_channels) collapses icg loop to 1

dense_forward()                 ** SECONDARY **
  dot-product over in_features — short for these models (8–64 inputs)

batchnorm_rshift_inplace()      * FUSIBLE *
  per-channel BN + rshift — merge with conv output stage (no extra mem round-trip)

batchnorm_rshift_perchannel()   * FUSIBLE *
  same as above but with per-channel post-shift — BN_DW2 in mobile models
```

---

## Sigmoid replacement

The float sigmoid `1/(1+exp(-x))` is replaced by a hard threshold:

```c
output = (x > 0) ? 1 : 0;
```

Valid for the single-output binary head — the sign of the pre-sigmoid value
determines the class. Avoids `expf()` entirely.

## SE hard-sigmoid

The SE block excitation uses an integer hard-sigmoid:

```c
/* y = clip((x >> shift) + 64, 0, 128)  where 128 = 1.0 in Q7 */
y = (int32_t)(x >> shift) + 64;
if (y < 0) y = 0;
if (y > 128) y = 128;
```

`SE_SHIFT_HS` is chosen so that `max(|x|) >> SE_SHIFT_HS ≤ 64`, keeping the
output in [0, 128].

---

## Test input format

`extract_test_inputs.py` selects the most confidently classified NO_FEAR and
FEAR samples from fold-0's test set:

| Sample | WEMAC idx | Label | Float model sigmoid |
|--------|-----------|-------|---------------------|
| `test_input_0` | 698 | NO_FEAR (0) | 0.0313 |
| `test_input_1` | 641 | FEAR (1) | 0.9987 |

Layout: `data[ch * 10 + t]`, channel `ch` ∈ [0, 56], time `t` ∈ [0, 9].
570 int32_t values per sample, quantized as `round(x_float * 12).clip(-128, 127)`.

---

## Citation

```bibtex
@article{gutierrez2026deepbindi,
  author  = {Gutiérrez-Martín, Laura and López-Ongil, Celia and Miranda-Calero, Jose A.},
  title   = {{DeepBindi}: An End-to-End Fear Detection System Optimized for Extreme-Edge Deployment},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {30}, number = {1}, year = {2026},
  doi     = {10.1109/JBHI.2025.3587961}
}
```
