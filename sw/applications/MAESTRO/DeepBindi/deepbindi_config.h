/**
 * deepbindi_config.h  --  X-HEEP / PC adaptation of the DeepBindi compile-time config.
 *
 * Key points:
 *   - DEEPBINDI_PRINTF maps to printf (UART on X-HEEP, stdout on PC).
 *   - All values in this port are int32_t; no %f anywhere.
 *   - DEEPBINDI_FATAL: spins forever on bare metal; calls exit(1) on PC.
 *
 * FPU control (X-HEEP only):
 *   This int32 port does NOT require the FPU.  Define DEEPBINDI_ENABLE_FPU
 *   (e.g. -DDEEPBINDI_ENABLE_FPU) only if you add floating-point code and
 *   need to enable mstatus.FS before the first FP instruction.
 *
 * PC testing:
 *   Build with -DTARGET_PC to stub all CSR macros and use stdlib exit().
 *   The Makefile sets this automatically.
 */

#ifndef DEEPBINDI_CONFIG_H
#define DEEPBINDI_CONFIG_H

#include <stdio.h>

/* Verbose layer / arena trace.
 * Define -DDEEPBINDI_TRACE_LAYERS at build time to enable per-layer tensor
 * shape+checksum prints (nn_runtime.c) and arena usage stats (arena.c).
 * The final result output in main.c (label, cycles) is always printed. */
#ifdef DEEPBINDI_TRACE_LAYERS
#  define DEEPBINDI_PRINTF(...)     printf(__VA_ARGS__)
#  define DEEPBINDI_LOG_ERROR(...)  printf(__VA_ARGS__)
#else
#  define DEEPBINDI_PRINTF(...)     ((void)0)
#  define DEEPBINDI_LOG_ERROR(...)  ((void)0)
#endif

#ifdef DEEPBINDI_TRACE_LAYERS
#  define DEEPBINDI_TRACE(...)      printf(__VA_ARGS__)
#else
#  define DEEPBINDI_TRACE(...)
#endif

/* Fatal: spin forever on bare metal; exit(1) on PC. */
#ifdef TARGET_PC
#  include <stdlib.h>
#  ifndef DEEPBINDI_FATAL
#    define DEEPBINDI_FATAL(msg)  do { printf("FATAL: %s\r\n", (msg)); exit(1); } while(0)
#  endif
#else
#  ifndef DEEPBINDI_FATAL
#    define DEEPBINDI_FATAL(msg)  do { printf("FATAL: %s\r\n", (msg)); for(;;){} } while(0)
#  endif
#endif

/* CSR stubs for PC testing (no RISC-V CSRs on the host). */
#ifdef TARGET_PC
#  define CSR_SET_BITS(reg, val)    ((void)0)
#  define CSR_CLEAR_BITS(reg, val)  ((void)0)
#  define CSR_WRITE(reg, val)       ((void)0)
#  define CSR_READ(reg, ptr)        (*(ptr) = 0u)
#  define CSR_REG_MSTATUS           0
#  define CSR_REG_MCOUNTINHIBIT     0
#  define CSR_REG_MCYCLE            0
#endif

#endif /* DEEPBINDI_CONFIG_H */
