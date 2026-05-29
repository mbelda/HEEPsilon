/**
 * deepbindi_config.h  –  Compile-time configuration for DeepBindi CNN inference.
 *
 * This header is the single place to adapt the library to a target platform.
 * Include it first in every translation unit that uses logging or error handling.
 *
 * ── Logging ──────────────────────────────────────────────────────────────────
 *
 *   Define DEEPBINDI_ENABLE_LOGGING before including this header (or pass
 *   -DDEEPBINDI_ENABLE_LOGGING to the compiler) to enable all printf/fprintf
 *   output.  Leave it undefined for a fully silent embedded deployment.
 *
 *   Example (Makefile):
 *     CFLAGS += -DDEEPBINDI_ENABLE_LOGGING     # host / debug build
 *   or leave it absent for a production embedded build.
 *
 * ── Fatal error handler ───────────────────────────────────────────────────────
 *
 *   DEEPBINDI_FATAL(msg) is called on unrecoverable errors (shape mismatches,
 *   pool overflow).  It must not return.
 *
 *   Default behaviour:
 *     - With logging enabled  → print message + exit(1)   (host)
 *     - Without logging       → infinite loop              (embedded trap / WDT reset)
 *
 *   Override for your target by defining DEEPBINDI_FATAL before including this
 *   header.  Examples:
 *
 *     // ARM Cortex-M: trigger a breakpoint / hard-fault
 *     #define DEEPBINDI_FATAL(msg)  do { __BKPT(0); for(;;){} } while(0)
 *
 *     // RISC-V: illegal instruction trap
 *     #define DEEPBINDI_FATAL(msg)  do { __asm__("unimp"); for(;;){} } while(0)
 *
 *     // Custom UART logger + reset
 *     #define DEEPBINDI_FATAL(msg)  do { uart_puts(msg); system_reset(); } while(0)
 *
 * ── Embedded portability notes ────────────────────────────────────────────────
 *
 *   • int size: on 16-bit MCUs (AVR, MSP430) int is 16-bit.  Tensor shape
 *     fields (n*c*h*w) can exceed 32 767 for the larger 2-D models.  Use a
 *     32-bit toolchain or change Tensor shape fields to int32_t / uint32_t.
 *
 *   • float vs double: all arithmetic uses float throughout.  No implicit
 *     promotion to double occurs in the compute kernels.  printf("%f") does
 *     promote to double on some hosts; this only matters when logging is on.
 *
 *   • memset to zero for float: relies on IEEE-754 (all-zero bits == 0.0f).
 *     This holds on every Cortex-M, RISC-V, and x86 target in common use.
 *
 *   • %zu format specifier: not supported by newlib-nano (--specs=nano.specs).
 *     This header avoids %zu; arena_stats() uses explicit (unsigned) casts.
 */

#ifndef DEEPBINDI_CONFIG_H
#define DEEPBINDI_CONFIG_H

/* ── Logging macros ─────────────────────────────────────────────────────── */

#ifdef DEEPBINDI_ENABLE_LOGGING
#  include <stdio.h>
   /** Print an informational message (stdout). */
#  define DEEPBINDI_PRINTF(...)       printf(__VA_ARGS__)
   /** Print an error message (stderr on host; redirected on embedded). */
#  define DEEPBINDI_LOG_ERROR(...)    fprintf(stderr, __VA_ARGS__)
#else
#  define DEEPBINDI_PRINTF(...)       ((void)0)
#  define DEEPBINDI_LOG_ERROR(...)    ((void)0)
#endif

/* ── Fatal error handler ────────────────────────────────────────────────── */

#ifndef DEEPBINDI_FATAL
#  ifdef DEEPBINDI_ENABLE_LOGGING
#    include <stdlib.h>
     /** Log msg and terminate (host behaviour). */
#    define DEEPBINDI_FATAL(msg) \
         do { DEEPBINDI_LOG_ERROR("FATAL: %s\n", (msg)); exit(1); } while(0)
#  else
     /** Spin forever – triggers watchdog reset on most embedded targets. */
#    define DEEPBINDI_FATAL(msg) \
         do { for(;;){} } while(0)
#  endif
#endif

#endif /* DEEPBINDI_CONFIG_H */
