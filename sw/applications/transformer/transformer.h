#ifndef _STIMULI_H_
#define _STIMULI_H_

#include <stdint.h>
#include "cgra.h"

#define ITERATIONS_PER_KERNEL 1

#define ROWS_A 4
#define COLS_A 4
#define ROWS_B 4
#define COLS_B 4
#define BLOCK_SIZE 4
#define OUTPUT_COLS COLS_B
#define OUTPUT_ROWS ROWS_A

#define WATCHDOG_CYCLES         100000
#define HART_ID                         0

typedef uint32_t    kcom_time_t;
typedef kcom_time_t kcom_param_t;

typedef struct
{
    uint32_t cyc_act;
    uint32_t cyc_stl;
} kcom_col_perf_t;

typedef struct
{
    kcom_time_t start_cy;
    kcom_time_t end_cy;
    kcom_time_t spent_cy;
} kcom_time_diff_t;

typedef struct
{
    kcom_time_diff_t    sw;
    kcom_time_diff_t    cgra;
    kcom_time_diff_t    load;
    kcom_time_diff_t    conf;
    kcom_time_diff_t    dead;
} kcom_timing_t;

typedef struct
{
    kcom_col_perf_t    cols[CGRA_N_COLS];
    kcom_col_perf_t    cols_max;
    uint32_t           cyc_ratio; // Stored *CGRA_STAT_PERCENT_MULTIPLIER
    kcom_timing_t      time;
} kcom_perf_t;

typedef struct
{
    kcom_param_t sw;
    kcom_param_t conf;
    kcom_param_t cgra;
    kcom_param_t repo;
} kcom_run_t;

typedef struct
{
   kcom_run_t   avg;
   uint32_t     n;
   uint32_t     errors;
   uint8_t      *name;
} kcom_stats_t;

#endif // _STIMULI_H_
