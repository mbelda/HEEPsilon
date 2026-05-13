

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


// #include "performance.h"
// #include "transformer.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

#include <stdint.h>
#include "cgra.h"

// #include "performance.h"

// For the timer
#include "rv_timer.h"
#include "soc_ctrl.h"
#include "core_v_mini_mcu.h"

#define HART_ID 0


#define ROWS_A 121
#define COLS_A 16
#define COLS_B 4
#define ROWS_B COLS_A
#define ROWS_C ROWS_A
#define COLS_C COLS_B

// Timer
static rv_timer_t          timer;

typedef long int    kcom_time_t;
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
    kcom_time_diff_t    input;
    kcom_time_diff_t    output;
    kcom_time_diff_t    reprogramCols;
    kcom_time_diff_t    bitstream;
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


static void timerInit();
static uint64_t getTime_cy( );
static void timeStop( kcom_time_diff_t *perf );
static void timeStart( kcom_time_diff_t *perf );
static void kcom_perfRecordStop( kcom_time_diff_t *perf );
static void kcom_perfRecordStart( kcom_time_diff_t *perf );


/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Matrix multiplication using the standard three loops
void mmulSoftware(int32_t * output);
// Fill input matrixes with numbers
void fillMatrixInputs();

void showPerformance( kcom_perf_t* kperf, int full);

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

// Performance variable
static kcom_perf_t  kperf;






// Input and output matrixes
static int32_t __attribute__((section(".xheep_data_interleaved"))) matrixA[ROWS_A*COLS_A];
static int32_t __attribute__((section(".xheep_data_interleaved"))) matrixB[ROWS_B*COLS_B];
static int32_t __attribute__((section(".xheep_data_interleaved"))) matrixC[ROWS_C*COLS_C];
int32_t outSW[ROWS_C*COLS_C];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  fillMatrixInputs();

  // Init timer
  timerInit();


  
    // Software 
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(outSW);
    kcom_perfRecordStop(&(kperf.time.sw));


  showPerformance(&kperf, 1);
  
  return EXIT_SUCCESS;
}





// Fill matrix inputs
void fillMatrixInputs(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0; j < COLS_A; j++){
      matrixA[i*COLS_A+j] = (i*COLS_A+j+1)%100;
    }
  }

  for(int i = 0; i < ROWS_B; i++){
    for(int j=0;j < COLS_B; j++){
      matrixB[i*COLS_B+j] = (i*COLS_B+j+1)%100;
    }
  }
}

// Software matrix multiplication
void mmulSoftware(int32_t * out){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        out[i*COLS_C+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
    }
  }
}

// Display the performance values
void showPerformance( kcom_perf_t* kperf, int full){
  printf("\rA:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B);
  printf("\rSw: %d\n", kperf->time.sw.spent_cy);
}


static void kcom_perfRecordStart( kcom_time_diff_t *perf )
{
    timeStart( perf );
}

static void kcom_perfRecordStop( kcom_time_diff_t *perf )
{
    timeStop( perf );
}

static void timeStart( kcom_time_diff_t *perf )
{
    perf->start_cy = getTime_cy();
}

static void timeStop( kcom_time_diff_t *perf )
{
    perf->end_cy = getTime_cy();
    perf->spent_cy += perf->end_cy - perf->start_cy;
}

static uint64_t getTime_cy( )
{
    static uint64_t out;
    rv_timer_counter_read( &timer, HART_ID, &out );
    return out;
}

//Initialize the timer
static void timerInit()
{
    soc_ctrl_t soc_ctrl;
    soc_ctrl.base_addr  = mmio_region_from_addr((uintptr_t)SOC_CTRL_START_ADDRESS);
    uint32_t freq_hz  = soc_ctrl_get_frequency(&soc_ctrl);

    mmio_region_t timer_0_reg = mmio_region_from_addr(RV_TIMER_AO_START_ADDRESS);

    rv_timer_init( timer_0_reg, (rv_timer_config_t) { .hart_count = 2, .comparator_count = 1 }, &timer );

    rv_timer_tick_params_t tick_params;

    // The same frequency is provaided to get one tick per cycle.
    rv_timer_approximate_tick_params( freq_hz, freq_hz, &tick_params );
    rv_timer_set_tick_params(&timer, HART_ID, tick_params);

    // Juan: see if i cannot remove this!
    rv_timer_irq_enable(&timer, HART_ID, 0, kRvTimerEnabled);
    rv_timer_arm(&timer, HART_ID, 0, 1);

    rv_timer_counter_set_enabled(&timer, HART_ID, kRvTimerEnabled);

}
