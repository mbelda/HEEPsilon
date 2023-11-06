/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : CGRA-X-HEEP                                                  **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 05/04/23                                                     **
**                                                                         **
*****************************************************************************
**                                                                         **
** Copyright (c) EPFL                                                      **
** All rights reserved.                                                    **
**                                                                         **
*****************************************************************************
*/

/***************************************************************************/
/***************************************************************************/

/**
* @file   main.c
* @date   05/04/23
* @brief  An application to run a number of kernels under a same given
* structure.
*
*/

#define _KERNELS_C

/****************************************************************************/
/**                                                                        **/
/*                             MODULES USED                                 */
/**                                                                        **/
/****************************************************************************/

#include <stdlib.h>

#include "transformer.h"
#include "cgra_bitstream.h"
#include "cgra_x_heep.h"


// For GPIO managing
#include "gpio.h"
#include "pad_control.h"
#include "pad_control_regs.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

// For the timer
#include "rv_timer.h"
#include "soc_ctrl.h"
#include "core_v_mini_mcu.h"
/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                        TYPEDEFS AND STRUCTURES                           */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

void mmulSoftware();
void fillMatrixInputs();
void handler_irq_ext(uint32_t id);

/****************************************************************************/
/**                                                                        **/
/*                           EXPORTED VARIABLES                             */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

static kcom_perf_t  kperf;
static kcom_stats_t stats;
static rv_timer_t          timer;

// Plic controller variables
volatile bool               cgra_intr_flag;

static cgra_t               cgra;
static uint8_t              cgra_slot;

static int32_t cgra_input[CGRA_N_COLS][ROWS_A+ROWS_B]    __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][ROWS_A*COLS_B]   __attribute__ ((aligned (4)));

static int16_t outSW[ROWS_A*COLS_B];
static int16_t outCGRA[ROWS_A*COLS_B];

static int16_t matrixA[ROWS_A*COLS_A];
static int16_t matrixB[ROWS_B*COLS_B];

/****************************************************************************/
/**                                                                        **/
/*                           EXPORTED FUNCTIONS                             */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  fillMatrixInputs();

  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);

  // Init timer
  timerInit();

  // Enable interrupt on processor side
  // Enable global interrupt for machine-level interrupts
  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  // Set mie.MEIE bit to one to enable machine-level external interrupts
  const uint32_t mask = 1 << 11;//IRQ_EXT_ENABLE_OFFSET;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  // Load kernel
  //timeStart( &(kperf.time.load) );
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  // Select request slot of CGRA
  cgra_slot = cgra_get_slot(&cgra);
  cgra_perf_cnt_enable(&cgra, 1);
  // Set CGRA kernel L/S pointers
  for(int8_t col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
  }
  //timeStop( &(kperf.time.load) );

 
  // Reset the CGRA performance counters
  //cgra_perf_cnt_reset( &cgra );

  // CGRA input values
  //printf("Input values\n");
  int cont = 0;
  for (int i=0; i < ROWS_A; i++,cont++){
    for (int j=0; j < COLS_A; j++){
      cgra_input[j][cont] = matrixA[i*COLS_A+j];
    }
  }
  for (int i=0; i < ROWS_B; i++,cont++){
    for (int j=0; j < COLS_B; j++){
      cgra_input[j][cont] = matrixB[i*COLS_B+j];
    }
  }

  // Software 
  //timeStart(   &(kperf.time.sw) );
  mmulSoftware();
  //timeStop(    &(kperf.time.sw) );

  // CGRA Execution
  //kcom_perfRecordIntrSet( &(kperf.time.cgra) );
  //printf("CGRA multiplication\n");
  cgra_intr_flag = 0;
  //timeStart( &(kperf.time.cgra) );
  cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
  // Wait CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  cgra_intr_flag = 0;
  //timeStop( &(kperf.time.sw) );
  //printf("Done\n");

  // Move CGRA output
  cont = 0;
  for(int16_t i = 0; i < CGRA_N_ROWS; i++) {
    for(int16_t j=0; j < CGRA_N_COLS; j++, cont++){
      //printf("%d ", cgra_output[i][j]);
      outCGRA[cont] = cgra_output[j][i];
    }
  }

  int errors=0;
  for( uint16_t i = 0; i < ROWS_A*COLS_B; i++ ){
    if(outSW[i]!=outCGRA[i]){
      errors++;
      printf("[%d] %d != %d\n", i, outSW[i],outCGRA[i] );
    }
  }
  printf("Errors: %d\n", errors);
  // Performance counter display
  /*
  printf("CGRA kernel executed: %d\n", cgra_perf_cnt_get_kernel(&cgra));
  for(int column_idx = 0; column_idx < CGRA_N_COLS; column_idx++){
    printf("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
    printf("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  }
  */
  
  return EXIT_SUCCESS;
}

// Fill matrix inputs
void fillMatrixInputs(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0; j < COLS_A; j++){
      matrixA[i*COLS_A+j] = i*COLS_A+j+1;
    }
  }

  for(int i = 0; i < ROWS_B; i++){
    for(int j=0;j < COLS_B; j++){
      matrixB[i*COLS_B+j] = i*COLS_B+j+1;
    }
  }
}

// Software matrix multiplication
void mmulSoftware(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        outSW[i*COLS_B+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
    }
  }
}

void timerInit()
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


// Interrupt controller variables
void handler_irq_ext(uint32_t id) {
  if( id == CGRA_INTR) {
    cgra_intr_flag = 1;
  }
}

/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
