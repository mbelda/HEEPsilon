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

// (ROWS_A/BLOCK_SIZE)*CGRA_N_COLS/BLOCK_SIZE = numero de bloques de A que cabe su primera fila en la primera fila del CGRA
//#define CGRA_COL_INPUT_SIZE BLOCK_SIZE+(ROWS_A/BLOCK_SIZE)*CGRA_N_COLS/BLOCK_SIZE
//#define CGRA_COL_OUTPUT_SIZE CGRA_N_ROWS*((ROWS_A/BLOCK_SIZE)*CGRA_N_COLS/BLOCK_SIZE) 

//static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));
//static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

#define INPUT_SIZE_FOR_B CGRA_N_ROWS*BLOCK_SIZE
#define INPUT_SIZE_FOR_A ROWS_A*BLOCK_SIZE
#define INPUT_SIZE_FOR_ITER ROWS_A +1
#define INPUT_SIZE_NUMBER_ITERATIONS_OF_B COLS_B/(CGRA_N_COLS*CGRA_N_ROWS)*ROWS_B/BLOCK_SIZE

#define CGRA_COL_INPUT_SIZE INPUT_SIZE_FOR_B+INPUT_SIZE_FOR_A+INPUT_SIZE_FOR_ITER
#define CGRA_COL_OUTPUT_SIZE CGRA_N_ROWS*ROWS_A 
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

static int16_t outSW[ROWS_C*COLS_C];

static int16_t matrixA[ROWS_A*COLS_A];
static int16_t matrixB[ROWS_B*COLS_B];
static int16_t matrixC[ROWS_C*COLS_C] = {0};

volatile int cgraIteration;
static int numIterations = ROWS_A/CGRA_N_COLS*CGRA_N_COLS;

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
  // TODO: Caso en el que la matriz sea tan pequeña que no haya ni un bloque
  fillMatrixInputs();

  // Reset the CGRA performance counters
  //cgra_perf_cnt_reset( &cgra );

  initCGRA();

  //printf("\rA: %dx%d, B: %dx%d, cgra_input: %dx%d, cgra_output: %dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B, CGRA_N_COLS, CGRA_COL_INPUT_SIZE, CGRA_N_COLS, CGRA_COL_OUTPUT_SIZE);
  
  int CGRA_block_B = CGRA_N_COLS*CGRA_N_ROWS;
  int nBlocksCGRA = CGRA_N_ROWS*CGRA_N_COLS/BLOCK_SIZE;
  int tamBlockCGRA = COLS_B/nBlocksCGRA;
  for (int rB = 0; rB < ROWS_B; rB += BLOCK_SIZE){ // TODO: Gestionar no multiplos (ROWS_B no multiplo de BLOCK_SIZE)
    for (int cB = 0; cB + CGRA_block_B <= COLS_B; cB += CGRA_block_B){ //TODO: gestionar no multiplos (COLS_B no multiplo de CGRA_block_B)
      int cont = 0;
      for (int subRowB = 0; subRowB < BLOCK_SIZE; subRowB++){
        int cBAux=cB;
        while(cBAux < cB + CGRA_block_B){
          for(int j = 0; j < CGRA_N_COLS; j++, cBAux++){
            //printf("\rcgraIn[%d][%d] = B[%d][%d]\n", j, cont, rB+subRowB, cBAux);
            cgra_input[j][cont] = matrixB[(rB+subRowB)*COLS_B+cBAux];
          }
          cont++;
        }
      }

      cgraIteration = 0;

      // Loop control
      cgra_input[0][cont] = &cgraIteration;
      cgra_input[1][cont] = 0;
      cgra_input[2][cont] = numIterations;
      for(int a = 3; a < CGRA_N_COLS; a++){ // Zeros for the rest of the columns
        cgra_input[a][cont] = 0;
      }
      cont++;

      int cA = rB;
      for(int rA = 0; rA+CGRA_N_COLS <= ROWS_A; rA+=CGRA_N_COLS){
        for(int offset = 0; offset < CGRA_N_COLS; offset++){ // Load A values again but shifted one row
          for(int subColA = 0; subColA < BLOCK_SIZE; subColA++){
            int offsetAux = offset;
            for(int j=0; j < CGRA_N_COLS; j++,offsetAux++){
              //printf("\rcgraIn[%d][%d] = A[%d][%d]\n", j, cont, rA + offsetAux%CGRA_N_COLS, cA+subColA);
              cgra_input[j][cont] = matrixA[(rA + offsetAux%CGRA_N_COLS)*COLS_A+(cA+subColA)];
            }
            cont++;
          }

          // Loop control
          cgra_input[0][cont] = &cgraIteration;
          cgra_input[1][cont] = 0;
          cgra_input[2][cont] = numIterations;
          for(int a = 3; a < CGRA_N_COLS; a++){ // Zeros for the rest of the columns
            cgra_input[a][cont] = 0;
          }
          cont++;
        }
      }

      // CGRA Execution
      cgra_intr_flag = 0;
      cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
      // Wait until CGRA is done
      while(cgra_intr_flag==0) {
        wait_for_interrupt();
      }
      cgra_intr_flag = 0;

      // Move CGRA output
      //for(int blockA = 0; blockA <= ROWS_A/CGRA_N_ROWS; blockA++){
      //  int contAux = 0;
      //  for(int offset = 0; offset < CGRA_N_COLS; offset++){
        int contAux = 0;
        for(int it = 0; it < ROWS_A; it++){
          int blockA = it/CGRA_N_ROWS;  
          for(int i=0; i < CGRA_N_ROWS; i++, contAux++){
            for(int j=0; j < CGRA_N_COLS; j++){
              int colC = cB + i*CGRA_N_ROWS + j;
              int rowC = (it + j)%CGRA_N_COLS + blockA*CGRA_N_COLS; 
              //printf("\rmC[%d][%d] += cOut[%d][%d] = %d\n", rowC, colC, j, blockA*CGRA_N_ROWS + i, cgra_output[j][blockA*CGRA_N_ROWS + i]);
              matrixC[rowC*COLS_C + colC] += cgra_output[j][contAux];
            }
          }
        }   
      //}

      // Set CGRA kernel L/S pointers for the next iteration
      for(int8_t col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
        cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
        cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
      }
    }
  }  

  checkErrors();
  
  //return EXIT_SUCCESS;
  return 0;
}

void checkErrors(){
  // Software 
  mmulSoftware();

  int errors = 0;
  for( uint16_t i = 0; i < ROWS_C*COLS_C; i++ ){
    if(outSW[i]!=matrixC[i]){
      errors++; 
      //printf("\rC[%d][%d] %d != %d\n", i/COLS_C, i%COLS_C, outSW[i],matrixC[i] );
    }
  }

  printf("\rErrors: %d\n", errors);
  /*
  printf("\rSW\n");
  printf("\r");
  for (int i = 0; i < ROWS_C; i++){
    for (int j = 0; j < COLS_C; j++){
      printf("%.4d  ", outSW[i*COLS_C+j]);
    }
    printf("\n\r");
  }*/
  
  printf("\rMatrixC\n");
  printf("\r");
  for (int i = 0; i < ROWS_C; i++){
    for (int j = 0; j < COLS_C; j++){
      printf("%.4d  ", matrixC[i*COLS_C+j]);
    }
    printf("\n\r");
  }

  /*
  printf("\rCGRA\n");
  printf("\r");
  for (int i = 0; i < CGRA_COL_OUTPUT_SIZE; i++){
    for (int j = 0; j < CGRA_N_COLS; j++){
      printf("%.4d  ", cgra_output[j][i]);
    }
    printf("\n\r");
  }*/
}

void initCGRA(){
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

//Initialize the timer
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
