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
#include "performance.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"


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
void processExtraRowsCols(int rB, int cB);
void kcom_perfRecordStart( kcom_time_diff_t *perf );
void kcom_perfRecordStop( kcom_time_diff_t *perf );

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

int16_t matrixA[ROWS_A*COLS_A];
int16_t matrixB[ROWS_B*COLS_B];
int16_t matrixC[ROWS_C*COLS_C] = {0};

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

  // Init timer
  timerInit();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  kcom_perfRecordStart( &(kperf.time.load) );
  initCGRA();
  kcom_perfRecordStop( &(kperf.time.load) );

  kcom_perfRecordStart( &(kperf.time.input) );
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
      kcom_perfRecordStop( &(kperf.time.input) );

      // CGRA Execution
      //kcom_perfRecordIntrSet( &(kperf.time.cgra) );
      kcom_perfRecordStart(   &(kperf.time.cgra) );
      cgra_intr_flag = 0;
      cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
      // Wait until CGRA is done, while waiting process extra columns and rows
      int processed = 0;
      while(cgra_intr_flag==0) {
        /*if (!processed){
          processExtraRowsCols(rB, cB);
          processed = 1;
        }*/
        wait_for_interrupt();
      }
      kcom_perfRecordStop(   &(kperf.time.cgra) );

      /*if (!processed){
        processExtraRowsCols(rB, cB);
      }*/
      cgra_intr_flag = 0;

      kcom_perfRecordStart( &(kperf.time.output) );
      // Move CGRA output
      int contAux = 0;
      for(int it = 0; it < ROWS_A; it++){
        int blockA = it/CGRA_N_COLS;  
        for(int i=0; i < CGRA_N_ROWS; i++, contAux++){
          for(int j=0; j < CGRA_N_COLS; j++){
            int colC = cB + i*CGRA_N_ROWS + j;
            int rowC = (it + j)%CGRA_N_COLS + blockA*CGRA_N_COLS; 
            matrixC[rowC*COLS_C + colC] += cgra_output[j][contAux];
          }
        }
      }
      kcom_perfRecordStop( &(kperf.time.output) );

      kcom_perfRecordStart( &(kperf.time.reprogramCols) );
      // Set CGRA kernel L/S pointers for the next iteration
      for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
        cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
        cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
      }
      kcom_perfRecordStop( &(kperf.time.reprogramCols) );
    }
  }  

  checkErrors();

  showPerformance( &kperf);
  
  //return EXIT_SUCCESS;
  return 0;
}

void processExtraRowsCols(int rB, int cB){
  // Process extra rows for A
  for(int rA = ROWS_A/CGRA_N_COLS*CGRA_N_COLS; rA < ROWS_A; rA++){
    for(int c = cB; c < cB + CGRA_N_COLS*CGRA_N_ROWS; c++){
      for(int b = 0; b < BLOCK_SIZE; b++){
        matrixC[rA*COLS_C + c] += matrixA[rA*COLS_A + rB + b] * matrixB[(b+rB)*COLS_B + c];
      }
    } 
  }
}

void checkErrors(){
  // Software 
  kcom_perfRecordStart(   &(kperf.time.sw) );
  mmulSoftware();
  kcom_perfRecordStop(   &(kperf.time.sw) );

  int errors = 0;
  for(int i = 0; i < ROWS_C*COLS_C; i++ ){
    if(outSW[i]!=matrixC[i]){
      errors++; 
      printf("\rC[%d][%d] %d != %d\n", i/COLS_C, i%COLS_C, outSW[i],matrixC[i] );
    }
  }

  printf("\rErrors: %d\n", errors);
}

void initCGRA(){
  // Init the PLIC
  kcom_perfRecordStart(   &(kperf.time.plic) );
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  kcom_perfRecordStop(   &(kperf.time.plic) );

  // Enable interrupt on processor side
  kcom_perfRecordStart(   &(kperf.time.interrupt) );
  // Enable global interrupt for machine-level interrupts
  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  // Set mie.MEIE bit to one to enable machine-level external interrupts
  const uint32_t mask = 1 << 11;//IRQ_EXT_ENABLE_OFFSET;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;
  kcom_perfRecordStop(   &(kperf.time.interrupt) );

  // Load kernel
  kcom_perfRecordStart(   &(kperf.time.bitstream) );
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  kcom_perfRecordStop(   &(kperf.time.bitstream) );

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  // Select request slot of CGRA
  cgra_slot = cgra_get_slot(&cgra);

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
  }
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
void mmulSoftware(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        outSW[i*COLS_B+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
    }
  }
}


// Interrupt controller variables
void handler_irq_ext(uint32_t id) {
  if( id == CGRA_INTR) {
    cgra_intr_flag = 1;
  }
}

showPerformance( kcom_perf_t* kperf){
  printf("\rA:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B);
  printf("\r-----------------------------\n");
  printf("\rSw: %d\n", kperf->time.sw.spent_cy);
  printf("\rLoad: %d\n", kperf->time.load.spent_cy);
  printf("\r    Plic: %dn\r    Interrupt: %d\n\r    Bitstream: %d\n", kperf->time.plic.spent_cy, kperf->time.interrupt.spent_cy, kperf->time.bitstream.spent_cy);
  printf("\rReprogram cols: %d\n", kperf->time.reprogramCols.spent_cy);
  printf("\rInput: %d\n", kperf->time.input.spent_cy);
  printf("\rOutput: %d\n", kperf->time.output.spent_cy);
  printf("\rCgra: %d\n", kperf->time.cgra.spent_cy);
  printf("\r-----------------------------\n");
  int32_t overhead = kperf->time.input.spent_cy + kperf->time.output.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy;
  printf("\rOverhead: %d\n", overhead);
  printf("\rTotal cgra: %d\n", overhead + kperf->time.cgra.spent_cy);
  //float speedup = ((float) kperf->time.sw.spent_cy) / ((float) (overhead + kperf->time.cgra.spent_cy));
  //printf("\rSpeedup: %f\n", speedup);  
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
