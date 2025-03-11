/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : GeMM                                                         **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 04/03/2025                                                   **
**                                                                         **
*****************************************************************************
**                                                                         **
** Copyright (c) UCM                                                       **
** All rights reserved.                                                    **
**                                                                         **
*****************************************************************************
*/

/***************************************************************************/
/***************************************************************************/

/**
* @file   main.c
* @date   04/03/2025
* @brief  An application to run a matrix multiplication.
*
*/

/****************************************************************************/
/**                                                                        **/
/*                             MODULES USED                                 */
/**                                                                        **/
/****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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

#define CGRA_COL_INPUT_SIZE 6 // Size of the input buffer for the CGRA
#define BLOCK_SIZE 4

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Matrix multiplication using the standard three loops
void mmulSoftware(int32_t * output);
// Fill input matrixes with numbers
void fillMatrixInputs(int * matrix, int rows, int cols);
// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Record the cycle number at the start
void kcom_perfRecordStart( kcom_time_diff_t *perf );
// Record the cycle number and compute the total cycles
void kcom_perfRecordStop( kcom_time_diff_t *perf );
// Show the performance metrics
void showPerformance( kcom_perf_t* kperf, int full);
// Print a matrix
void printMatrix(int * matrix, int rows, int cols);
// Process non multiple number of rows and cols
void processExtraRowsAColsB();

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

// Performance variable
static kcom_perf_t  kperf;

// Plic controller variables
volatile bool               cgra_intr_flag;

// CGRA variables
static cgra_t               cgra;
static uint8_t              cgra_slot;

// CGRA input and output buffers
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
int32_t matrixA[(ROWS_A+1)*COLS_A];
int32_t matrixB[COLS_A*COLS_B];
int32_t matrixC[(ROWS_A+1)*COLS_B];
int32_t outSW[ROWS_A*COLS_B];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  printf("Launching mmul_os opt v2 for dimension %dx%dx%d\n", ROWS_A, COLS_A, COLS_B);
  fillMatrixInputs(matrixA, ROWS_A, COLS_A);
  fillMatrixInputs(matrixB, COLS_A, COLS_B);

  // Init timer
  timerInit();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  if(ROWS_A < CGRA_N_COLS || COLS_B < CGRA_N_ROWS){
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(matrixC);
    kcom_perfRecordStop(&(kperf.time.sw));
  } else {
    // Initialize the CGRA
    kcom_perfRecordStart(&(kperf.time.load));
    initCGRA();
    kcom_perfRecordStop(&(kperf.time.load));
    int nRowsA = ROWS_A;
    if (ROWS_A%4 == 3){
      // Special case
      nRowsA = ROWS_A +1;
    }

    // Prepare the input vector for the CGRA
    //kcom_perfRecordStart(&(kperf.time.input));
    kcom_perfRecordStart(   &(kperf.time.cgra) );
    // ----------------------
    // &B[0][0]          &C[0][1]        &A[0][0]    nRowsBlocksC
    // nColsBlocksC      &B[0][1]        &C[1][2]    &A[1][0]
    // &A[2][0]          loopColsA       &B[0][2]    &C[2][3]
    // &C[3][0]          &A[3][0]        -           &B[0][3]
    // ----------------------
    // -4*colsB          colsA           -           -
    // -                 -4*colsB        colsA       -
    // -                 -               -4*colsB    colsA
    // colsA             -               -           -4*colsB
    int nItLoopColsA = COLS_A;
    int nColsBlocksC = COLS_B/CGRA_N_ROWS;
    int nRowsBlocksC = nRowsA/CGRA_N_ROWS;
    // Col 0
    cgra_input[0][0] = &matrixB[0];
    cgra_input[0][1] = nColsBlocksC;
    cgra_input[0][2] = &matrixA[2*COLS_A];
    cgra_input[0][3] = &matrixC[3*COLS_B];
    cgra_input[0][4] = -4*COLS_B;
    cgra_input[0][5] = COLS_A;
    // Col 1
    cgra_input[1][0] = &matrixC[1];
    cgra_input[1][1] = &matrixB[1];
    cgra_input[1][2] = nItLoopColsA;
    cgra_input[1][3] = &matrixA[3*COLS_A];
    cgra_input[1][4] = COLS_A;
    cgra_input[1][5] = -4*COLS_B;
    // Col 2
    cgra_input[2][0] = &matrixA[0];
    cgra_input[2][1] = &matrixC[COLS_B+2];
    cgra_input[2][2] = &matrixB[2];
    cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
    cgra_input[2][4] = COLS_A;
    cgra_input[2][5] = -4*COLS_B;
    // Col 3
    cgra_input[3][0] = nRowsBlocksC;
    cgra_input[3][1] = &matrixA[COLS_A];
    cgra_input[3][2] = &matrixC[2*COLS_B+3];
    cgra_input[3][3] = &matrixB[3];
    cgra_input[3][4] = COLS_A;
    cgra_input[3][5] = -4*COLS_B;
    //kcom_perfRecordStop(&(kperf.time.input));

    // Set CGRA kernel L/S pointers
    //kcom_perfRecordStart( &(kperf.time.reprogramCols) );
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
      cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }
    //kcom_perfRecordStop( &(kperf.time.reprogramCols) );

    // CGRA Execution
    
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
    // Process extra rows/cols for non multiple dimensions
    processExtraRowsAColsB();
    // Wait until CGRA is done
    while(cgra_intr_flag==0) {
      wait_for_interrupt();
    }
    kcom_perfRecordStop(   &(kperf.time.cgra) );

    // Software 
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(outSW);
    kcom_perfRecordStop(&(kperf.time.sw));
    
    checkErrors();
  }

  showPerformance(&kperf, 1);
  
  return EXIT_SUCCESS;
}

void processExtraRowsAColsB(){
  int rAMax = ROWS_A;
  // Extra A rows
  if (ROWS_A%4 != 3){
    for(int rA = ROWS_A - ROWS_A%BLOCK_SIZE; rA < ROWS_A; rA++){
      for(int cB = 0; cB < COLS_B; cB++){
        int sum = 0;
        for(int k = 0; k < COLS_A; k++){
        sum += matrixA[rA*COLS_A + k] * matrixB[k*COLS_B + cB];
        }
        matrixC[rA*COLS_B + cB] = sum;
      } 
    }
    rAMax = ROWS_A - ROWS_A%BLOCK_SIZE;
  }

  // Extra cols B
  for(int cB = COLS_B - COLS_B%BLOCK_SIZE; cB < COLS_B; cB++){
    for(int rA = 0; rA < rAMax; rA++){
      int sum = 0;
      for(int k = 0; k < COLS_A; k++){
       sum += matrixA[rA*COLS_A + k] * matrixB[k*COLS_B + cB];
      }
      matrixC[rA*COLS_B + cB] = sum;
    } 
  }
}

int errors_idx[ROWS_A*COLS_B];

void printVectorIdx(int * vec, int * idx, int n){
  printf("[");
  for (int i = 0; i < n; i++){
    printf("%d, ", vec[idx[i]]);
  }
  printf("]\n");
}

void printVector(int * vec, int n){
  printf("[");
  for (int i = 0; i < n; i++){
    printf("%d, ", vec[i]);
  }
  printf("]\n");
}

// Check if the SW and CGRA executions give the same result
void checkErrors(){
  int errors = 0;
  //printf("CGRA res: [");
  for(int i = 0; i < ROWS_A*COLS_B; i++ ){
    if (i < 5){
      //printf("%d, ", matrixC[i]);
    }
    if(outSW[i]!=matrixC[i]){
      errors_idx[errors] = i;
      errors++;
    }
  }
  //printf("...]\n");
  if (errors > 0){
    printf("Errors: %d\n", errors);
  } else{
    printf("OK\n");
  }
  

  if(errors>0){
    printVector(errors_idx, errors);
    //printVectorIdx(matrixC, errors_idx, errors);
  }
}

// Initialize the CGRA
void initCGRA(){
  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, (void *) &handler_irq_cgra);

  // Enable interrupt on processor side
  // Enable global interrupt for machine-level interrupts
  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  // Set mie.MEIE bit to one to enable machine-level external interrupts
  const uint32_t mask = 1 << 11;//IRQ_EXT_ENABLE_OFFSET;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  // Load kernel
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  // Select request slot of CGRA
  cgra_slot = cgra_get_slot(&cgra);
}

// Fill matrix inputs
void fillMatrixInputs(int * matrix, int rows, int cols){
  for(int i = 0; i < rows; i++){
    for(int j=0; j < cols; j++){
      matrix[i*cols+j] = (i*cols+j+1)%100;
    }
  }
}

// Software matrix multiplication
void mmulSoftware(int32_t * out){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      int sum = 0;
      for(int k=0; k < COLS_A; k++){
        sum += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
      out[i*COLS_B+j] = sum;
    }
  }
}

// Print matrix
void printMatrix(int * matrix, int rows, int cols){
  for(int i = 0; i < rows; i++){
    printf("[ ");
    for(int j=0; j < cols; j++){
      printf("%d ", matrix[i*cols+j]);
    }
    printf("]\n");
  }
}

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}

// Display the performance values
void showPerformance( kcom_perf_t* kperf, int full){
  
  printf("Sw: %d\n", kperf->time.sw.spent_cy);
  if (full){
    printf("Load: %d\n", kperf->time.load.spent_cy);
    //printf("Program cols: %d\n", kperf->time.reprogramCols.spent_cy);
    //printf("Input: %d\n", kperf->time.input.spent_cy);
    printf("Cgra: %d\n", kperf->time.cgra.spent_cy);
    //printf("ExtraRowsCols: %d\n", kperf->time.extraRowsCols.spent_cy);
  }
  int32_t overhead = kperf->time.input.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy;
  //printf("Total cgra: %d\n", overhead + kperf->time.cgra.spent_cy); 
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
