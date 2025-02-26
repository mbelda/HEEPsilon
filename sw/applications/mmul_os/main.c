/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : HEEPsilon                                                    **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 01/10/23                                                     **
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
* @date   01/10/23
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

// Size of the input buffer for the CGRA
#define CGRA_COL_INPUT_SIZE 4
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
void processExtraRowsA();
// Manage padding
void add_padding(int *matrix, int rows, int cols, int padded_cols);
void remove_padding(int *matrix, int rows, int cols, int padded_cols);

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
int padded_COLS_B = COLS_B;

// CGRA input and output buffers
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
int32_t matrixA[ROWS_A*COLS_A];
int32_t matrixB[COLS_A*(COLS_B+BLOCK_SIZE-COLS_B%BLOCK_SIZE)]; // to be able to fit the padding if needed
int32_t matrixC[ROWS_A*(COLS_B+BLOCK_SIZE-COLS_B%BLOCK_SIZE)];
int32_t outSW[ROWS_A*COLS_B];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
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
    // Software 
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(outSW);
    kcom_perfRecordStop(&(kperf.time.sw));

    // Initialize the CGRA
    kcom_perfRecordStart(&(kperf.time.load));
    initCGRA();
    kcom_perfRecordStop(&(kperf.time.load));
    
    // Add padding to the matrix if needed
    kcom_perfRecordStart(&(kperf.time.padding));
    if (COLS_B % BLOCK_SIZE != 0){
      padded_COLS_B = COLS_B + 4 - (COLS_B%4);
      add_padding(matrixB, COLS_A, COLS_B, padded_COLS_B);
    }
    kcom_perfRecordStop(&(kperf.time.padding));

    // Prepare the input vector for the CGRA
    kcom_perfRecordStart(&(kperf.time.input));
    // Col 0: &B[0][0], nItLoopColsC, &A[0][0], &C[0][3]
    cgra_input[0][0] = &matrixB[0];
    cgra_input[0][1] = padded_COLS_B/CGRA_N_ROWS;
    cgra_input[0][2] = &matrixA[0];
    cgra_input[0][3] = &matrixC[3];
    // Col 1: &C[1][0], &B[0][1], nItLoopsColsA, &A[1][0]
    cgra_input[1][0] = &matrixC[padded_COLS_B];
    cgra_input[1][1] = &matrixB[1];
    cgra_input[1][2] = COLS_A;
    cgra_input[1][3] = &matrixA[COLS_A];
    // Col 2: &A[2][0], &C[2][1], &B[0][2], nItLoopColsC
    cgra_input[2][0] = &matrixA[2*COLS_A];
    cgra_input[2][1] = &matrixC[2*padded_COLS_B+1];
    cgra_input[2][2] = &matrixB[2];
    cgra_input[2][3] = padded_COLS_B/CGRA_N_ROWS;
    // Col 3: nItLoopRowsC, &A[3][0], &C[3][2], &B[0][3], 
    cgra_input[3][0] = ROWS_A/CGRA_N_COLS;
    cgra_input[3][1] = &matrixA[3*COLS_A];
    cgra_input[3][2] = &matrixC[3*padded_COLS_B+2];
    cgra_input[3][3] = &matrixB[3];
    kcom_perfRecordStop(&(kperf.time.input));

    // Set CGRA kernel L/S pointers
    kcom_perfRecordStart( &(kperf.time.reprogramCols) );
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
      cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }
    kcom_perfRecordStop( &(kperf.time.reprogramCols) );

    // CGRA Execution
    kcom_perfRecordStart(   &(kperf.time.cgra) );
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
    // Process extra rows/cols for non multiple dimensions
    processExtraRowsA();
    // Wait until CGRA is done
    while(cgra_intr_flag==0) {
      wait_for_interrupt();
    }
    kcom_perfRecordStop(   &(kperf.time.cgra) );

    // Remove the padding if added
    kcom_perfRecordStart(&(kperf.time.padding));
    if (COLS_B % BLOCK_SIZE != 0){
      remove_padding(matrixC, ROWS_A, COLS_B, padded_COLS_B);
    }
    kcom_perfRecordStop(&(kperf.time.padding));

    checkErrors();
  }

  showPerformance(&kperf, 1);
  
  return EXIT_SUCCESS;
}



void add_padding(int *matrix, int rows, int cols, int padded_cols) {
  // Desplazar elementos y añadir ceros al final de cada fila
  for (int i = rows - 1; i >= 0; i--) {
    for (int j = cols - 1; j >= 0; j--) {
      matrix[i * padded_cols + j] = matrix[i * cols + j];
    }
    // Rellenar el padding con ceros
    for (int j = cols; j < padded_cols; j++) {
      matrix[i * padded_cols + j] = 0;
    }
  }
}

void remove_padding(int *matrix, int rows, int cols, int padded_cols) {
  // Desplazar elementos hacia la izquierda, eliminando el padding
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] = matrix[i * padded_cols + j];
    }
  }
}

void processExtraRowsA(){
  // Extra A rows
  for(int rA = ROWS_A - ROWS_A%BLOCK_SIZE; rA < ROWS_A; rA++){
    for(int cB = 0; cB < padded_COLS_B; cB++){
      int sum = 0;
      for(int k = 0; k < COLS_A; k++){
       sum += matrixA[rA*COLS_A + k] * matrixB[k*padded_COLS_B + cB];
      }
      matrixC[rA*padded_COLS_B + cB] = sum;
    } 
  }
}


// Check if the SW and CGRA executions give the same result
void checkErrors(){
  int errors = 0;
  for(int i = 0; i < ROWS_A*COLS_B; i++ ){
    if(outSW[i]!=matrixC[i]){
      errors++;
    }
  }
  printf("\rErrors: %d\n", errors);

  if(errors>0){
    printMatrix(matrixC, ROWS_A, COLS_B);
    printMatrix(outSW, ROWS_A, COLS_B);
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
      matrixA[i*cols+j] = (i*cols+j+1)%100;
    }
  }
}

// Software matrix multiplication
void mmulSoftware(int32_t * out){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        out[i*COLS_B+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
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
  printf("A:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, COLS_A, COLS_B);
  printf("Sw: %d\n", kperf->time.sw.spent_cy);
  if (full){
    printf("Load: %d\n", kperf->time.load.spent_cy);
    printf("Program cols: %d\n", kperf->time.reprogramCols.spent_cy);
    printf("Input: %d\n", kperf->time.input.spent_cy);
    printf("Cgra: %d\n", kperf->time.cgra.spent_cy);
    printf("Padding: %d\n", kperf->time.padding.spent_cy);
  }
  int32_t overhead = kperf->time.input.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy + kperf->time.padding.spent_cy;
  //printf("\rOverhead: %d\n", overhead);
  printf("Total cgra: %d\n", overhead + kperf->time.cgra.spent_cy); 
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
