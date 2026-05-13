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

#include "dataset.h"
#include "cgra_bitstream.h"
#include "cgra_x_heep.h"

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
void mmul_cpu(int * in1, int* in2, int * out, int rows1, int cols1, int cols2);
// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Process non multiple number of rows and cols
void processExtraRowsAColsB();
void printMetrics();

void checkErrors(int * res, int * in, int rows, int cols);

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/


// Plic controller variables
volatile bool               cgra_intr_flag;

// CGRA variables
static cgra_t               cgra;
static uint8_t              cgra_slot;

// CGRA input and output buffers
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
int32_t outSWE[ROWS_A*COLS_B];
int32_t outSWF[ROWS_C*COLS_D];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  printf("Launching mmul_os opt v2 for dimension %dx%dx%d\n", ROWS_A, COLS_A, COLS_B);

  // Initialize the CGRA
  initCGRA();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  // Initialize the CGRA
  initCGRA();
  

  // Prepare the input vector for the CGRA
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
  int nRowsBlocksC = ROWS_A/CGRA_N_ROWS;
  // Col 0
  cgra_input[0][0] = &matrixB[0];
  cgra_input[0][1] = nColsBlocksC;
  cgra_input[0][2] = &matrixA[2*COLS_A];
  cgra_input[0][3] = &matrixE[3*COLS_B];
  cgra_input[0][4] = -4*COLS_B;
  cgra_input[0][5] = COLS_A;
  // Col 1
  cgra_input[1][0] = &matrixE[1];
  cgra_input[1][1] = &matrixB[1];
  cgra_input[1][2] = nItLoopColsA;
  cgra_input[1][3] = &matrixA[3*COLS_A];
  cgra_input[1][4] = COLS_A;
  cgra_input[1][5] = -4*COLS_B;
  // Col 2
  cgra_input[2][0] = &matrixA[0];
  cgra_input[2][1] = &matrixE[COLS_B+2];
  cgra_input[2][2] = &matrixB[2];
  cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
  cgra_input[2][4] = COLS_A;
  cgra_input[2][5] = -4*COLS_B;
  // Col 3
  cgra_input[3][0] = nRowsBlocksC;
  cgra_input[3][1] = &matrixA[COLS_A];
  cgra_input[3][2] = &matrixE[2*COLS_B+3];
  cgra_input[3][3] = &matrixB[3];
  cgra_input[3][4] = COLS_A;
  cgra_input[3][5] = -4*COLS_B;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
  // Process extra rows/cols for non multiple dimensions
  //processExtraRowsCols(matrixA, matrixB, matrixE, ROWS_A, COLS_A, COLS_B);
  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  printf("Finished E\n"); 
  mmul_cpu(matrixA, matrixB, outSWE, ROWS_A, COLS_A, COLS_B);
  checkErrors(outSWE, matrixE, ROWS_A, COLS_B); 

  // Prepare the input vector for the CGRA
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
  nItLoopColsA = COLS_C;
  nColsBlocksC = COLS_D/CGRA_N_ROWS;
  nRowsBlocksC = ROWS_C/CGRA_N_ROWS;
  // Col 0
  cgra_input[0][0] = &matrixD[0];
  cgra_input[0][1] = nColsBlocksC;
  cgra_input[0][2] = &matrixC[2*COLS_C];
  cgra_input[0][3] = &matrixF[3*COLS_D];
  cgra_input[0][4] = -4*COLS_D;
  cgra_input[0][5] = COLS_C;
  // Col 1
  cgra_input[1][0] = &matrixF[1];
  cgra_input[1][1] = &matrixD[1];
  cgra_input[1][2] = nItLoopColsA;
  cgra_input[1][3] = &matrixC[3*COLS_C];
  cgra_input[1][4] = COLS_C;
  cgra_input[1][5] = -4*COLS_D;
  // Col 2
  cgra_input[2][0] = &matrixC[0];
  cgra_input[2][1] = &matrixF[COLS_D+2];
  cgra_input[2][2] = &matrixD[2];
  cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
  cgra_input[2][4] = COLS_C;
  cgra_input[2][5] = -4*COLS_D;
  // Col 3
  cgra_input[3][0] = nRowsBlocksC;
  cgra_input[3][1] = &matrixC[COLS_C];
  cgra_input[3][2] = &matrixF[2*COLS_D+3];
  cgra_input[3][3] = &matrixD[3];
  cgra_input[3][4] = COLS_C;
  cgra_input[3][5] = -4*COLS_D;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
  // Process extra rows/cols for non multiple dimensions
  processExtraRowsCols(matrixC, matrixD, matrixF, ROWS_C, COLS_C, COLS_D);
  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }


  printf("Finished F\n");
  mmul_cpu(matrixC, matrixD, outSWF, ROWS_C, COLS_C, COLS_D);
  checkErrors(outSWF, matrixF, ROWS_C, COLS_D);


  // Prepare the input vector for the CGRA
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
  nItLoopColsA = COLS_E;
  nColsBlocksC = COLS_F/CGRA_N_ROWS;
  nRowsBlocksC = COLS_E/CGRA_N_ROWS;
  // Col 0
  cgra_input[0][0] = &matrixF[0];
  cgra_input[0][1] = nColsBlocksC;
  cgra_input[0][2] = &matrixE[2*COLS_E];
  cgra_input[0][3] = &matrixG[3*COLS_F];
  cgra_input[0][4] = -4*COLS_F;
  cgra_input[0][5] = COLS_E;
  // Col 1
  cgra_input[1][0] = &matrixG[1];
  cgra_input[1][1] = &matrixF[1];
  cgra_input[1][2] = nItLoopColsA;
  cgra_input[1][3] = &matrixE[3*COLS_E];
  cgra_input[1][4] = COLS_E;
  cgra_input[1][5] = -4*COLS_F;
  // Col 2
  cgra_input[2][0] = &matrixE[0];
  cgra_input[2][1] = &matrixG[COLS_F+2];
  cgra_input[2][2] = &matrixF[2];
  cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
  cgra_input[2][4] = COLS_E;
  cgra_input[2][5] = -4*COLS_F;
  // Col 3
  cgra_input[3][0] = nRowsBlocksC;
  cgra_input[3][1] = &matrixE[COLS_E];
  cgra_input[3][2] = &matrixG[2*COLS_F+3];
  cgra_input[3][3] = &matrixF[3];
  cgra_input[3][4] = COLS_E;
  cgra_input[3][5] = -4*COLS_F;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
  // Process extra rows/cols for non multiple dimensions
  processExtraRowsCols(matrixE, matrixF, matrixG, ROWS_E, COLS_E, COLS_F);
  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }


  printf("Finished G\n");
  checkErrors(expected_result, matrixG, ROWS_E, COLS_F);
  printMetrics();

  
  return EXIT_SUCCESS;
}

void processExtraRowsCols(int *in1, int * in2, int * out, int rows1, int cols1, int cols2){
  int rAMax = rows1;
  // Extra A rows
  if (rows1%4 != 3){
    for(int rA = rows1 - rows1%BLOCK_SIZE; rA < rows1; rA++){
      for(int cB = 0; cB < cols2; cB++){
        int sum = 0;
        for(int k = 0; k < cols1; k++){
        sum += in1[rA*cols1 + k] * in2[k*cols2 + cB];
        }
        out[rA*cols2 + cB] = sum;
      } 
    }
    rAMax = rows1 - rows1%BLOCK_SIZE;
  }

  // Extra cols B
  for(int cB = cols2 - cols2%BLOCK_SIZE; cB < cols2; cB++){
    for(int rA = 0; rA < rAMax; rA++){
      int sum = 0;
      for(int k = 0; k < cols1; k++){
       sum += in1[rA*cols1 + k] * in2[k*cols2 + cB];
      }
      out[rA*cols2 + cB] = sum;
    } 
  }
}




void checkErrors(int * res, int * in, int rows, int cols){
  int errors = 0;
  
  for(int i = 0; i < rows*cols; i++ ){
    if(res[i]!=in[i]){
      errors++;
    }
  }

  if (errors > 0){
    printf("Errors: %d\n", errors);
  } else{
    printf("OK\n");
  }
  
}

void mmul_cpu(int * in1, int* in2, int * out, int rows1, int cols1, int cols2){
  for(int i = 0; i < rows1; i++){
    for(int j=0;j < cols2; j++){
      int sum = 0;
      for(int k=0; k < cols1; k++){
        sum += in1[i*COLS_A+k]*in2[k*COLS_B+j];
      }
      out[i*COLS_B+j] = sum;
    }
  }
}

// Initialize the CGRA
void initCGRA(){
  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, handler_irq_cgra);

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

// Print metrics
void printMetrics(){
  // Performance counter display
  printf("CGRA kernel executed: %d\n\r", cgra_perf_cnt_get_kernel(&cgra));
  int column_idx = 0;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
}




/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
