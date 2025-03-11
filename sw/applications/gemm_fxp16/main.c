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

// Plic controller variables
volatile bool               cgra_intr_flag;

// CGRA variables
static cgra_t               cgra;
static uint8_t              cgra_slot;

// CGRA input buffers
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
int32_t matrixA[(ROWS_A+1)*COLS_A];
int32_t matrixB[COLS_A*COLS_B];
int32_t matrixC[(ROWS_A+1)*COLS_B];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void gemm_on_cgra(int32_t *mA, int32_t * mB, int32_t * mC, int rowsA, int colsA, int colsB){
  if(rowsA < CGRA_N_COLS || colsB < CGRA_N_ROWS){
    mmulSoftware(mC);
  } else {
    
    // Check for an special case
    int nRowsAaux = rowsA;
    if (rowsA%4 == 3){
      nRowsAaux = rowsA +1;
    }

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
    int nItLoopColsA = colsA;
    int nColsBlocksC = colsB/CGRA_N_ROWS;
    int nRowsBlocksC = nRowsAaux/CGRA_N_ROWS;
    // Col 0
    cgra_input[0][0] = &matrixB[0];
    cgra_input[0][1] = nColsBlocksC;
    cgra_input[0][2] = &matrixA[2*colsA];
    cgra_input[0][3] = &matrixC[3*colsB];
    cgra_input[0][4] = -4*colsB;
    cgra_input[0][5] = colsA;
    // Col 1
    cgra_input[1][0] = &matrixC[1];
    cgra_input[1][1] = &matrixB[1];
    cgra_input[1][2] = nItLoopColsA;
    cgra_input[1][3] = &matrixA[3*colsA];
    cgra_input[1][4] = colsA;
    cgra_input[1][5] = -4*colsB;
    // Col 2
    cgra_input[2][0] = &matrixA[0];
    cgra_input[2][1] = &matrixC[colsB+2];
    cgra_input[2][2] = &matrixB[2];
    cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
    cgra_input[2][4] = colsA;
    cgra_input[2][5] = -4*colsB;
    // Col 3
    cgra_input[3][0] = nRowsBlocksC;
    cgra_input[3][1] = &matrixA[colsA];
    cgra_input[3][2] = &matrixC[2*colsB+3];
    cgra_input[3][3] = &matrixB[3];
    cgra_input[3][4] = colsA;
    cgra_input[3][5] = -4*colsB;

    // Set CGRA kernel L/S pointers
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
      cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }

    // CGRA Execution
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, GEMM_FXP );
    // Process extra rows/cols for non multiple dimensions
    processExtraRowsAColsB();
    // Wait until CGRA is done
    while(cgra_intr_flag==0) {
      wait_for_interrupt();
    }
  }
}

void main()
{
  printf("Launching mmul_os opt v2 for dimension %dx%dx%d\n", ROWS_A, COLS_A, COLS_B);
  fillMatrixInputs(matrixA, ROWS_A, COLS_A);
  fillMatrixInputs(matrixB, COLS_A, COLS_B);

  // Initialize the CGRA
  initCGRA(); // This only needs to be done once

  // Run gemm
  gemm_on_cgra(matrixA, matrixB, matrixC, ROWS_A, COLS_A, COLS_B);
  
  
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

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}

/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
