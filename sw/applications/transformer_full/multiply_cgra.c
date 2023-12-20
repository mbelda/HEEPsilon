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
#include "cgra_x_heep.h"
#include "multiply_cgra.h"
#include "hart.h"


/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

// CGRA input and output buffers
static int32_t cgra_input[CGRA_N_COLS][4]    __attribute__ ((aligned (4)));

// Plic controller variables
volatile bool cgra_intr_flag;

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void multiply_cgra(int32_t * matrixC, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, cgra_t * cgra, uint8_t cgra_slot)
{ 
  int ROWS_C = rowsA;
  int COLS_C = colsB;
  int COLS_A = colsA; 

  if(ROWS_C % CGRA_N_COLS != 0 || COLS_C % CGRA_N_ROWS != 0){
    mmulSoftware(matrixC, matrixA, rowsA, colsA, matrixB, colsB);
  } else {
    // Prepare the input vector for the CGRA
    // Col 0: &B[0][0], nItLoopColsC, &A[0][0], &C[0][3]
    cgra_input[0][0] = &matrixB[0];
    cgra_input[0][1] = COLS_C/CGRA_N_ROWS;
    cgra_input[0][2] = &matrixA[0];
    cgra_input[0][3] = &matrixC[3];
    // Col 1: &C[1][0], &B[0][1], nItLoopsColsA, &A[1][0]
    cgra_input[1][0] = &matrixC[COLS_C];
    cgra_input[1][1] = &matrixB[1];
    cgra_input[1][2] = COLS_A;
    cgra_input[1][3] = &matrixA[COLS_A];
    // Col 2: &A[2][0], &C[2][1], &B[0][2], nItLoopColsC
    cgra_input[2][0] = &matrixA[2*COLS_A];
    cgra_input[2][1] = &matrixC[2*COLS_C+1];
    cgra_input[2][2] = &matrixB[2];
    cgra_input[2][3] = COLS_C/CGRA_N_ROWS;
    // Col 3: nItLoopRowsC, &A[3][0], &C[3][2], &B[0][3], 
    cgra_input[3][0] = ROWS_C/CGRA_N_COLS;
    cgra_input[3][1] = &matrixA[3*COLS_A];
    cgra_input[3][2] = &matrixC[3*COLS_C+2];
    cgra_input[3][3] = &matrixB[3];

    // Set CGRA kernel L/S pointers
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
      cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }

    // CGRA Execution
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, 1 );
    // Wait until CGRA is done
    while(cgra_intr_flag==0) {
      wait_for_interrupt();
    }
  }
}

// Software matrix multiplication
void mmulSoftware(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB){
  for(int i = 0; i < rowsA; i++){
    for(int j=0;j < colsB; j++){
      for(int k=0; k < colsA; k++){
        out[i*colsB+j] += matrixA[i*colsA+k]*matrixB[k*colsB+j];
      }
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
