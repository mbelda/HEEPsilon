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

#include "cgra_x_heep.h"
#include "cgra_bitstream.h"
#include "multiply_cgra.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"
#include "cgra.h"


/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/

// Size of the input buffer for the CGRA
#define CGRA_COL_INPUT_SIZE 4

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/


// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);


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


/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void multiply_cgra(int * matrixA, int ROWS_A, int COLS_A, int * matrixB, int COLS_B, int * matrixC)
{
  // ----------------------
    // &B[0][0]          &C[0][1]        &A[0][0]       nRowsBlocksC
    // nColsBlocksC      &B[0][1]        &C[1][2]       &A[1][0]
    // &A[2][0]          loopColsA       &B[0][2]       &C[2][3]
    // &C[3][0]          &A[3][0]        nColsBlocksC   &B[0][3]
    // ----------------------
    // -4*colsB          colsA           -              -
    // -                 -4*colsB        colsA          -
    // -                 -               -4*colsB       colsA
    // colsA             -               -              -4*colsB

    int nRowsA = ROWS_A;
    if (ROWS_A%4 !=0){
      // Special case
      nRowsA = ROWS_A + 4 - ROWS_A%4;
    }
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

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, 1 );
  processExtraRowsAColsB(matrixA, ROWS_A, COLS_A, matrixB, COLS_B, matrixC);
  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
}

void processExtraRowsAColsB(int * matrixA, int ROWS_A, int COLS_A, int * matrixB, int COLS_B, int * matrixC){
  int rAMax = ROWS_A;
  // Extra A rows
  if (ROWS_A%4 == 0){
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

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}

void countersInit(){
  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );
}

/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
