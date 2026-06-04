/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : ReLu                                                         **
** author   : Maria Jose Belda (mbelda@ucm.e                               **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 21/07/2025                                                   **
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

#include "cgra_bitstream.h"
#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

// Dataset
#include "dataset.h"
#include "performance.h"


/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/



/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Pirnt metrics
void printMetrics();
// Initialize the CGRA
void initCGRA();

void check_errors();
void processExtraRowsAColsB();
void printAsMatrix(int *array, int rows, int cols);

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
#define CGRA_COL_INPUT_SIZE 7
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));


/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
    printf("Gemm meth execution with sizes %dx%dx%d\n", NI, NK, NJ);
    init_csr_counters();
    // Initialize the CGRA
    initCGRA();

    printf("CPU Execution\n");
    reset_csr_counters();
    gemm_cpu(expected_result);
    read_csr_counters();

    // Enable and reset the CGRA performance counters
    cgra_perf_cnt_enable(&cgra, 1);
    cgra_perf_cnt_reset( &cgra );

    //printf("Running gemm for size %dx%dx%d...\n", NI, NK, NJ);


    int nRowsA = NI;
    if (NI%4 == 3){
        // Special case
        nRowsA = NI +1;
    }
    printf("CGRA Config cycles\n");
    reset_csr_counters();
    // Prepare the input vector for the CGRA
    // ----------------------
    // &B[0][0]          &C[0][1]        &A[0][0]       nRowsBlocksC
    // nColsBlocksC      &B[0][1]        &C[1][2]       &A[1][0]
    // &A[2][0]          loopColsA       &B[0][2]       &C[2][3]
    // &C[3][0]          &A[3][0]        nColsBlocksC   &B[0][3]
    // ----------------------
    // -4*colsB          colsA           alpha       -
    // -                 -4*colsB        colsA       alpha
    // beta              -               -4*colsB    colsA
    // colsA             beta            -           -4*colsB
    int nItLoopColsA = NK;
    int nColsBlocksC = NJ/CGRA_N_ROWS;
    int nRowsBlocksC = nRowsA/CGRA_N_ROWS;
    // Col 0
    cgra_input[0][0] = &inputY[0];
    cgra_input[0][1] = nColsBlocksC;
    cgra_input[0][2] = &inputX[2*NK];
    cgra_input[0][3] = &inputZ[3*NJ];
    cgra_input[0][4] = -4*NJ;
    cgra_input[0][5] = BETA;
    cgra_input[0][6] = NK;
    // Col 1
    cgra_input[1][0] = &inputZ[1];
    cgra_input[1][1] = &inputY[1];
    cgra_input[1][2] = nItLoopColsA;
    cgra_input[1][3] = &inputX[3*NK];
    cgra_input[1][4] = NK;
    cgra_input[1][5] = -4*NJ;
    cgra_input[1][6] = BETA;
    // Col 2
    cgra_input[2][0] = &inputX[0];
    cgra_input[2][1] = &inputZ[NJ+2];
    cgra_input[2][2] = &inputY[2];
    cgra_input[2][3] = nColsBlocksC; // Duplicate for faster spread
    cgra_input[2][4] = ALPHA;
    cgra_input[2][5] = NK;
    cgra_input[2][6] = -4*NJ;
    // Col 3
    cgra_input[3][0] = nRowsBlocksC;
    cgra_input[3][1] = &inputX[NK];
    cgra_input[3][2] = &inputZ[2*NJ+3];
    cgra_input[3][3] = &inputY[3];
    cgra_input[3][4] = ALPHA;
    cgra_input[3][5] = NK;
    cgra_input[3][6] = -4*NJ;

    // Set CGRA kernel L/S pointers
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }
    read_csr_counters();


    // CGRA Execution
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, GEMM );

    // Wait until CGRA is done
    while(cgra_intr_flag==0) {
    wait_for_interrupt();
    }

    uint32_t sw_time;

    // Check errrors
    check_errors();
    printMetrics();

    


    return EXIT_SUCCESS;
}

void printAsMatrix(int *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", array[i * cols + j]);
        }
        printf("\n"); // Salto de línea tras cada fila
    }
}

void processExtraRowsAColsB(){
  int ROWS_A = NI;
  int COLS_A = NK;
  int COLS_B = NJ;
  int BLOCK_SIZE = 4;
  int rAMax = ROWS_A;
  // Extra A rows
  if (ROWS_A%4 != 3){
    for(int rA = ROWS_A - ROWS_A%BLOCK_SIZE; rA < ROWS_A; rA++){
      for(int cB = 0; cB < COLS_B; cB++){
        int sum = 0;
        for(int k = 0; k < COLS_A; k++){
          sum += inputX[rA*COLS_A + k] * inputY[k*COLS_B + cB];
        }
        inputZ[rA*COLS_B + cB] = ALPHA * sum + BETA * inputZ[rA*COLS_B + cB];
      } 
    }
    rAMax = ROWS_A - ROWS_A%BLOCK_SIZE;
  }

  // Extra cols B
  for(int cB = COLS_B - COLS_B%BLOCK_SIZE; cB < COLS_B; cB++){
    for(int rA = 0; rA < rAMax; rA++){
      int sum = 0;
      for(int k = 0; k < COLS_A; k++){
        sum += inputX[rA*COLS_A + k] * inputY[k*COLS_B + cB];
      }
      inputZ[rA*COLS_B + cB] = ALPHA * sum + BETA * inputZ[rA*COLS_B + cB];
    } 
  }
}

void gemm_cpu(int32_t * output){
  int i,j,k;
  int32_t sum;
  for(i = 0; i < NI; i ++) {
        for(j = 0; j < NJ; j ++) {
            sum = 0;
            for(k = 0; k < NK; k++) {
                sum += inputX[i * NK + k] * inputY[k * NJ + j];
            }
            output[i * NJ + j] = ALPHA * sum + BETA * inputZ[i * NJ + j];
        }
    }
}

// Check errors
void check_errors() {

    int error = 0;
    for(int i = 0; i < NI * NJ; i++) {
        if(inputZ[i] != expected_result[i]) {
          error++;
        }
    }

    if(error) {
        printf("FAIL with %d errors!!!\n\r", error);
        /*printAsMatrix(inputZ, NI, NJ);
        printf("Expected result:\n\r");
        printAsMatrix(expected_result, NI, NJ);*/
    } else {
        printf("OK!\n\r");
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



// Print metrics
void printMetrics(){
  // Performance counter display
  int column_idx = 0;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
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
