/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : 3MM                                                          **
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

#define CGRA_COL_INPUT_SIZE 2 // Size of the input buffer for the CGRA
#define BLOCK_SIZE 4

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);

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


/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  printf("Launching 3MM for dimension %dx%dx%dx%dx%d\n", ROWS_A, COLS_A, COLS_B, COLS_C, COLS_D);

  // Initialize the CGRA
  initCGRA();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  // 16x20x18
  // 40x60x50
  loadKernelCGRA(cgra_imem_bitstream_16x20x18, cgra_kmem_bitstream_16x20x18);

  // Prepare the input vector for the CGRA
  // ----------------------
  // -             -            &output[0]    &input1[0]
  // -             -            -             &input2[0] 

  // Col 0
  // Col 1
  // Col 2
  cgra_input[2][0] = &matrixE[0];
  // Col 3
  cgra_input[3][0] = &matrixA[0];
  cgra_input[3][1] = &matrixB[0];

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, MMUL_COMPI );
  
  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  printf("Finished E\n");



  // 18x24x22
  // 50x80x70
  loadKernelCGRA(cgra_imem_bitstream_18x24x22, cgra_kmem_bitstream_18x24x22);

    // Prepare the input vector for the CGRA
  // ----------------------
  // -             -            &output[0]    &input1[0]
  // -             -            -             &input2[0] 
  // Col 0
  // Col 1
  // Col 2
  cgra_input[2][0] = &matrixF[0];
  // Col 3
  cgra_input[3][0] = &matrixC[0];
  cgra_input[3][1] = &matrixD[0];


  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, MMUL_COMPI );

  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }


  printf("Finished F\n");

  // 16x18x22
  // 40x50x70
  loadKernelCGRA(cgra_imem_bitstream_16x18x22, cgra_kmem_bitstream_16x18x22);

  // Prepare the input vector for the CGRA
  // ----------------------
  // -             -            &output[0]    &input1[0]
  // -             -            -             &input2[0] 

  // Col 0
  // Col 1
  // Col 2
  cgra_input[2][0] = &matrixG[0];
  // Col 3
  cgra_input[3][0] = &matrixE[0];
  cgra_input[3][1] = &matrixF[0];


  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, MMUL_COMPI );

  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }


  printf("Finished G\n");
  checkErrors(expected_result, matrixG, ROWS_E, COLS_F);
  printMetrics();

  
  return EXIT_SUCCESS;
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

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  // Select request slot of CGRA
  cgra_slot = cgra_get_slot(&cgra);
}

void loadKernelCGRA(uint32_t imem_bitstream, uint32_t kmem_bitstream){
  // Load kernel
  cgra_cmem_init(imem_bitstream, kmem_bitstream);
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
