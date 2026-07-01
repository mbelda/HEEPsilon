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
// Check output
void check_errors();

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

  // Initialize the CGRA
  initCGRA();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  printf("Running gemver part2 for size %d...", N);
    


  // Prepare the input vector for the CGRA
  // ----------------------
  // Config values


  int loopIit = (N / 16) - 1;
  int loopJit = N;

  // Map the pointers directly to memory variables defined in dataset.h
  uint32_t first_addr_A = (uint32_t)&A[0];
  uint32_t first_addr_x = (uint32_t)&x[0];
  uint32_t first_addr_y = (uint32_t)&y[0];
  uint32_t first_addr_z = (uint32_t)&z[0];

  // Col 0
  cgra_input[0][0] = first_addr_A;
  cgra_input[0][1] = first_addr_x + (4 * 4);
  cgra_input[0][2] = N;
  cgra_input[0][3] = first_addr_z + (12 * 4);
  cgra_input[0][4] = BETA;
  cgra_input[0][5] = 0;    // Padding
  cgra_input[0][6] = 0;    // Padding

  // Col 1
  cgra_input[1][0] = first_addr_z + (1 * 4);
  cgra_input[1][1] = loopIit;
  cgra_input[1][2] = first_addr_A + (9 * 4);
  cgra_input[1][3] = first_addr_x + (13 * 4);
  cgra_input[1][4] = first_addr_y;
  cgra_input[1][5] = N;
  cgra_input[1][6] = BETA;

  // Col 2
  cgra_input[2][0] = first_addr_x + (2 * 4);
  cgra_input[2][1] = first_addr_A + (6 * 4);
  cgra_input[2][2] = first_addr_z + (10 * 4);
  cgra_input[2][3] = N;
  cgra_input[2][4] = BETA;
  cgra_input[2][5] = 0;    // Padding
  cgra_input[2][6] = 0;    // Padding

  // Col 3
  cgra_input[3][0] = N;
  cgra_input[3][1] = first_addr_z + (7 * 4);
  cgra_input[3][2] = first_addr_x + (11 * 4);
  cgra_input[3][3] = first_addr_A + (15 * 4);
  cgra_input[3][4] = first_addr_y;
  cgra_input[3][5] = BETA;
  cgra_input[3][6] = 0;    // Padding

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, PART2_KERNEL_ID);

  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }

  printMetrics();
  check_errors();


  return EXIT_SUCCESS;
}

void check_errors() {
    int error = 0;
    for(int i = 0; i < N; i++) {
        if(x[i] != x_expected[i]) {
          error++;
        }
    }

    if(error) {
        printf("FAIL with %d errors!!!\n\r", error);
    } else {
        printf("SUCCESS!\n\r");
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
  printf("CGRA kernel executed: %d\n\r", cgra_perf_cnt_get_kernel(&cgra));
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
