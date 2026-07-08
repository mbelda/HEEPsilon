/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : MVT                                                          **
** author   : Maria Jose Belda (mbelda@ucm.es)                             **
** filename : main.c                                                       **
** version  : 2                                                            **
** date     : 08/07/2026                                                   **
** **
*****************************************************************************
** **
** Copyright (c) UCM                                                       **
** All rights reserved.                                                    **
** **
*****************************************************************************
*/

/**************************************************************************edited*/
/***************************************************************************/

/**
* @file   main.c
* @date   08/07/2026
* @brief  An application to run MVT Part 2 (x2 = x2 + A^T * y2) on CGRA.
*
*/

/****************************************************************************/
/** **/
/* MODULES USED                                 */
/** **/
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
/** **/
/* DEFINITIONS AND MACROS                            */
/** **/
/****************************************************************************/



/****************************************************************************/
/** **/
/* PROTOTYPES OF LOCAL FUNCTIONS                       */
/** **/
/****************************************************************************/

// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Print metrics
void printMetrics();
// Initialize the CGRA
void initCGRA();
// Check output
void check_errors();

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/

// Plic controller variables
volatile bool               cgra_intr_flag;

// CGRA variables
static cgra_t               cgra;
static uint8_t              cgra_slot;



// CGRA input buffers (Mantenido a 8 para asegurar espacio suficiente de configuración)
#define CGRA_COL_INPUT_SIZE 8
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));


/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

void main()
{

  // Initialize the CGRA
  initCGRA();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  printf("Running mvt part2 for size N=%d...", N);
  

  // Prepare the input vector for the CGRA
  // ----------------------
  // Config values (MVT Part 2)
  // ----------------------
  /*# Config vals
    # &A[0][0]      N              &x2[2]         N
    # &x2[4]        N              &A[0][6]       N
    # N             &A[0][9]       N              &x2[11]
    # N             &x2[13]        N              &A[0][15]
    # -----------------------------------------------------
    #               loopJit                       &y2[0]
    #               loopIit                       
    #               &y2[0]                       
    #                                             
*/

  int loopIit = N - 1;
  int loopJit = (N / 16) - 1;

  // Map the pointers directly to memory variables defined in dataset.h para MVT P2
  uint32_t first_addr_A  = (uint32_t)&A[0];
  uint32_t first_addr_x2 = (uint32_t)&x2[0]; // Entrada inicial y destino acumulado final
  uint32_t first_addr_y2 = (uint32_t)&y2[0]; 

  // Col 0
  cgra_input[0][0] = first_addr_A;
  cgra_input[0][1] = first_addr_x2 + (4 * 4);
  cgra_input[0][2] = N;
  cgra_input[0][3] = N;

  // Col 1
  cgra_input[1][0] = N;
  cgra_input[1][1] = N;
  cgra_input[1][2] = first_addr_A + (9 * 4);
  cgra_input[1][3] = first_addr_x2 + (13 * 4);
  cgra_input[1][4] = loopJit;
  cgra_input[1][5] = loopIit;
  cgra_input[1][6] = first_addr_y2;

  // Col 2
  cgra_input[2][0] = first_addr_x2 + (2 * 4);
  cgra_input[2][1] = first_addr_A + (6 * 4);
  cgra_input[2][2] = N;
  cgra_input[2][3] = N;

  // Col 3
  cgra_input[3][0] = N;
  cgra_input[3][1] = N;
  cgra_input[3][2] = first_addr_x2 + (11 * 4);
  cgra_input[3][3] = first_addr_A + (15 * 4);
  cgra_input[3][4] = first_addr_y2;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  // Cambiado al ID del Kernel correspondiente a la Parte 2 de MVT
  cgra_set_kernel( &cgra, cgra_slot, MVT_P2_KERNEL_ID);

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
    // En MVT Parte 2, la comprobación de errores se realiza sobre el vector x2
    for(int i = 0; i < N; i++) {
        if(x2[i] != x2_expected[i]) {
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
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/