/*
                              *******************
******************************* C SOURCE FILE *******************************
** **
** project  : MMUL Benchmark                                               **
** author   : Maria Jose Belda (mbelda@ucm.es)                             **
** filename : main.c                                                       **
** version  : 3                                                            **
** date     : 2026                                                         **
** **
*****************************************************************************
** **
** Copyright (c) UCM                                                       **
** All rights reserved.                                                    **
** **
*****************************************************************************
*/

/***************************************************************************/
/***************************************************************************/

/**
* @file   main.c
* @date   04/03/2025
* @brief  An application to run a simple MMUL matrix multiplication on CGRA.
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

// Dataset (Debe contener NI, NK, NJ, A, B y C_expected)
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

void handler_irq_cgra(uint32_t id);
void printMetrics();
void initCGRA();
void check_errors();

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/

volatile bool               cgra_intr_flag;
static cgra_t               cgra;
static uint8_t              cgra_slot;

// La columna 2 requiere ahora un máximo de 6 elementos de configuración para MMUL
#define CGRA_COL_INPUT_SIZE 6
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE] __attribute__ ((aligned (4)));

// Output matrix
int32_t C[]; 

/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

int main()
{
  // Initialize the CGRA
  initCGRA();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  printf("Running MMUL for size %dx%dx%d...\n\r", NI, NK, NJ);

  // Iteraciones calculadas según lógica de Python para MMUL
  int loopIit = (NI / 4) - 1;
  int loopJit = (NJ / 4) - 1;
  int loopKit = NK - 1;

  // Direcciones base de las matrices
  uint32_t first_addr_A = (uint32_t)&A[0];
  uint32_t first_addr_B = (uint32_t)&B[0];
  uint32_t first_addr_C = (uint32_t)&C[0]; // Destino de los datos de salida

  // Direcciones calculadas para el mapeo por fila/columna en memoria
  uint32_t addr_A_0_0 = first_addr_A;
  uint32_t addr_A_1_0 = first_addr_A + (1 * NK * 4);
  uint32_t addr_A_2_0 = first_addr_A + (2 * NK * 4);
  uint32_t addr_A_3_0 = first_addr_A + (3 * NK * 4);

  uint32_t addr_B_0_0 = first_addr_B;
  uint32_t addr_B_0_1 = first_addr_B + 4;
  uint32_t addr_B_0_2 = first_addr_B + 8;
  uint32_t addr_B_0_3 = first_addr_B + 12;

  uint32_t addr_C_0_2 = first_addr_C + 8;
  uint32_t addr_C_1_3 = first_addr_C + (1 * NJ * 4) + 12;
  uint32_t addr_C_2_0 = first_addr_C + (2 * NJ * 4);
  uint32_t addr_C_3_1 = first_addr_C + (3 * NJ * 4) + 4;

  // Llenado de buffers respetando el mapeo exacto de Python para MMUL:
  
  // Col 0: [addr_A_0_0, loopIit, addr_C_2_0, addr_B_0_0, NK]
  cgra_input[0][0] = addr_A_0_0;
  cgra_input[0][1] = loopIit;
  cgra_input[0][2] = addr_C_2_0;
  cgra_input[0][3] = addr_B_0_0;
  cgra_input[0][4] = NK;

  // Col 1: [addr_B_0_1, addr_A_1_0, addr_C_3_1, NK, NJ]
  cgra_input[1][0] = addr_B_0_1;
  cgra_input[1][1] = addr_A_1_0;
  cgra_input[1][2] = addr_C_3_1;
  cgra_input[1][3] = NK;
  cgra_input[1][4] = NJ;

  // Col 2: [addr_C_0_2, addr_B_0_2, addr_A_2_0, loopKit, NJ, NK]
  cgra_input[2][0] = addr_C_0_2;
  cgra_input[2][1] = addr_B_0_2;
  cgra_input[2][2] = addr_A_2_0;
  cgra_input[2][3] = loopKit;
  cgra_input[2][4] = NJ;
  cgra_input[2][5] = NK;

  // Col 3: [loopJit, addr_C_1_3, addr_B_0_3, addr_A_3_0, NJ]
  cgra_input[3][0] = loopJit;
  cgra_input[3][1] = addr_C_1_3;
  cgra_input[3][2] = addr_B_0_3;
  cgra_input[3][3] = addr_A_3_0;
  cgra_input[3][4] = NJ;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, METH_KERNEL_ID);

  // Wait until CGRA is done
  while(cgra_intr_flag == 0) {
    wait_for_interrupt();
  }

  printMetrics();
  check_errors();

  return EXIT_SUCCESS;
}


void check_errors() {
    int error = 0;
    int total_elements = NI * NJ;
    for(int i = 0; i < total_elements; i++) {
        if(C[i] != C_expected[i]) {
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
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, handler_irq_cgra);

  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  const uint32_t mask = 1 << 11;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  cgra_slot = cgra_get_slot(&cgra);
}

// Print metrics
void printMetrics(){
  printf("CGRA kernel executed: %d\n\r", cgra_perf_cnt_get_kernel(&cgra));
  int column_idx = 0;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
}

void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}