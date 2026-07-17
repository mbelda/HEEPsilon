/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : 3-Loops Convolution Benchmark                                 **
** author   : Maria Jose Belda (mbelda@ucm.es)                             **
** filename : main.c                                                       **
** version  : 2                                                            **
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
* @date   16/07/2026
* @brief  An application to run a 3-Loops Convolution on CGRA.
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

// Dataset generado dinámicamente
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
int cgra_sum_out;

// El mapeo requiere de 9 elementos de configuración por columna
#define CGRA_COL_INPUT_SIZE 8
#define CGRA_COL_OUTPUT_SIZE 1
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE] __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE] __attribute__ ((aligned (4)));

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

  printf("Running 3-Loops Conv for C=%d, KH=%d, KW=%d, IH=%d, IW=%d...\n\r", 
          CHANNELS_PER_GROUP, KERNEL_H, KERNEL_W, INPUT_H, INPUT_W);

  // Direcciones base físicas de memoria (coincidiendo con las variables del .h)
  uint32_t first_addr_in = (uint32_t)&input_lider[0];
  uint32_t first_addr_w  = (uint32_t)&weights[0];

  // Constantes de desplazamiento calculadas con las nuevas macros del .h
  uint32_t size_ch_im = 4 * TAM_CANAL_INPUT;
  uint32_t size_ch_f  = 4 * KERNEL_H * KERNEL_W;

  // Direcciones de canales específicos según mapeo
  uint32_t addr_Im_0  = first_addr_in + (0 * size_ch_im);
  uint32_t addr_Im_6  = first_addr_in + (6 * size_ch_im);
  uint32_t addr_Im_9  = first_addr_in + (9 * size_ch_im);
  uint32_t addr_Im_15 = first_addr_in + (15 * size_ch_im);

  uint32_t addr_F_2   = first_addr_w + (2 * size_ch_f);
  uint32_t addr_F_4   = first_addr_w + (4 * size_ch_f);
  uint32_t addr_F_11  = first_addr_w + (11 * size_ch_f);
  uint32_t addr_F_13  = first_addr_w + (13 * size_ch_f);

  // Iteradores de bucle adaptados a las macros del generador
  int loopCIt  = (CHANNELS_PER_GROUP / 16) - 1;
  int loopKHIt = KERNEL_H - 1;
  int loopKWIt = KERNEL_W - 1;

  // Llenado de buffers respetando el mapeo del generador:
  
  // Col 0: [addr_Im_0, addr_F_4, size_ch_f, size_ch_im, IH, KH, KW, loopCIt, 0]
  cgra_input[0][0] = addr_Im_0;
  cgra_input[0][1] = addr_F_4;
  cgra_input[0][2] = size_ch_f;
  cgra_input[0][3] = size_ch_im;
  cgra_input[0][4] = INPUT_H;
  cgra_input[0][5] = KERNEL_H;
  cgra_input[0][6] = KERNEL_W;
  cgra_input[0][7] = loopCIt;

  // Col 1: [size_ch_im, size_ch_f, addr_Im_9, addr_F_13, IW, KW, IH, KH, store_address]
  cgra_input[1][0] = size_ch_im;
  cgra_input[1][1] = size_ch_f;
  cgra_input[1][2] = addr_Im_9;
  cgra_input[1][3] = addr_F_13;
  cgra_input[1][4] = INPUT_W;
  cgra_input[1][5] = KERNEL_W;
  cgra_input[1][6] = INPUT_H;
  cgra_input[1][7] = KERNEL_H;

  // Col 2: [addr_F_2, addr_Im_6, size_ch_im, size_ch_f, KH, loopKHIt, IW, KW, 0]
  cgra_input[2][0] = addr_F_2;
  cgra_input[2][1] = addr_Im_6;
  cgra_input[2][2] = size_ch_im;
  cgra_input[2][3] = size_ch_f;
  cgra_input[2][4] = KERNEL_H;
  cgra_input[2][5] = loopKHIt;
  cgra_input[2][6] = INPUT_W;
  cgra_input[2][7] = KERNEL_W;

  // Col 3: [size_ch_f, size_ch_im, addr_F_11, addr_Im_15, KW, loopKWIt, KH, IW, 0]
  cgra_input[3][0] = size_ch_f;
  cgra_input[3][1] = size_ch_im;
  cgra_input[3][2] = addr_F_11;
  cgra_input[3][3] = addr_Im_15;
  cgra_input[3][4] = KERNEL_W;
  cgra_input[3][5] = loopKWIt;
  cgra_input[3][6] = KERNEL_H;
  cgra_input[3][7] = INPUT_W;

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx],  col_idx );
    cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, CONV_KERNEL_ID);

  // Wait until CGRA is done
  while(cgra_intr_flag == 0) {
    wait_for_interrupt();
  }

  cgra_sum_out = cgra_output[1][0];

  printMetrics();
  check_errors();

  return EXIT_SUCCESS;
}

void check_errors() {
    printf("Check CGRA output result:\n\r");
    if(cgra_sum_out == EXPECTED_SUM) {
        printf("OK (Obtained: %d | Expected: %d)\n\r", cgra_sum_out, EXPECTED_SUM);
        printf("SUCCESS!\n\r");
    } else {
        printf("FAIL -> Expected: %d | CGRA Obtained: %d\n\r", EXPECTED_SUM, cgra_sum_out);
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