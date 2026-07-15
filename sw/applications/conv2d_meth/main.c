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

void check_errors();

void extraRowsCols( int *image_data, int *filter_data, int rowsIm, int colsIm, int *output);

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
#define CGRA_COL_INPUT_SIZE 6
#define BLOCK_SIZE 4
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

  printf("Running conv2d for image size %d...", IM_HEIGHT);
    

  // Prepare the input vector for the CGRA
  // ----------------------
  // Config values
  // ----------------------
  /*# 4*nCols               -16*colBlocks           16*nCols            nItLoop1
    # nItLoop2              4*nCols                 -16*colBlocks       16*nCols
    # 16*nCols              -                       4*nCols             -16*colBlocks
    # -16*colBlocks         16*nCols                -                   4*nCols
    # ----------------------
    # &Im[0]                &out[1*nCols + 2]      &Filter[0]              -
    # -                     &Im[1*nCols + 1]       &out[2*nCols + 3]       -
    # -                     -                      &Im[2*nCols + 2]     &out[3*nCols + 4]      
    # &out[4*nCols + 1]     -                      -                    &Im[3*nCols + 3]
*/
  int nColBlocks = IM_WIDTH / BLOCK_SIZE;
  int nRowBlocks = IM_HEIGHT / BLOCK_SIZE;
  int nItLoop1 = nRowBlocks;
  int nItLoop2 = nColBlocks;
  // Col 0
  cgra_input[0][0] = 4*IM_WIDTH;
  cgra_input[0][1] = nItLoop2;
  cgra_input[0][2] = 16*IM_WIDTH;
  cgra_input[0][3] = -16*nColBlocks;
  cgra_input[0][4] = &image[0];
  cgra_input[0][5] = &output[4*IM_WIDTH + 1];
  // Col 1
  cgra_input[1][0] = -16*nColBlocks;
  cgra_input[1][1] = 4*IM_WIDTH;
  cgra_input[1][2] = 16*IM_WIDTH;
  cgra_input[1][3] = &output[1*IM_WIDTH + 2];
  cgra_input[1][4] = &image[1*IM_WIDTH + 1];
  cgra_input[1][5] = 0; // Not used in this column
  // Col 2
  cgra_input[2][0] = 16*IM_WIDTH;
  cgra_input[2][1] = -16*nColBlocks;
  cgra_input[2][2] = 4*IM_WIDTH;
  cgra_input[2][3] = &filter[0];
  cgra_input[2][4] = &output[2*IM_WIDTH + 3];
  cgra_input[2][5] = &image[2*IM_WIDTH + 2];
  // Col 3
  cgra_input[3][0] = nItLoop1;
  cgra_input[3][1] = 16*IM_WIDTH;
  cgra_input[3][2] = -16*nColBlocks;
  cgra_input[3][3] = 4*IM_WIDTH;
  cgra_input[3][4] = &output[3*IM_WIDTH + 4];
  cgra_input[3][5] = &image[3*IM_WIDTH + 3];

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }

  // CGRA Execution
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, CONV );

  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);
  CSR_WRITE(CSR_REG_MCYCLE, 0);
  extraRowsCols( image, filter,IM_HEIGHT, IM_WIDTH, output);
  CSR_READ(CSR_REG_MCYCLE, &sw_time);
  printf("SW cycles: %lu\n", sw_time);

  printMetrics();
  check_errors();


  return EXIT_SUCCESS;
}

void check_errors() {

    int error = 0;
    for(int i = 0; i < IM_HEIGHT * IM_WIDTH; i++) {
        if(output[i] != expected_result[i]) {
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
  column_idx = 1;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 2;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 3;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
}

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}

void extraRowsCols( int *image_data, int *filter_data, int rowsIm, int colsIm, int *output) {
    int valid_rows = rowsIm - 2;
    int valid_cols = colsIm - 2;

    int full_blocks_rows = valid_rows / 4;
    int full_blocks_cols = valid_cols / 4;

    int extra_rows = valid_rows % 4;
    int extra_cols = valid_cols % 4;

    int start_row = 1 + 4 * full_blocks_rows;
    int start_col = 1 + 4 * full_blocks_cols;

    // Procesar filas sobrantes
    for (int i = start_row; i < start_row + extra_rows; i++) {
        for (int j = 1; j < colsIm - 1; j++) {
            int acc = 0;
            for (int fi = 0; fi < 3; fi++) {
                for (int fj = 0; fj < 3; fj++) {
                    int im_i = i + fi - 1;
                    int im_j = j + fj - 1;
                    int im_index = im_i * colsIm + im_j;
                    int filt_index = fi * 3 + fj;
                    acc += image_data[im_index] * filter_data[filt_index];
                }
            }
            output[i * colsIm + j] = acc;
        }
    }

    // Procesar columnas sobrantes (sin repetir filas ya procesadas)
    for (int i = 1; i < rowsIm - 1 - extra_rows; i++) {
        for (int j = start_col; j < start_col + extra_cols; j++) {
            int acc = 0;
            for (int fi = 0; fi < 3; fi++) {
                for (int fj = 0; fj < 3; fj++) {
                    int im_i = i + fi - 1;
                    int im_j = j + fj - 1;
                    int im_index = im_i * colsIm + im_j;
                    int filt_index = fi * 3 + fj;
                    acc += image_data[im_index] * filter_data[filt_index];
                }
            }
            output[i * colsIm + j] = acc;
        }
    }
}


/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
