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

#include "dataset.h"
#include "cgra_bitstream.h"
#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

#include "performance.h"



/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/

// Sizes of the input and ouput buffers of the CGRA
#define CGRA_COL_INPUT_SIZE 12 
#define CGRA_COL_OUTPUT_SIZE 3*ROWS_A // Each RC stores an element for each row of A
#define BLOCK_SIZE 3

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/


// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);

int checkErrors(int32_t * cgra_out, int32_t *expected_out);


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
static int32_t __attribute__((section(".xheep_data_interleaved"))) cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  init_csr_counters();
  
  // Minimum dims: 4x3x12
  printf("Running mmul_ws_il_rv_summit with A: %dx%d, B: %dx%d\n", ROWS_A, COLS_A, COLS_A, COLS_B);
  // Initialize the CGRA
  printf("Initialize CGRA\n");
  initCGRA();
  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );


  for (int rB = 0; rB < ROWS_B; rB += 3){
    for(int cB = 0; cB < COLS_B; cB += 12){

      printf("Prepare input for B block (%d, %d)\n", rB, cB);
      reset_csr_counters();
      // Col 0
      cgra_input[0][0] = &matrixA[0];
      cgra_input[0][1] = matrixB[rB*COLS_B+cB];
      cgra_input[0][2] = matrixB[rB*COLS_B+cB+4];
      cgra_input[0][3] = matrixB[rB*COLS_B+cB+8];
      cgra_input[0][4] = 4*COLS_A;
      cgra_input[0][5] = matrixB[(rB+1)*COLS_B+cB];
      cgra_input[0][6] = matrixB[(rB+1)*COLS_B+cB+4];
      cgra_input[0][7] = matrixB[(rB+1)*COLS_B+cB+8];
      cgra_input[0][8] = matrixB[(rB+2)*COLS_B+cB];
      cgra_input[0][9] = matrixB[(rB+2)*COLS_B+cB+4];
      cgra_input[0][10] = matrixB[(rB+2)*COLS_B+cB+8];

      // Col 1
      cgra_input[1][0] = &matrixA[4*COLS_A];
      cgra_input[1][1] = matrixB[rB*COLS_B+cB+1];
      cgra_input[1][2] = matrixB[rB*COLS_B+cB+5];
      cgra_input[1][3] = matrixB[rB*COLS_B+cB+9];
      cgra_input[1][4] = 4*COLS_A;
      cgra_input[1][5] = matrixB[(rB+1)*COLS_B+cB+1];
      cgra_input[1][6] = matrixB[(rB+1)*COLS_B+cB+5];
      cgra_input[1][7] = matrixB[(rB+1)*COLS_B+cB+9];
      cgra_input[1][8] = ROWS_A / 4;
      cgra_input[1][9] = matrixB[(rB+2)*COLS_B+cB+1];
      cgra_input[1][10] = matrixB[(rB+2)*COLS_B+cB+5];
      cgra_input[1][11] = matrixB[(rB+2)*COLS_B+cB+9];

      // Col 2
      cgra_input[2][0] = &matrixA[8*COLS_A];
      cgra_input[2][1] = matrixB[rB*COLS_B+cB+2];
      cgra_input[2][2] = matrixB[rB*COLS_B+cB+6];
      cgra_input[2][3] = matrixB[rB*COLS_B+cB+10];
      cgra_input[2][4] = 4*COLS_A;
      cgra_input[2][5] = matrixB[(rB+1)*COLS_B+cB+2];
      cgra_input[2][6] = matrixB[(rB+1)*COLS_B+cB+6];
      cgra_input[2][7] = matrixB[(rB+1)*COLS_B+cB+10];
      cgra_input[2][8] = matrixB[(rB+2)*COLS_B+cB+2];
      cgra_input[2][9] = matrixB[(rB+2)*COLS_B+cB+6];
      cgra_input[2][10] = matrixB[(rB+2)*COLS_B+cB+10];

      // Col 3
      cgra_input[3][0] = &matrixA[12*COLS_A];
      cgra_input[3][1] = matrixB[rB*COLS_B+cB+3];
      cgra_input[3][2] = matrixB[rB*COLS_B+cB+7];
      cgra_input[3][3] = matrixB[rB*COLS_B+cB+11];
      cgra_input[3][4] = 4*COLS_A;
      cgra_input[3][5] = matrixB[(rB+1)*COLS_B+cB+3];
      cgra_input[3][6] = matrixB[(rB+1)*COLS_B+cB+7];
      cgra_input[3][7] = matrixB[(rB+1)*COLS_B+cB+11];
      cgra_input[3][8] = matrixB[(rB+2)*COLS_B+cB+3];
      cgra_input[3][9] = matrixB[(rB+2)*COLS_B+cB+7];
      cgra_input[3][10] = matrixB[(rB+2)*COLS_B+cB+11];

      for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
        cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
        cgra_set_write_ptr( &cgra, cgra_slot, cgra_output[col_idx], col_idx );
      }
      read_csr_counters();
      printf("Run kernel for B block (%d, %d)\n", rB, cB);
      reset_csr_counters();
      // CGRA Execution
      cgra_intr_flag = 0;
      cgra_set_kernel( &cgra, cgra_slot, 1 );

      while(cgra_intr_flag==0) {
        //wait_for_interrupt();          
      }
      read_csr_counters();

      printf("Write back results for B block (%d, %d)\n", rB, cB);
      reset_csr_counters();

      int rAux = 0;
      int rMod[4] = {0,1,2,3};
      int counter[4] = {0,0,0,0};
      for(int i = 0; i < ROWS_A / 4; i++){
        for(int j = 0; j < 4; j++){
          for (int col = 0; col < 4; col++){
            matrixC[(rAux + rMod[col]*COLS_B + cB + 0 + col)] += cgra_output[col][counter[col]];
            counter[col]++;
            matrixC[(rAux + rMod[col]*COLS_B + cB + 4 + col)] += cgra_output[col][counter[col]];
            counter[col]++;
            matrixC[(rAux + rMod[col]*COLS_B + cB + 8 + col)] += cgra_output[col][counter[col]];
            counter[col]++;
            rMod[col]++;
            rMod[col] = rMod[col] % 4;
          }
          rAux += 4;
        }
      }
      reset_csr_counters();

    }
  }
  
  int errors = checkErrors(matrixC, cpu_out);
  if (errors){
    printf("Test failed with %d errors (out of %d elems)\n", errors, ROWS_C*COLS_C);
  } else {
    printf("OK\n");
  }
  
  return EXIT_SUCCESS;
}




// Check if the SW and CGRA executions give the same result
int checkErrors(int32_t * cgra_out, int32_t *expected_out){
  int errors = 0;
  for(int i = 0; i < ROWS_C*COLS_C; i++ ){
    if(cgra_out[i]!=expected_out[i]){
      errors++;
    }
  }
  return errors;
  
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
  reset_csr_counters();
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  read_csr_counters();

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  // Select request slot of CGRA
  cgra_slot = cgra_get_slot(&cgra);
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
