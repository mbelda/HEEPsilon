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

#include "cgra_bitstream.h"
#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

#include "multiply_cgra.h"


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
static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void multiply_cgra(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, int transpose)
{ 
  int rowsB = colsA;
  fillOutputZeroes(out, rowsA, colsB);

  if(rowsA < CGRA_N_COLS || colsA < BLOCK_SIZE || colsB < (CGRA_N_ROWS-1)*CGRA_N_COLS){
    mmulSoftware(out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
  } else {
    // Initialize the CGRA
    initCGRA();

    // Prepare the input vector for the CGRA
    // Load 4*4*COLS_A
    for(int j = 0; j < CGRA_N_COLS; j++){
      cgra_input[j][2] = 4*4*colsA;
    }
    // Load the number of iterations of the rows_A loop
    cgra_input[1][3] = rowsA/CGRA_N_COLS;

    // Divide the rows of B in blocks
    for (int rB = 0; rB < rowsB-rowsB%BLOCK_SIZE; rB += BLOCK_SIZE){ 
      // Prepare the input vector for the CGRA
      int cA = rB;
      int32_t addrA = (int32_t) (matrixA + cA);
      // Load &A + {0,1,2,3}*COLS_A
      for(int j = 0; j < CGRA_N_COLS; j++){
        cgra_input[j][0] = (int32_t) (addrA) + j*colsA*4;
      }

      int processedExtraBCols = 0;
      // Divide the cols of B in blocks
      for (int cB = 0; cB + CGRA_N_COLS*(CGRA_N_ROWS-1) <= colsB; cB += CGRA_N_COLS*(CGRA_N_ROWS-1)){
        // Prepare the input vector for the CGRA
        int cont[4] = {1,1,1,1};
        int subRowB = 0;
        for(int j = 0; j < CGRA_N_COLS; j++){
          cgra_input[j][cont[j]] = &(matrixB[(rB+subRowB)*colsB+cB+j]);
          cont[j] += 2;
        }
        subRowB++;
        cont[1]+=1;
        // Load the rest of B
        while(subRowB < BLOCK_SIZE){
          for(int j = 0; j < CGRA_N_COLS; j++){
            cgra_input[j][cont[j]] = &(matrixB[(rB+subRowB)*colsB+cB+j]);
            cont[j] += 1;
          }
          subRowB++;
        }

        // Set CGRA kernel L/S pointers
        for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
          cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
          cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
        }

        // CGRA Execution
        cgra_intr_flag = 0;
        cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
        // Wait until CGRA is done, while waiting process extra columns and rows
        int processedExtraARows = 0;
        while(cgra_intr_flag==0) {
          if (!processedExtraARows){
            processExtraARows(rB, cB, out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
            processedExtraARows = 1;
          }
          if (!processedExtraBCols){
            processExtraBCols(rB, out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
            processedExtraBCols = 1;
          }
        }

        // Ensure the extra rows have been processed
        if (!processedExtraARows){
          processExtraARows(rB, cB, out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
        }

        // Move CGRA output
        int contAux = 0;
        for(int it = 0; it < rowsA-rowsA%CGRA_N_COLS; it++){
          int blockA = it/CGRA_N_COLS;  
          for(int i=0; i < CGRA_N_ROWS-1; i++, contAux++){
            for(int j=0; j < CGRA_N_COLS; j++){
              int colC = cB + i*CGRA_N_ROWS + j;
              int rowC = (it + j)%CGRA_N_COLS + blockA*CGRA_N_COLS;
              if(transpose){
                out[colC*colsB + rowC] += cgra_output[j][contAux];
              } else {
                out[rowC*colsB + colC] += cgra_output[j][contAux];
              }
            }
          }
        }


        // Ensure the extra cols have been processed
        if (!processedExtraBCols){
          processExtraBCols(rB, out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
        }
      }
    } 
    // Process the extra cols of A
    processExtraACols(out, matrixA, rowsA, colsA, matrixB, colsB, transpose);
  }
}

// Process the extra A cols
void processExtraACols(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, int transpose){
  for(int colA = colsA-colsA%BLOCK_SIZE; colA < colsA; colA++){
    for(int colB=0; colB < colsB; colB++){
      for(int rowA = 0; rowA < rowsA; rowA++){
        int rowB=colA;
        if (transpose){
          out[colB*colsB+rowA] += matrixA[rowA*colsA+colA]*matrixB[rowB*colsB+colB];
        } else {
          out[rowA*colsB+colB] += matrixA[rowA*colsA+colA]*matrixB[rowB*colsB+colB];
        }
      }
    }
  }
}

// Process the extra B cols
void processExtraBCols(int rB, int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, int transpose){
  for(int colB = colsB-colsB%(CGRA_N_COLS*(CGRA_N_ROWS-1)); colB < colsB; colB++){
    for(int rA = 0; rA < rowsA; rA++){
      for(int rowB = rB; rowB < rB + BLOCK_SIZE; rowB++){
        int cA = rowB;
        if (transpose){
          out[colB*colsB+rA] += matrixA[rA*colsA+cA]*matrixB[rowB*colsB+colB];
        } else {
          out[rA*colsB+colB] += matrixA[rA*colsA+cA]*matrixB[rowB*colsB+colB];
        }
      }
    }
  }
}

// Process the extra A rows
void processExtraARows(int rB, int cB, int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, int transpose){
  int cA = rB;
  int nBlocksA = rowsA/CGRA_N_COLS;
  for(int rA = nBlocksA*CGRA_N_COLS; rA < rowsA; rA++){
    for(int c = cB; c < cB + CGRA_N_COLS*(CGRA_N_ROWS-1); c++){
      for(int subColA = 0; subColA < BLOCK_SIZE; subColA++){
        int subRowB = subColA; 
        if (transpose){
          out[c*colsB + rA] += matrixA[rA*colsA + cA + subColA] * matrixB[(subRowB+rB)*colsB + c];
        } else {
          out[rA*colsB + c] += matrixA[rA*colsA + cA + subColA] * matrixB[(subRowB+rB)*colsB + c];
        }
      }
    } 
  }
}

// Initialize the CGRA
void initCGRA(){
  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);

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


// Software matrix multiplication
void mmulSoftware(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, int transpose){
  for(int i = 0; i < rowsA; i++){
    for(int j=0;j < colsB; j++){
      for(int k=0; k < colsA; k++){
        if (transpose){
          out[j*colsB+i] += matrixA[i*colsA+k]*matrixB[k*colsB+j];
        } else {
          out[i*colsB+j] += matrixA[i*colsA+k]*matrixB[k*colsB+j];
        }
      }
    }
  }
}

// Interrupt controller variables
void handler_irq_ext(uint32_t id) {
  if( id == CGRA_INTR) {
    cgra_intr_flag = 1;
  }
}


// Fill output buffers with zeroes
void fillOutputZeroes(int32_t * output, int rows, int cols){
  for(int i = 0; i < rows; i++){
    for(int j=0; j < cols; j++){
      output[i*cols+j] = 0;
    }
  }
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
