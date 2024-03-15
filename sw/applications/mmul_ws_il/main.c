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

#include "transformer.h"
#include "cgra_bitstream.h"
#include "cgra_x_heep.h"
#include "performance.h"

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

// Sizes of the input and ouput buffers of the CGRA
#define CGRA_COL_INPUT_SIZE 6 // Since the matrixes are loaded with LWI this does not depend on their size
#define CGRA_COL_OUTPUT_SIZE CGRA_N_COLS*(CGRA_N_ROWS-1)*(ROWS_A/CGRA_N_COLS)

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Matrix multiplication using the standard three loops
void mmulSoftware(int32_t * output);
// Fill input matrixes with numbers
void fillMatrixInputs();
// Fill output buffers with zeroes
void fillOutputZeroes();
// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Record the cycle number at the start
void kcom_perfRecordStart( kcom_time_diff_t *perf );
// Record the cycle number and compute the total cycles
void kcom_perfRecordStop( kcom_time_diff_t *perf );
// Process extra A rows
void processExtraARows(int rB, int cB);
// Process extra A cols
void processExtraACols();
// Process extra B cols
void processExtraBCols(int rB);
// Move output from buffer to matrixC
void moveOutput(int32_t ** buffer, int cB);

void printMatrix(int * matrix, int rows, int cols);
void printCGRAOutput(int32_t ** output);

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

// Performance variable
static kcom_perf_t  kperf;

// Plic controller variables
volatile bool               cgra_intr_flag;

// CGRA variables
static cgra_t               cgra;
static uint8_t              cgra_slot;

// CGRA input and output buffers
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));
static int32_t __attribute__((section(".xheep_data_interleaved"))) cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));
static int32_t __attribute__((section(".xheep_data_interleaved"))) cgra_output_aux[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

// Input and output matrixes
static int32_t __attribute__((section(".xheep_data_interleaved"))) matrixA[ROWS_A*COLS_A];
static int32_t __attribute__((section(".xheep_data_interleaved"))) matrixB[ROWS_B*COLS_B];
static int32_t matrixC[ROWS_C*COLS_C];
int32_t outSW[ROWS_C*COLS_C];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  fillMatrixInputs();
  fillOutputZeroes();

  // Init timer
  timerInit();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  // Minimum dims: 4x3x12
  if(ROWS_A < CGRA_N_COLS || COLS_A < BLOCK_SIZE || COLS_B < (CGRA_N_ROWS-1)*CGRA_N_COLS){
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(matrixC);
    kcom_perfRecordStop(&(kperf.time.sw));
  } else {
    kcom_perfRecordStart(   &(kperf.time.cgra) );
    // Initialize the CGRA
    //kcom_perfRecordStart(&(kperf.time.load));
    initCGRA();
    //kcom_perfRecordStop(&(kperf.time.load));

    // Prepare the input vector for the CGRA
    //kcom_perfRecordStart(&(kperf.time.input));
    // Load 4*4*COLS_A
    for(int j = 0; j < CGRA_N_COLS; j++){
      cgra_input[j][2] = 4*4*COLS_A;
    }
    // Load the number of iterations of the rows_A loop
    cgra_input[1][3] = ROWS_A/CGRA_N_COLS;
    //kcom_perfRecordStop(&(kperf.time.input));

    int firstTime = 1;
    // Divide the rows of B in blocks
    for (int rB = 0; rB < ROWS_B-ROWS_B%BLOCK_SIZE; rB += BLOCK_SIZE){ 
      // Prepare the input vector for the CGRA
      //kcom_perfRecordStart(&(kperf.time.input));
      int cA = rB;
      int32_t addrA = (int32_t) (matrixA + cA);
      // Load &A + {0,1,2,3}*COLS_A
      for(int j = 0; j < CGRA_N_COLS; j++){
        cgra_input[j][0] = (int32_t) (addrA) + j*COLS_A*4;
      }
      //kcom_perfRecordStop(&(kperf.time.input));

      // Extra B cols if not multiple of CGRA_N_COLS*(CGRA_N_ROWS-1)
      int processedExtraBCols = 0;
      if (COLS_B%(CGRA_N_COLS*(CGRA_N_ROWS-1)) == 0) processedExtraBCols = 1;
      // Divide the cols of B in blocks
      for (int cB = 0; cB + CGRA_N_COLS*(CGRA_N_ROWS-1) <= COLS_B; cB += CGRA_N_COLS*(CGRA_N_ROWS-1)){
        // Count even/odd iterations
        int evenOdd = 0;
        // Prepare the input vector for the CGRA
        //kcom_perfRecordStart( &(kperf.time.input) );
        int cont[4] = {1,1,1,1};
        int subRowB = 0;
        for(int j = 0; j < CGRA_N_COLS; j++){
          cgra_input[j][cont[j]] = &(matrixB[(rB+subRowB)*COLS_B+cB+j]);
          cont[j] += 2;
        }
        subRowB++;
        cont[1]+=1;
        // Load the rest of B
        while(subRowB < BLOCK_SIZE){
          for(int j = 0; j < CGRA_N_COLS; j++){
            cgra_input[j][cont[j]] = &(matrixB[(rB+subRowB)*COLS_B+cB+j]);
            cont[j] += 1;
          }
          subRowB++;
        }
        //kcom_perfRecordStop( &(kperf.time.input) );

        // Set CGRA kernel L/S pointers
        //kcom_perfRecordStart( &(kperf.time.reprogramCols) );
        for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
          cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
          // Change the output buffer
          uint32_t output_buffer = (uint32_t) cgra_output[col_idx];
          if (evenOdd){
            output_buffer = (uint32_t) cgra_output_aux[col_idx];
          }
          cgra_set_write_ptr( &cgra, cgra_slot, output_buffer, col_idx );
        }
        //kcom_perfRecordStop( &(kperf.time.reprogramCols) );

        // CGRA Execution
        //kcom_perfRecordStart(   &(kperf.time.cgra) );
        cgra_intr_flag = 0;
        cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
        // Wait until CGRA is done, while waiting process extra columns and rows and moving output
    
        int processedExtraARows = 0; // Extra A rows if not multiple of 4
        if (ROWS_A%CGRA_N_COLS == 0) processedExtraARows = 1;
        int processedOutput = 0;
        if (firstTime) processedOutput = 1;

        while(cgra_intr_flag==0) {
          // Process extra rows/cols
          if (!processedExtraARows){
            processExtraARows(rB, cB);
            processedExtraARows = 1;
          }
          if (!processedExtraBCols){
            processExtraBCols(rB);
            processedExtraBCols = 1;
          }
          // Move output
          if (!processedOutput){
            if (evenOdd){
              moveOutput(cgra_output, cB);
            }
            else {
              moveOutput(cgra_output_aux, cB);
            }
            processedOutput = 1;
          }
          //wait_for_interrupt();          
        }

        if (evenOdd){
          printCGRAOutput(cgra_output);
        } else {
          printCGRAOutput(cgra_output_aux);
        }

        //kcom_perfRecordStop(   &(kperf.time.cgra) );

        // Ensure the extra rows have been processed
        if (!processedExtraARows){
          processExtraARows(rB, cB);
        }
        // Ensure the extra cols have been processed
        if (!processedExtraBCols){
          processExtraBCols(rB);
        }

        printMatrix(matrixC, ROWS_C, COLS_C);

        firstTime=0;
        evenOdd = (evenOdd+1)%2;
      }
    } 
    // Process the extra cols of A
    processExtraACols();
    kcom_perfRecordStop(   &(kperf.time.cgra) );
  }
  
  // Software 
  kcom_perfRecordStart(&(kperf.time.sw));
  mmulSoftware(outSW);
  kcom_perfRecordStop(&(kperf.time.sw));

  checkErrors();
  showPerformance(&kperf);
  printf("Software:\n");
  printMatrix(outSW, ROWS_C, COLS_C);
  printf("CRGA:\n");
  printMatrix(matrixC, ROWS_C, COLS_C);
  
  return EXIT_SUCCESS;
}


void printCGRAOutput(int32_t ** output){
  // output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]
  for (int i = 0; i < CGRA_N_COLS; i++){
    printf("Col %d: [", i);
    for(int j=0; j < CGRA_COL_OUTPUT_SIZE; j++){
      printf("%d, ", output[i][j]);
    }
    printf("]\n");
  }
}

void moveOutput(int32_t ** buffer, int cB){
  int contAux = 0;
  for(int it = 0; it < ROWS_A-ROWS_A%CGRA_N_COLS; it++){
    int blockA = it/CGRA_N_COLS;  
    for(int i=0; i < CGRA_N_ROWS-1; i++, contAux++){
      for(int j=0; j < CGRA_N_COLS; j++){
        int colC = cB + i*CGRA_N_ROWS + j;
        int rowC = (it + j)%CGRA_N_COLS + blockA*CGRA_N_COLS; 
        matrixC[rowC*COLS_C + colC] += buffer[j][contAux];
      }
    }
  }
}

// Process the extra A cols
void processExtraACols(){
  for(int colA = COLS_A-COLS_A%BLOCK_SIZE; colA < COLS_A; colA++){
    for(int colB=0; colB < COLS_B; colB++){
      for(int rowA = 0; rowA  < ROWS_A; rowA++){
        int rowB=colA;
        matrixC[rowA*COLS_C+colB] += matrixA[rowA*COLS_A+colA]*matrixB[rowB*COLS_B+colB];
      }
    }
  }
}

// Process the extra B cols
void processExtraBCols(int rB){
  for(int colB = COLS_B-COLS_B%(CGRA_N_COLS*(CGRA_N_ROWS-1)); colB < COLS_B; colB++){
    for(int rA = 0; rA < ROWS_A; rA++){
      for(int rowB = rB; rowB < rB + BLOCK_SIZE; rowB++){
        int cA = rowB;
        matrixC[rA*COLS_C+colB] += matrixA[rA*COLS_A+cA]*matrixB[rowB*COLS_B+colB];
      }
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

// Process the extra A rows
void processExtraARows(int rB, int cB){
  int cA = rB;
  int nBlocksA = ROWS_A/CGRA_N_COLS;
  for(int rA = nBlocksA*CGRA_N_COLS; rA < ROWS_A; rA++){
    for(int c = cB; c < cB + CGRA_N_COLS*(CGRA_N_ROWS-1); c++){
      for(int subColA = 0; subColA < BLOCK_SIZE; subColA++){
        int subRowB = subColA; 
        matrixC[rA*COLS_C + c] += matrixA[rA*COLS_A + cA + subColA] * matrixB[(subRowB+rB)*COLS_B + c];
      }
    } 
  }
}

// Check if the SW and CGRA executions give the same result
void checkErrors(){
  int errors = 0;
  for(int i = 0; i < ROWS_C*COLS_C; i++ ){
    if(outSW[i]!=matrixC[i]){
      errors++;
    }
  }
  printf("\rErrors: %d\n", errors);
}

// Initialize the CGRA
void initCGRA(){
  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, (void *) &handler_irq_cgra);

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

// Fill matrix inputs
void fillMatrixInputs(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0; j < COLS_A; j++){
      matrixA[i*COLS_A+j] = (i*COLS_A+j+1)%100;
    }
  }

  for(int i = 0; i < ROWS_B; i++){
    for(int j=0;j < COLS_B; j++){
      matrixB[i*COLS_B+j] = (i*COLS_B+j+1)%100;
    }
  }
}

// Software matrix multiplication
void mmulSoftware(int32_t * out){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        out[i*COLS_C+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
      }
    }
  }
}

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}

// Display the performance values
showPerformance( kcom_perf_t* kperf){
  printf("\rA:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B);
  printf("\r-----------------------------\n");
  printf("\rSw: %d\n", kperf->time.sw.spent_cy);
  /*printf("\rLoad: %d\n", kperf->time.load.spent_cy);
  printf("\rReprogram cols: %d\n", kperf->time.reprogramCols.spent_cy);
  printf("\rInput: %d\n", kperf->time.input.spent_cy);
  printf("\rCgra: %d\n", kperf->time.cgra.spent_cy);
  printf("\r-----------------------------\n");
  */
  int32_t overhead = kperf->time.input.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy;
  //printf("\rOverhead: %d\n", overhead);
  printf("\rTotal cgra: %d\n", overhead + kperf->time.cgra.spent_cy); 
}

// Fill output buffers with zeroes
void fillOutputZeroes(){
  for(int i = 0; i < ROWS_C; i++){
    for(int j=0; j < COLS_C; j++){
      matrixC[i*COLS_C+j] = 0;
      outSW[i*COLS_C+j] = 0;
    }
  }
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
