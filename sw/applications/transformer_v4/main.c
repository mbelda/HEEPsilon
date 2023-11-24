/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : CGRA-X-HEEP                                                  **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 05/04/23                                                     **
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
* @date   05/04/23
* @brief  An application to run a number of kernels under a same given
* structure.
*
*/

#define _KERNELS_C

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

/****************************************************************************/
/**                                                                        **/
/*                        TYPEDEFS AND STRUCTURES                           */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

void mmulSoftware();
void fillMatrixInputs();
void handler_irq_ext(uint32_t id);
void processExtraRowsCols(int rB, int cB);
void kcom_perfRecordStart( kcom_time_diff_t *perf );
void kcom_perfRecordStop( kcom_time_diff_t *perf );

/****************************************************************************/
/**                                                                        **/
/*                           EXPORTED VARIABLES                             */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

static kcom_perf_t  kperf;

// Plic controller variables
volatile bool               cgra_intr_flag;

static cgra_t               cgra;
static uint8_t              cgra_slot;

#define CGRA_COL_INPUT_SIZE CGRA_N_ROWS*BLOCK_SIZE
#define CGRA_COL_OUTPUT_SIZE CGRA_N_COLS*(CGRA_N_ROWS-1)*(ROWS_A/CGRA_N_COLS)
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE]   __attribute__ ((aligned (4)));

static int32_t outSW[ROWS_C*COLS_C];

int32_t matrixA[ROWS_A*COLS_A];
int32_t matrixB[ROWS_B*COLS_B];
int32_t matrixC[ROWS_C*COLS_C] = {0};

volatile int cgraIteration;
static int numIterations = ROWS_A/CGRA_N_COLS;

int8_t offsetVec[CGRA_N_COLS] = {0,1,2,3}; // Depends on the cgra_n_cols. Modified MANUALLY!!

/****************************************************************************/
/**                                                                        **/
/*                           EXPORTED FUNCTIONS                             */
/**                                                                        **/
/****************************************************************************/

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  // TODO: Caso en el que la matriz sea tan pequeña que no haya ni un bloque
  fillMatrixInputs();

  // Init timer
  timerInit();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  kcom_perfRecordStart( &(kperf.time.load) );
  initCGRA();
  kcom_perfRecordStop( &(kperf.time.load) );

  kcom_perfRecordStart( &(kperf.time.input1) );

  // Load 4*4*COLS_A
  for(int j = 0; j < CGRA_N_COLS; j++){
    cgra_input[j][2] = 4*4*COLS_A;
  }
  // Load nIterations
  cgra_input[1][3] = numIterations;

  kcom_perfRecordStop( &(kperf.time.input1) );

  int CGRA_block_B = CGRA_N_COLS*(CGRA_N_ROWS-1);
  for (int rB = 0; rB < ROWS_B; rB += BLOCK_SIZE){ // TODO: Gestionar no multiplos (ROWS_B no multiplo de BLOCK_SIZE)
    kcom_perfRecordStart( &(kperf.time.input2) );
    int cA = rB;
    int32_t addrA = (int32_t) (matrixA + cA);
    // Load &A + {0,1,2,3}*COLS_A
    for(int j = 0; j < CGRA_N_COLS; j++){
      cgra_input[j][0] = (int32_t) (addrA) + j*COLS_A*4;
    }
    kcom_perfRecordStop( &(kperf.time.input2) );
    for (int cB = 0; cB + CGRA_block_B <= COLS_B; cB += CGRA_block_B){ //TODO: gestionar no multiplos (COLS_B no multiplo de CGRA_block_B)
      kcom_perfRecordStart( &(kperf.time.input3) );
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
      kcom_perfRecordStop( &(kperf.time.input3) );


      /*printf("\rB: %d\n", matrixB);
      for(int i=0; i < 5 ; i++){
        printf("\r%d %d %d %d\n", cgra_input[0][i], cgra_input[1][i], cgra_input[2][i], cgra_input[3][i]);
      }*/
      kcom_perfRecordStart( &(kperf.time.reprogramCols) );
      // Set CGRA kernel L/S pointers
      for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
        cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
        cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
      }
      kcom_perfRecordStop( &(kperf.time.reprogramCols) );

      // CGRA Execution
      kcom_perfRecordStart(   &(kperf.time.cgra) );
      cgra_intr_flag = 0;
      cgra_set_kernel( &cgra, cgra_slot, TRANSFORMER );
      // Wait until CGRA is done, while waiting process extra columns and rows
      int processed = 0;
      while(cgra_intr_flag==0) {
        /*if (!processed){
          processExtraRowsCols(rB, cB);
          processed = 1;
        }*/
        wait_for_interrupt();
      }
      kcom_perfRecordStop(   &(kperf.time.cgra) );

      /*if (!processed){
        processExtraRowsCols(rB, cB);
      }*/
      cgra_intr_flag = 0;

      kcom_perfRecordStart( &(kperf.time.output) );
      // Move CGRA output
      int contAux = 0;
      for(int it = 0; it < ROWS_A; it++){
        int blockA = it/CGRA_N_COLS;  
        for(int i=0; i < CGRA_N_ROWS-1; i++, contAux++){
          for(int j=0; j < CGRA_N_COLS; j++){
            int colC = cB + i*CGRA_N_ROWS + j;
            int rowC = (it + j)%CGRA_N_COLS + blockA*CGRA_N_COLS; 
            //printf("\rC[%d][%d] += cOut[%d][%d] = %d\n", rowC, colC, j, contAux, cgra_output[j][contAux]);
            matrixC[rowC*COLS_C + colC] += cgra_output[j][contAux];
          }
        }
      }
      kcom_perfRecordStop( &(kperf.time.output) );
    }
  }  

  checkErrors();

  showPerformance( &kperf);
  
  //return EXIT_SUCCESS;
  return 0;
}

void processExtraRowsCols(int rB, int cB){
  // Process extra rows for A
  for(int rA = ROWS_A/CGRA_N_COLS*CGRA_N_COLS; rA < ROWS_A; rA++){
    for(int c = cB; c < cB + CGRA_N_COLS*CGRA_N_ROWS; c++){
      for(int b = 0; b < BLOCK_SIZE; b++){
        matrixC[rA*COLS_C + c] += matrixA[rA*COLS_A + rB + b] * matrixB[(b+rB)*COLS_B + c];
      }
    } 
  }
}

void checkErrors(){
  // Software 
  kcom_perfRecordStart(   &(kperf.time.sw) );
  mmulSoftware();
  kcom_perfRecordStop(   &(kperf.time.sw) );

  int errors = 0;
  for(int i = 0; i < ROWS_C*COLS_C; i++ ){
    if(outSW[i]!=matrixC[i]){
      errors++; 
      //printf("\rC[%d][%d] %d != %d\n", i/COLS_C, i%COLS_C, outSW[i],matrixC[i] );
    }
  }
  
  printf("\rErrors: %d\n", errors);
  
  /*
  printf("\rSW\n");
  for(int i=0; i < ROWS_C; i++){
    printf("\r");
    for(int j=0; j < COLS_C; j++){
      printf("%d ", outSW[i*COLS_C+j]);
    }
    printf("\n");
  }

  printf("\rC\n");
  for(int i=0; i < ROWS_C; i++){
    printf("\r");
    for(int j=0; j < COLS_C; j++){
      printf("%d ", matrixC[i*COLS_C+j]);
    }
    printf("\n");
  }*/
}

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
  kcom_perfRecordStart(   &(kperf.time.bitstream) );
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  kcom_perfRecordStop(   &(kperf.time.bitstream) );

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
void mmulSoftware(){
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_B; j++){
      for(int k=0; k < COLS_A; k++){
        outSW[i*COLS_B+j] += matrixA[i*COLS_A+k]*matrixB[k*COLS_B+j];
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

showPerformance( kcom_perf_t* kperf){
  printf("\rA:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B);
  printf("\r-----------------------------\n");
  printf("\rSw: %d\n", kperf->time.sw.spent_cy);
  printf("\rLoad: %d\n", kperf->time.load.spent_cy);
  printf("\r    Bitstream: %d\n", kperf->time.bitstream.spent_cy);
  printf("\rReprogram cols: %d\n", kperf->time.reprogramCols.spent_cy);
  printf("\rInput: %d\n", kperf->time.input1.spent_cy+kperf->time.input2.spent_cy+kperf->time.input3.spent_cy);
  printf("\rOutput: %d\n", kperf->time.output.spent_cy);
  printf("\rCgra: %d\n", kperf->time.cgra.spent_cy);
  printf("\r-----------------------------\n");
  int32_t overhead = kperf->time.input1.spent_cy+kperf->time.input2.spent_cy+kperf->time.input3.spent_cy + kperf->time.output.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy;
  printf("\rOverhead: %d\n", overhead);
  printf("\rTotal cgra: %d\n", overhead + kperf->time.cgra.spent_cy);
  //float speedup = ((float) kperf->time.sw.spent_cy) / ((float) (overhead + kperf->time.cgra.spent_cy));
  //printf("\rSpeedup: %f\n", speedup);  
}



/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
