/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : GeMM                                                         **
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

#include "sizes.h"
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
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Vector addition on cpu
void addVectorsCPU(int32_t * output);
// Fill input vectors with numbers
void fillVector(int * vec, int n);
// Handler for the CGRA interruption
void handler_irq_cgra(uint32_t id);
// Record the cycle number at the start
void kcom_perfRecordStart( kcom_time_diff_t *perf );
// Record the cycle number and compute the total cycles
void kcom_perfRecordStop( kcom_time_diff_t *perf );

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

// Measure performance variable
static kcom_perf_t  kperf;

// CGRA input buffers
#define CGRA_COL_INPUT_SIZE 12
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
static int32_t __attribute__((section(".xheep_data_interleaved"))) vecA[VECTOR_SIZE];
static int32_t __attribute__((section(".xheep_data_interleaved"))) vecB[VECTOR_SIZE];
static int32_t __attribute__((section(".xheep_data_interleaved"))) vecOut[VECTOR_SIZE];
static int32_t __attribute__((section(".xheep_data_interleaved"))) vecOutCPU[VECTOR_SIZE];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

#define DEBUG 0

void main()
{
  printf("Launching add vectors (v1) for vector size %d\n", VECTOR_SIZE);

  // Generate data
  fillVector(vecA, VECTOR_SIZE);
  fillVector(vecB, VECTOR_SIZE);

  // Init timer
  timerInit();

  // Enable and reset the CGRA performance counters
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  // Initialize the CGRA
  printf("Config cgra..."); 
  initCGRA(); // This only needs to be done once
  
  // Prepare the input vector for the CGRA
  // ----------------------
  // Config values
  // ----------------------
  // nIt           &A[0]         &A[1]         &A[2]
  // &A[3]         &A[4]         &A[5]         &A[6]
  // &A[7]         &A[8]         &A[9]         &A[10]
  // &A[11]        &A[12]        &A[13]        &A[14]
  // ----------------------
  // -             &B[0]         &B[1]         &B[2]
  // &B[3]         &B[4]         &B[5]         &B[6]
  // &B[7]         &B[8]         &B[9]         &B[10]
  // &B[11]        &B[12]        &B[13]        &B[14]
  // ----------------------
  // -             &C[0]         &C[1]         &C[2]
  // &C[3]         &C[4]         &C[5]         &C[6]
  // &C[7]         &C[8]         &C[9]         &C[10]
  // &C[11]        &C[12]        &C[13]        &C[14]
  
  int nIt = VECTOR_SIZE/15;
  // Col 0
  cgra_input[0][0] = nIt;
  cgra_input[0][1] = &vecA[3];
  cgra_input[0][2] = &vecA[7];
  cgra_input[0][3] = &vecA[11];
  cgra_input[0][4] = &vecB[3];
  cgra_input[0][5] = &vecB[7];
  cgra_input[0][6] = &vecB[11];
  cgra_input[0][7] = &vecOut[3];
  cgra_input[0][8] = &vecOut[7];
  cgra_input[0][9] = &vecOut[11];
  // Col 1
  cgra_input[1][0] = &vecA[0];
  cgra_input[1][1] = &vecA[4];
  cgra_input[1][2] = &vecA[8];
  cgra_input[1][3] = &vecA[12];
  cgra_input[1][4] = &vecB[0];
  cgra_input[1][5] = &vecB[4];
  cgra_input[1][6] = &vecB[8];
  cgra_input[1][7] = &vecB[12];
  cgra_input[1][8] = &vecOut[0];
  cgra_input[1][9] = &vecOut[4];
  cgra_input[1][10] = &vecOut[8];
  cgra_input[1][11] = &vecOut[12];
  // Col 2
  cgra_input[2][0] = &vecA[1];
  cgra_input[2][1] = &vecA[5];
  cgra_input[2][2] = &vecA[9];
  cgra_input[2][3] = &vecA[13];
  cgra_input[2][4] = &vecB[1];
  cgra_input[2][5] = &vecB[5];
  cgra_input[2][6] = &vecB[9];
  cgra_input[2][7] = &vecB[13];
  cgra_input[2][8] = &vecOut[1];
  cgra_input[2][9] = &vecOut[5];
  cgra_input[2][10] = &vecOut[9];
  cgra_input[2][11] = &vecOut[13];
  // Col 3
  cgra_input[3][0] = &vecA[2];
  cgra_input[3][1] = &vecA[6];
  cgra_input[3][2] = &vecA[10];
  cgra_input[3][3] = &vecA[14];
  cgra_input[3][4] = &vecB[2];
  cgra_input[3][5] = &vecB[6];
  cgra_input[3][6] = &vecB[10];
  cgra_input[3][7] = &vecB[14];
  cgra_input[3][8] = &vecOut[2];
  cgra_input[3][9] = &vecOut[6];
  cgra_input[3][10] = &vecOut[10];
  cgra_input[3][11] = &vecOut[14];

  // Set CGRA kernel L/S pointers
  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
  }
  printf("done\n"); 

  // CGRA Execution
  printf("Execute on cgra..."); 
  kcom_perfRecordStart( &(kperf.time.cgra) );
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, ADD_VECTORS_V1 );

  // Wait until CGRA is done
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  kcom_perfRecordStop( &(kperf.time.cgra) );
  printf("done\nCgra: %d\n", kperf.time.cgra.spent_cy); 

  // Execute on cpu
  printf("Execute on cpu..."); 
  kcom_perfRecordStart( &(kperf.time.cpu) );
  addVectorsCPU(vecOutCPU);
  kcom_perfRecordStop( &(kperf.time.cpu) );
  printf("done\nCPU: %d\n", kperf.time.cpu.spent_cy);

  // Check results
  printf("Check results..."); 
  int errors = 0;
  for(int i = 0; i < VECTOR_SIZE; i++){
    if(vecOut[i] != vecOutCPU[i]){
      errors++;
    }
  }
  if(errors > 0){
    printf("done\nErrors: %d\n", errors);
    int endIdx = (errors < 5) ? errors : 5;
    if (DEBUG){
      endIdx = VECTOR_SIZE;
    } 
    printf("CGRA vs CPU output:\n");
    for(int i = 0; i < endIdx; i++){
      printf("[%d] : %d != %d\n", i, vecOut[i], vecOutCPU[i]);
    }
  } else{
    printf("done\nOK\n");
  }

  return EXIT_SUCCESS;
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
void fillVector(int * vec, int n){
  for(int i = 0; i < n; i++){
    vec[i] = (i+1)%100;
  }
}

// Cpu vector addition
void addVectorsCPU(int32_t * out){
  for(int i = 0; i < VECTOR_SIZE; i++){
    out[i] = vecA[i] + vecB[i];
  }
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
