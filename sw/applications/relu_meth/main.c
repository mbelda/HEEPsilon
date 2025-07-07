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

// relu on cpu
void reluCPU(int32_t * out, int32_t siz);
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



// CGRA input buffers
#define CGRA_COL_INPUT_SIZE 2
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE]    __attribute__ ((aligned (4)));

// Input and output matrixes
#define VECTOR_SIZE 512*512
static int32_t __attribute__((section(".xheep_data_interleaved"))) image[VECTOR_SIZE];
static int32_t __attribute__((section(".xheep_data_interleaved"))) vecOutCPU[VECTOR_SIZE];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

#define DEBUG 0
#define N_TAMANYOS 8
int tamanyos[N_TAMANYOS] = {8,16,32,64,128,256,512};
int tiempos_cpu[N_TAMANYOS] = {0,0,0,0,0,0,0,0};
int tiempos_cgra[N_TAMANYOS] = {0,0,0,0,0,0,0,0};
int image_size;

void main()
{

  // Initialize the CGRA
  initCGRA(); // This only needs to be done once


  for (int i= 0; i < N_TAMANYOS; i++) {
    // Init timer
    timerInit();

    // Enable and reset the CGRA performance counters
    cgra_perf_cnt_enable(&cgra, 1);
    cgra_perf_cnt_reset( &cgra );

    // Reset performnace counters
    kcom_perf_t  kperf;

    image_size = tamanyos[i];
    printf("Running relu for image size %d...", image_size);
    
    // Generate data
    fillVector(image, image_size);

    // Prepare the input vector for the CGRA
    // ----------------------
    // Config values
    // ----------------------
    // &Im           -         &Im         -
    // nIt
    
    int nIt = image_size/16;
    // Col 0
    cgra_input[0][0] = &image[0];
    cgra_input[0][1] = nIt;
    // Col 1
    // Col 2
    cgra_input[2][0] = &image[0];
    // Col 3

    // Set CGRA kernel L/S pointers
    for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
      cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx], col_idx );
    }

    // Execute on cpu 
    kcom_perfRecordStart( &(kperf.time.cpu) );
    reluCPU(vecOutCPU, image_size);
    kcom_perfRecordStop( &(kperf.time.cpu) );
    tiempos_cpu[i] = kperf.time.cpu.spent_cy;

    // CGRA Execution
    kcom_perfRecordStart( &(kperf.time.cgra) );
    cgra_intr_flag = 0;
    cgra_set_kernel( &cgra, cgra_slot, RELU );

     // Wait until CGRA is done
    while(cgra_intr_flag==0) {
      wait_for_interrupt();
    }
    kcom_perfRecordStop( &(kperf.time.cgra) );
    tiempos_cgra[i] = kperf.time.cgra.spent_cy;

    // Check results
    int errors = 0;
    for(int i = 0; i < image_size; i++){
      if(image[i] != vecOutCPU[i]){
        errors++;
      }
    }
    if(errors > 0){
     printf("ERR\n");
    } else{
      printf("OK\n");
    }

  }
 
  for (int i = 0; i < N_TAMANYOS; ++i) {
    int tam = tamanyos[i] * tamanyos[i];
    printf("Tamaño: %d\tCPU: %d\tCGRA: %d\n", tam, tiempos_cpu[i], tiempos_cgra[i]);
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
void reluCPU(int32_t * out, int32_t siz){
  for(int i = 0; i < siz; i++){
    if(image[i] < 0){
      out[i] = 0;
    } else {
      out[i] = image[i];
    }
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
