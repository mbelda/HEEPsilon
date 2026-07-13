/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : GEMM CPU                                                     **
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
* @date   04/03/2025
* @brief  An application to run a GEMM matrix multiplication on CPU.
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

#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

// Dataset (Debe contener NI, NK, NJ, ALPHA, BETA, A, B, C y C_expected)
#include "dataset.h"

// Counters
#include "performance.h"


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

// Check output
void check_errors();
// GEMM Reference function
void gemm_cpu(int *A, int *B, int *C);

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/


/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

int main()
{
  init_csr_counters();

  printf("Running GEMM for size %dx%dx%d...", NI, NK, NJ);
    
  reset_csr_counters();
  gemm_cpu(A, B, C);
  read_csr_counters();

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

/**
 * @brief Computes standard integer GEMM: C = alpha * A * B + beta * C
 * Matrix dimensions: A (NI x NK), B (NK x NJ), C (NI x NJ)
 */
void gemm_cpu(int *A, int *B, int *C) {
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            int sum_val = 0;
            for (int k = 0; k < NK; k++) {
                sum_val += A[i * NK + k] * B[k * NJ + j];
            }
            // Modifica la matriz C in-place tal y como lo hace el algoritmo estándar
            C[i * NJ + j] = ALPHA * sum_val + BETA * C[i * NJ + j];
        }
    }
}

/****************************************************************************/
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/