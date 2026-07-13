/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : MMUL CPU                                                     **
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
* @brief  An application to run a simple MMUL matrix multiplication on CPU.
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

// Dataset (Debe contener NI, NK, NJ, A, B y C_expected)
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
// MMUL Reference function
void mmul_cpu(int *A, int *B, int *C_out);

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/

// Output matrix
int32_t C[NI*NJ];


/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

int main()
{
  init_csr_counters();

  printf("Running MMUL for size %dx%dx%d...", NI, NK, NJ);
    
  reset_csr_counters();
  mmul_cpu(A, B, C);
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
 * @brief Computes simple matrix multiplication: C_out = A * B
 * Matrix dimensions: A (NI x NK), B (NK x NJ), C_out (NI x NJ)
 */
void mmul_cpu(int *A, int *B, int *C_out) {
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            int sum_val = 0;
            for (int k = 0; k < NK; k++) {
                sum_val += A[i * NK + k] * B[k * NJ + j];
            }
            C_out[i * NJ + j] = sum_val;
        }
    }
}

/****************************************************************************/
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/