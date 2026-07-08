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

#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

// Dataset
#include "dataset.h"

// Counters
#include "performance.h"


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

// Check output
void check_errors();

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/


/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{

  init_csr_counters();

  printf("Running gesummv for size %d...", N);
    
  reset_csr_counters();
  gesummv_cpu(A, B, x, y);
  read_csr_counters();

  check_errors();


  return EXIT_SUCCESS;
}

void check_errors() {
    int error = 0;
    for(int i = 0; i < N; i++) {
        if(y[i] != y_expected[i]) {
          error++;
        }
    }

    if(error) {
        printf("FAIL with %d errors!!!\n\r", error);
    } else {
        printf("SUCCESS!\n\r");
    }
}


void gesummv_cpu(int *A, int *B, int *x, int *y_out) {
    for (int i = 0; i < N; i++) {
        int s_tmp = 0;
        int s_y = 0;
        for (int j = 0; j < N; j++) {
            s_tmp += A[i * N + j] * x[j];
            s_y += B[i * N + j] * x[j];
        }
        y_out[i] = ALPHA * s_tmp + BETA * s_y;
    }
}


/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
