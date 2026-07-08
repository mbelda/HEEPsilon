/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : MVT                                                          **
** author   : Maria Jose Belda (mbelda@ucm.es)                             **
** filename : main.c                                                       **
** version  : 1                                                            **
** date     : 08/07/2026                                                   **
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
* @date   08/07/2026
* @brief  An application to run the MVT (Matrix Vector Product Transpose) benchmark on CPU.
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

// Dataset
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
// MVT CPU execution function
void mvt_cpu(int *A, int *x1, int *x2, int *y1, int *y2);

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

void main()
{

  init_csr_counters();

  printf("Running mvt on cpu for size N=%d...", N);
    
  reset_csr_counters();
  mvt_cpu(A, x1, x2, y1, y2);
  read_csr_counters();

  check_errors();


  return EXIT_SUCCESS;
}

void check_errors()
{
    int error = 0;

    // Validación de x1
    for(int i = 0; i < N; i++) {
        if(x1[i] != x1_expected[i]) {
            error++;
        }
    }

    // Validación de x2
    for(int i = 0; i < N; i++) {
        if(x2[i] != x2_expected[i]) {
            error++;
        }
    }

    if(error) {
        printf("FAIL with %d errors!!!\n\r", error);
    } else {
        printf("SUCCESS!\n\r");
    }
}


void mvt_cpu(int *A, int *x1, int *x2, int *y1, int *y2)
{
    // Primera parte: x1 = x1 + A * y1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x1[i] = x1[i] + A[i * N + j] * y1[j];
        }
    }

    // Segunda parte: x2 = x2 + A^T * y2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x2[i] = x2[i] + A[j * N + i] * y2[j];
        }
    }
}


/****************************************************************************/
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/