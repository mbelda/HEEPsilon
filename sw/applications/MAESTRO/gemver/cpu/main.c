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

  printf("Running gemver on cpu for size %d...", N);
    
  reset_csr_counters();
  gemver_cpu(A, u1, v1, u2, v2, x, y, z, w);
  read_csr_counters();

  check_errors();


  return EXIT_SUCCESS;
}

void check_errors()
{
    int error = 0;

    for(int i = 0; i < N; i++) {
        if(w[i] != w_expected[i]) {
            error++;
        }
    }

    if(error) {
        printf("FAIL with %d errors!!!\n\r", error);
    } else {
        printf("SUCCESS!\n\r");
    }
}


void gemver_cpu(int *A,
                int *u1, int *v1,
                int *u2, int *v2,
                int *x, int *y,
                int *z, int *w)
{
    /* Parte 1: A = A + u1*v1^T + u2*v2^T */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] += u1[i] * v1[j] +
                            u2[i] * v2[j];
        }
    }

    /* Parte 2: x = x + beta*A^T*y + z */
    for (int i = 0; i < N; i++) {

        int aux = 0;

        for (int j = 0; j < N; j++) {
            aux += BETA * A[j * N + i] * y[j];
        }

        x[i] += aux + z[i];
    }

    /* Parte 3: w = w + alpha*A*x */
    for (int i = 0; i < N; i++) {

        int aux = 0;

        for (int j = 0; j < N; j++) {
            aux += ALPHA * A[i * N + j] * x[j];
        }

        w[i] += aux;
    }
}


/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
