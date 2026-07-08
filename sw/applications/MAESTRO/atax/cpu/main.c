/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : ATAX                                                         **
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
* @brief  An application to run the full ATAX benchmark on CPU.
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
// ATAX CPU execution function
void atax_cpu(int *A, int *x, int *tmp, int *y);

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/
int tmp[M];
int y[N];

/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

void main()
{

  init_csr_counters();

  printf("Running atax on cpu for size M=%d, N=%d...", M, N);
    
  reset_csr_counters();
  atax_cpu(A, x, tmp, y);
  read_csr_counters();

  check_errors();


  return EXIT_SUCCESS;
}

void check_errors()
{
    int error = 0;

    // Validación sobre el vector de salida final (y) de tamaño N
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


void atax_cpu(int *A, int *x, int *tmp, int *y)
{
    // Inicialización explícita de los vectores de salida/acumulación
    for (int i = 0; i < M; i++) {
        tmp[i] = 0;
    }
    for (int j = 0; j < N; j++) {
        y[j] = 0;
    }

    /* Parte 1: tmp = A * x */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            tmp[i] += A[i * N + j] * x[j];
        }
    }

    /* Parte 2: y = A^T * tmp + y */
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            y[j] += A[i * N + j] * tmp[i];
        }
    }
}


/****************************************************************************/
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/