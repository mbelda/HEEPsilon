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
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "main.h"
#include "multiply_cgra.h"
#include "stftVec.h"


/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/

// Size of the input buffer for the CGRA
#define CGRA_COL_INPUT_SIZE 4

/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

// Matrix multiplication using the standard three loops
void mmulSoftware(int * matrixA, int rowsA, int colsA, int * matrixB, int rowsB, int colsB, int * out, int rowsC, int colsC);
// Fill input matrixes with numbers
void fillMatrixInputs();
// Show the performance metrics
void showPerformance( kcom_perf_t* kperf, int full);
void printMatrix(int * matrix, int rows, int cols);


/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

// Performance variable
static kcom_perf_t  kperf;

// Input and output matrixes
//int32_t matrixA[ROWS_A*COLS_A];
//int32_t matrixB[ROWS_B*COLS_B];
int32_t matrixC[ROWS_C*COLS_C];
int32_t outSW[ROWS_C*COLS_C];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

void main()
{
  // Init timer
  timerInit();

  // Initialize the CGRA
  kcom_perfRecordStart(&(kperf.time.load));
  initCGRA();
  kcom_perfRecordStop(&(kperf.time.load));

  // Enable and reset the CGRA performance counters
  countersInit();

  if(ROWS_C < CGRA_N_COLS || COLS_C < CGRA_N_ROWS){
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(stftVec, ROWS_A, COLS_A, stftVec, ROWS_B, COLS_B, matrixC, ROWS_C, COLS_C);
    kcom_perfRecordStop(&(kperf.time.sw));
  } else {
    multiply_cgra(&kperf, stftVec, ROWS_A, COLS_A, stftVec, ROWS_B, COLS_B, matrixC, ROWS_C, COLS_C);
    // Software 
    kcom_perfRecordStart(&(kperf.time.sw));
    mmulSoftware(stftVec, ROWS_A, COLS_A, stftVec, ROWS_B, COLS_B, outSW, ROWS_C, COLS_C);
    kcom_perfRecordStop(&(kperf.time.sw));

    checkErrors();
  }

  showPerformance(&kperf, 0);
  
  return EXIT_SUCCESS;
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

  if(errors>0){
    printMatrix(matrixC, ROWS_C, COLS_C);
  }
}

// Fill matrix inputs
void fillMatrixInputs(int * matrixA, int * matrixB){
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
void mmulSoftware(int * matrixA, int rowsA, int colsA, int * matrixB, int rowsB, int colsB, int * out, int rowsC, int colsC){
  for(int i = 0; i < rowsA; i++){
    for(int j=0;j < colsB; j++){
      for(int k=0; k < colsA; k++){
        out[i*colsC+j] += matrixA[i*colsA+k]*matrixB[k*colsB+j];
      }
    }
  }
}

// Display the performance values
void showPerformance( kcom_perf_t* kperf, int full){
  printf("\rA:%dx%d, B:%dx%d\n", ROWS_A, COLS_A, ROWS_B, COLS_B);
  printf("\rSw: %d\n", kperf->time.sw.spent_cy);
  if (full){
    printf("\rLoad: %d\n", kperf->time.load.spent_cy);
    printf("\rProgram cols: %d\n", kperf->time.reprogramCols.spent_cy);
    printf("\rInput: %d\n", kperf->time.input.spent_cy);
    printf("\rCgra: %d\n", kperf->time.cgra.spent_cy);
  }
  int32_t overhead = kperf->time.input.spent_cy + kperf->time.reprogramCols.spent_cy + kperf->time.load.spent_cy;
  //printf("\rOverhead: %d\n", overhead);
  printf("\rTotal cgra: %d\n", overhead + kperf->time.cgra.spent_cy); 
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





/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/
