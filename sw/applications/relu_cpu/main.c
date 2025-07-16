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
#include <stdint.h>

#include "cgra_x_heep.h"
#include "core_v_mini_mcu.h"

// For interrupt handling
#include "csr.h"
#include "csr_registers.h"
#include "handler.h"
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

// Fill input vectors with numbers
void fillVector(int32_t * vec, int n);


/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/


// Input and output matrixes
#define VECTOR_SIZE 128*128
static int32_t __attribute__((section(".xheep_data_interleaved"))) image[VECTOR_SIZE];

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

#define N_TAMANYOS 5
int tamanyos[N_TAMANYOS] = {8,16,32,64,128};
uint32_t tiempos_cpu[N_TAMANYOS] = {0};
int image_size;

int main()
{

  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  for (int i= 0; i < N_TAMANYOS; i++) {

    uint32_t sw_time;

    image_size = tamanyos[i];
    printf("Running relu for image size %d...", image_size);
    
    // Generate data
    fillVector(image, image_size);
    
    // Execute on cpu 
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for(int i = 0; i < image_size; i++){
      if(image[i] < 0){ image[i] = 0; }
    }
    CSR_READ(CSR_REG_MCYCLE, &sw_time);
    tiempos_cpu[i] = sw_time;
    
  }
 
  for (int i = 0; i < N_TAMANYOS; ++i) {
    int tam = tamanyos[i];
    printf("Tamaño: %dx%d\tCPU: %d\n", tam, tam, tiempos_cpu[i]);
  }


  return EXIT_SUCCESS;
}


// Fill matrix inputs
void fillVector(int32_t * vec, int n){
  for(int i = 0; i < n; i++){
    vec[i] = (i+1)%100;
  }
}


/****************************************************************************/
/**                                                                        **/
/*                                 EOF                                      */
/**                                                                        **/
/****************************************************************************/


