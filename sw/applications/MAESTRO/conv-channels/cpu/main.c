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
int32_t conv_cpu(
    const int32_t *ptr_in_lider,
    const int32_t *ptr_w,
    int channels_per_group,
    int kernel_h,
    int kernel_w,
    int input_w,
    int tam_canal_input
);

/****************************************************************************/
/** **/
/* GLOBAL VARIABLES                              */
/** **/
/****************************************************************************/
int cpu_result = 0;

/****************************************************************************/
/** **/
/* LOCAL FUNCTIONS                               */
/** **/
/****************************************************************************/

int main()
{
  init_csr_counters();

  printf("Running 3-Loops Conv on CPU for C=%d, KH=%d, KW=%d, IH=%d, IW=%d...\n\r", 
          CHANNELS_PER_GROUP, KERNEL_H, KERNEL_W, INPUT_H, INPUT_W);

  reset_csr_counters();
  cpu_result = conv_cpu(
      input_lider,
      weights,
      CHANNELS_PER_GROUP,
      KERNEL_H,
      KERNEL_W,
      INPUT_W,
      TAM_CANAL_INPUT
  );
  read_csr_counters();

  check_errors();

  return EXIT_SUCCESS;
}

void check_errors() {
    printf("Check CPU output result:\n\r");
    if(cpu_result == EXPECTED_SUM) {
        printf("OK (Obtained: %d | Expected: %d)\n\r", cpu_result, EXPECTED_SUM);
        printf("SUCCESS!\n\r");
    } else {
        printf("FAIL -> Expected: %d | CPU Obtained: %d\n\r", EXPECTED_SUM, cpu_result);
    }
}

int32_t conv_cpu(
    const int32_t *ptr_in_lider,
    const int32_t *ptr_w,
    int channels_per_group,
    int kernel_h,
    int kernel_w,
    int input_w,
    int tam_canal_input
) {
    int32_t sum_val = 0;
    int idx_in = 0;
    int idx_w = 0;

    int salto_fila_filtro = input_w - kernel_w;
    int salto_siguiente_canal = tam_canal_input - (kernel_h * input_w);

    for (int c = 0; c < channels_per_group; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Multiplicación y acumulación
                sum_val += ptr_in_lider[idx_in] * ptr_w[idx_w];
                
                idx_in++;
                idx_w++;
            }
            // Fin de fila del filtro: salto
            idx_in += salto_fila_filtro;
        }
        // Fin de canal: salto al siguiente canal
        idx_in += salto_siguiente_canal;
    }

    return sum_val;
}

/****************************************************************************/
/** **/
/* EOF                                      */
/** **/
/****************************************************************************/