// Copyright 2022 EPFL and Politecnico di Torino.
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// File: main.c
// Author: Michele Caon
// Date: 22/06/2023
// Description: Main file for the matrix multiplication application

#include <stdlib.h>
#include <stdio.h>

//#include "heepatia.h"
#include "csr.h"
#include "fast_intr_ctrl.h"
#include "dma_sdk.h"
#include "vcd_util.h"
#include "timer_sdk.h"
#include "ext_irq.h"
// #include "carus.h"
// #include "carus_matmul.h"
#include "data.h"

#ifdef POWER_SIM
#pragma message "Power simulation ENABLED: disabling verification checks"
#endif

data_t R_cpu[R_ROWS * R_COLS] __attribute__((section(".xheep_data_interleaved"))); // Result computed by the CPU

// Software matrix multiplication
void cpuMatMul(data_t *A, data_t *B, data_t *R, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols);

void mmulSoftware(int32_t * out);



int main(void)
{

    dma_data_type_t dma_type = DMA_DATA_TYPE_WORD;
    data_t *row_ptr;
    unsigned int a_rows = A_ROWS;
    unsigned int a_cols = A_COLS;
    unsigned int b_cols = B_COLS;
    uint32_t cpu_cycles;




    timer_cycles_init();
    timer_start();

    // Compute result on the CPU
    cpuMatMul(A, B, R_cpu, a_rows, a_cols, b_cols);

    // Stop timer and disable VCD dump
    cpu_cycles = timer_stop();

    // Print the number of CPU cycles
    printf("CPU impl1: %u\n", cpu_cycles);

    timer_cycles_init();
    timer_start();

    mmulSoftware(R_cpu);

    cpu_cycles = timer_stop();
    printf("CPU impl2: %u\n", cpu_cycles);




    // Check the output data
    // ---------------------
    // Check NM-Carus output data
    int is_wrong = 0;
    for (unsigned int i = 0; i < R_ROWS; i++)
    {
        for (unsigned int j = 0; j < R_COLS; j++)
        {
            if (R_cpu[i * R_COLS + j] != R[i * R_COLS + j])
            {
                printf("CPU|gold R[%u,%u]: %x %x\n", i, j, R_cpu[i * R_COLS + j], R[i * R_COLS + j]);
                is_wrong = 1;
            }
            if (is_wrong) return 1;
        }
    }



    // Return success
    return 0;
}

void cpuMatMul(data_t *A, data_t *B, data_t *R, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols)
{
    for (unsigned int i = 0; i < a_rows; i++)
    {
        for (unsigned int j = 0; j < b_cols; j++)
        {
            R[i * b_cols + j] = 0;
            for (unsigned int k = 0; k < a_cols; k++)
            {
                R[i * b_cols + j] += A[i * a_cols + k] * B[k * b_cols + j];
            }
        }
    }
}

void mmulSoftware(int32_t * out){
    for(int i = 0; i < A_ROWS; i++){
      for(int j=0;j < B_COLS; j++){
        for(int k=0; k < A_COLS; k++){
          out[i*R_COLS+j] += A[i*A_COLS+k]*B[k*B_COLS+j];
        }
      }
    }
  }