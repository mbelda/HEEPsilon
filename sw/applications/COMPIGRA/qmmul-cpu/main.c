#include <stdio.h>
#include <stdint.h>
#include "core_v_mini_mcu.h"
#include "hart.h"
#include "handler.h"
#include "csr.h"
#include "csr_registers.h"
#include "dataset.h"
#include "cgra_x_heep.h"


// Temporary flat arrays to hold the transformed matrices
    int A_transformed[NI * NK];
    int B_transformed[NK * NJ];

int main()
{
    printf("Executing relu cpu\n");
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    int i, j;
    uint32_t sw_time;

    CSR_WRITE(CSR_REG_MCYCLE, 0);


    // 1. Pre-calculations: Apply bias to A
    for (int i = 0; i < NI; i++) {
        for (int k = 0; k < NK; k++) {
            A_transformed[i * NK + k] = matrix_A[i * NK + k] + bias[i];
        }
    }

    // 2. Pre-calculations: Apply scale to B
    for (int k = 0; k < NK; k++) {
        for (int j = 0; j < NJ; j++) {
            B_transformed[k * NJ + j] = matrix_B[k * NJ + j] * scale[j];
        }
    }
    CSR_READ(CSR_REG_MCYCLE, &sw_time);
    
    printf("SW cycles: %lu\n", sw_time);
    // Check results
    int error = 0;
    // for(int i = 0; i < DATA_SIZE; i++)
    //     if(output[i] != expected_result[i]) error++;

    // if(error) PRINTF("FAIL with %d errrors!!!\r\n", error);
    // else PRINTF("SUCCESS!\r\n");
    
    return error;
}