#include <stdio.h>
#include <stdint.h>
#include "core_v_mini_mcu.h"
#include "hart.h"
#include "handler.h"
#include "csr.h"
#include "csr_registers.h"
#include "dataset.h"
#include "cgra_x_heep.h"


// N*M*K matrix dimensions

/*
#  ifdef MINI_DATASET
#   define NI 20
#   define NJ 25
#   define NK 30
#  endif

#  ifdef SMALL_DATASET
#   define NI 60
#   define NJ 70
#   define NK 80
#  endif
*/

int main()
{
    printf("Executing gemm cpu dimensions (%d,%d,%d)\n", NI, NK, NJ);
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    int i, j, k;
    int32_t sum;
    uint32_t sw_time;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for(i = 0; i < NI; i ++) {
        for(j = 0; j < NJ; j ++) {
            sum = 0;
            for(k = 0; k < NK; k++) {
                sum += inputX[i * NK + k] * inputY[k * NJ + j];
            }
            inputZ[i * NJ + j] = ALPHA * sum + BETA * inputZ[i * NJ + j];
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
