#include <stdio.h>
#include <stdint.h>
#include "core_v_mini_mcu.h"
#include "hart.h"
#include "handler.h"
#include "csr.h"
#include "csr_registers.h"
#include "dataset.h"
#include "cgra_x_heep.h"


int main()
{
    printf("Executing gemm cpu dimensions (%d,%d,%d)\n", DIM_M, DIM_N, DIM_K);
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    int i, j, k;
    int32_t sum;
    uint32_t sw_time;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for(i = 0; i < DIM_N; i ++) {
        for(j = 0; j < DIM_M; j ++) {
            sum = 0;
            for(k = 0; k < DIM_K; k++) {
                sum += inputX[i * DIM_K + k] * inputY[k * DIM_M + j];
            }
            inputZ[i * DIM_M + j] = ALPHA * sum + BETA * inputZ[i * DIM_M + j];
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
