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
    printf("Executing relu cpu\n");
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    int i, j;
    uint32_t sw_time;

    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for(i = 0; i < DATA_SIZE; i ++) {
        if(input[i] < 0) input[i] = 0;
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
