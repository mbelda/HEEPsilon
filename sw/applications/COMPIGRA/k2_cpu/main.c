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
  printf("Executing only non-mmul for kalman2 (dim %d)\n", NI);
  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  int i, j, k;
  int32_t sum;
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCYCLE, 0);
  // AT = transpose(A)
for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NI; j++) {
        AT_expected[j*NI+i] = A[i*NI+j];
    }
}

// P = APA + Q
for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NI; j++) {
        P_expected[i*NI+j] = APA_expected[i*NI+j] + Q[i*NI+j];
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
