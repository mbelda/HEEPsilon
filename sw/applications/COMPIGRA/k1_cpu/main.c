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
  printf("Executing only non-mmul for kalman1 (dim %d)\n", NI);
  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  int i, j, k;
  int32_t sum;
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCYCLE, 0);
          // x = A x
  for (int i = 0; i < NI; i++) {
    float s = 0.0f;
    for (int k = 0; k < NI; k++) s += A[i*NI+k] * x[k];
    Ax_expected[i] = s;
  }
  for (int i = 0; i < NI; i++) x[i] = Ax_expected[i];

    /* ---- Update ---- */
  // y = z - H x
// AP = A * P
/*for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NI; j++) {
        int sum = 0;
        for (int k = 0; k < NI; k++) sum += A[i*NI+k] * P[k*NI+j];
        AP_expected[i*NI+j] = sum;
    }
}*/
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
