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
  printf("Executing only non-mmul for PCA (dim %d)\n", NI);
  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  int i, j, k;
  int32_t sum;
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCYCLE, 0);
  /* center */
    for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
            Xc_golden[i*NJ + j] = inputX[i*NJ+j] - mu[j];
            XcT_golden[j*NI + i] = Xc_golden[i* NJ + j];
        }
    }


    /* matmul_AT_B with M = D: C = Xc^T * Xc */
    /*for (int i = 0; i < NJ; i++) {
        for (int j = 0; j < NJ; j++) {
            int sum = 0;
            for (int k = 0; k < NI; k++) {
                sum += XcT[i][k] * Xc[k][j];
            }
            C[i][j] = sum;
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
