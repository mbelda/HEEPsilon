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
  printf("Executing 3mm cpu dimensions (%d,%d,%d,%d,%d)\n", NI, NJ, NK, NM, NL);
  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  int i, j, k;
  int32_t sum;
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCYCLE, 0);
  /* E := A*B */
  for (i = 0; i < NI; i++){
    for (j = 0; j < NJ; j++){
      outputE[i*NJ+j] = 0;
      for (k = 0; k < NK; k++){
        outputE[i*NJ +j] += inputA[i* NK + k] * inputB[k* NJ + j];
      }
    }
  }
  /* F := C*D */
  for (i = 0; i < NJ; i++){
    for (j = 0; j < NL; j++){
      outputF[i*NL +j] = 0;
      for (k = 0; k < NM; k++){
        outputF[i*NL + j] += inputC[i*NM +k] * inputD[k*NL + j];
      }
    }
  }
  /* G := E*F */
  for (i = 0; i < NI; i++){
    for (j = 0; j < NL; j++){
      outputG[i*NL+j] = 0;
      for (k = 0; k < NJ; ++k){
        outputG[i*NL +j] += outputE[i*NJ+k] * outputF[k*NL+j];
      }
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
