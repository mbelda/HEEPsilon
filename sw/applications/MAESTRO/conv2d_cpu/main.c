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
  printf("Executing conv2D dimensions (%d,%d)\n", IM_HEIGHT, IM_WIDTH);
  CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

  int x, y, kx, ky;
  int32_t sum;
  uint32_t sw_time;

  CSR_WRITE(CSR_REG_MCYCLE, 0);
  for (y = 1; y < IM_HEIGHT - 1; y++) {
      for (x = 1; x < IM_WIDTH - 1; x++) {
          sum = 0;
          for (ky = -1; ky <= 1; ky++) {
              for (kx = -1; kx <= 1; kx++) {
                  int in_x = x + kx;
                  int in_y = y + ky;

                  int input_val = image[in_y * IM_WIDTH + in_x];
                  int filter_val = filter[(ky + 1) * 3 + (kx + 1)];
                  sum += input_val * filter_val;
              }
          }
          output[y * IM_WIDTH + x] = sum;
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
