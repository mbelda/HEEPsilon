#include <stdio.h>
#include <stdlib.h>

#include "csr.h"
#include "hart.h"
#include "handler.h"
#include "core_v_mini_mcu.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "cgra_x_heep.h"
#include "cgra.h"
#include "cgra_bitstream.h"

#define DEBUG

// Use PRINTF instead of PRINTF to remove print by default
#ifdef DEBUG
  #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif

#define ROWS_A 16
#define COLS_A 16
#define ROWS_B 16
#define COLS_B 16
#define BLOCK_SIZE 4

static int32_t cgra_input[CGRA_N_COLS][CGRA_N_SLOTS][BLOCK_SIZE*2]     __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][CGRA_N_SLOTS][BLOCK_SIZE*BLOCK_SIZE]   __attribute__ ((aligned (4)));
int32_t cgra_res[CGRA_N_COLS][CGRA_N_ROWS][BLOCK_SIZE*BLOCK_SIZE] = {0};

static int16_t outCGRA[BLOCK_SIZE*BLOCK_SIZE];

int8_t cgra_intr_flag;

// Interrupt controller variables
void handler_irq_ext(uint32_t id) {
  if( id == CGRA_INTR) {
    cgra_intr_flag = 1;
  }
}

static int16_t matrixA[ROWS_A*COLS_A];
static int16_t matrixB[ROWS_B*COLS_B];


int main(void) {

  // Fill matrix inputs
  for(int i = 0; i < ROWS_A; i++){
    for(int j=0;j < COLS_A; j++){
      matrixA[i*COLS_A+j] = i*COLS_A+j;
    }
  }

  for(int i = 0; i < ROWS_B; i++){
    for(int j=0;j < COLS_B; j++){
      matrixB[i*COLS_B+j] = i*COLS_B+j;
    }
  }

  // CGRA initial config
  PRINTF("Init CGRA context memory...\n");
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  PRINTF("\rdone\n");

  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);

  // Enable interrupt on processor side
  // Enable global interrupt for machine-level interrupts
  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  // Set mie.MEIE bit to one to enable machine-level external interrupts
  const uint32_t mask = 1 << 11;//IRQ_EXT_ENABLE_OFFSET;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  cgra_t cgra;
  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  
  // Select request slot of CGRA (2 slots)
  uint32_t cgra_slot = cgra_get_slot(&cgra);
  cgra_perf_cnt_enable(&cgra, 1);
  int8_t column_idx;

  // The input data is going to change depending on each iteration of the for loop of the matrix multiplication
  int nBlockA = 0, nBlockB = 0;
  for (size_t rA = 0; rA < ROWS_A; rA+=BLOCK_SIZE){
    for (size_t cA = 0; cA < COLS_A; cA+=BLOCK_SIZE){
      // Multiply the block --> CGRA
      int cont = 0;
      // Matrix A
      for (size_t i=0; i < BLOCK_SIZE; i++, cont++){
        cgra_input[0][cgra_slot][cont] = matrixA[(rA+i)*COLS_A+cA];
        cgra_input[1][cgra_slot][cont] = matrixA[(rA+i)*COLS_A+cA+1];
        cgra_input[2][cgra_slot][cont] = matrixA[(rA+i)*COLS_A+cA+2];
        cgra_input[3][cgra_slot][cont] = matrixA[(rA+i)*COLS_A+cA+3];
      }
      // For each block on A on column i, I need to multiply it to the entire row i of B
      // rB = cA
      for(int cB = 0; cB < COLS_B; cB+=BLOCK_SIZE){

        // Matrix B (rB = cA, cB = rA)
        for (size_t i=0; i < BLOCK_SIZE; i++, cont++){
          cgra_input[0][cgra_slot][cont] = matrixB[(cA+i)*COLS_B+cB];
          cgra_input[1][cgra_slot][cont] = matrixB[(cA+i)*COLS_B+cB+1];
          cgra_input[2][cgra_slot][cont] = matrixB[(cA+i)*COLS_B+cB+2];
          cgra_input[3][cgra_slot][cont] = matrixB[(cA+i)*COLS_B+cB+3];
        }

        // It returns the values ordered by columns, so the matrix is transposed
        // Set CGRA kernel pointers
        column_idx = 0;
        cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[0][cgra_slot], column_idx);
        cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_res[0][cgra_slot], column_idx);
        // Set CGRA kernel pointers column 1
        column_idx = 1;
        cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[1][cgra_slot], column_idx);
        cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_res[1][cgra_slot], column_idx);
        // Set CGRA kernel pointers column 2
        column_idx = 2;
        cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[2][cgra_slot], column_idx);
        cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_res[2][cgra_slot], column_idx);
        // Set CGRA kernel pointers column 3
        column_idx = 3;
        cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[3][cgra_slot], column_idx);
        cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_res[3][cgra_slot], column_idx);

        // Launch CGRA kernel
        cgra_set_kernel(&cgra, cgra_slot, CGRA_FUNC_TEST);

        // Wait CGRA is done
        cgra_intr_flag=0;
        while(cgra_intr_flag==0) {
          wait_for_interrupt();
        }
      }

    }
  }


  // Last, performance

  printf("CGRA mmul finished\n");

  // Performance counter display
  printf("CGRA kernel executed: %d\n", cgra_perf_cnt_get_kernel(&cgra));
  column_idx = 0;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 1;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 2;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 3;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  cgra_perf_cnt_reset(&cgra);
  printf("CGRA kernel executed (after counter reset): %d\n", cgra_perf_cnt_get_kernel(&cgra));
  column_idx = 0;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 1;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 2;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
  column_idx = 3;
  PRINTF("CGRA column %d active cycles: %d\n", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  PRINTF("CGRA column %d stall cycles : %d\n", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));

  return EXIT_SUCCESS;
}
