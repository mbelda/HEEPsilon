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
#include "fxp.h"
#include "defines.h"
#include "fft_data.h"

#include "gpio.h"

#ifdef CPLX_FFT
  #if FFT_SIZE==512
    #include "fft_factors_512_32b_int.h"
  #endif
  #if FFT_SIZE==1024
    #include "fft_factors_1024_32b_int.h"
  #endif
  #if FFT_SIZE==2048
    #include "fft_factors_2048_32b_int.h"
  #endif
#endif // CPLX_FFT
#ifdef REAL_FFT
  #if FFT_SIZE==512
    #include "fft_factors_256_32b_int.h"
  #endif
  #if FFT_SIZE==1024
    #include "fft_factors_512_32b_int.h"
  #endif
  #if FFT_SIZE==2048
    #include "fft_factors_1024_32b_int.h"
  #endif
#endif // REAL_FFT

#define DEBUG

// Use PRINTF instead of PRINTF to remove print by default
#ifdef DEBUG
  #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif

/****************************************************************************/
/**                                                                        **/
/*                        VCD Generation Functions                          */
/**                                                                        **/
/****************************************************************************/

#define VCD_TRIGGER_GPIO 0

void dump_on(void);
void dump_off(void);

/* --------------------------------------------------------------------------
 *                     Functions declaration
 * --------------------------------------------------------------------------*/
uint16_t ReverseBits ( uint16_t index, uint16_t numBits );
uint16_t NumberOfBitsNeeded ( uint16_t powerOfTwo );

/* --------------------------------------------------------------------------
 *                     Global variables
 * --------------------------------------------------------------------------*/

// FFT radix-2 variables
fxp RealOut_fft0_fxp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };
fxp ImagOut_fft0_fxp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };

#ifdef CGRA_100_PERCENT
  fxp RealOut_fft1_fxp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };
  fxp ImagOut_fft1_fxp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };
#endif

fxp RealOut_fxp_exp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };
fxp ImagOut_fxp_exp[FFT_SIZE] __attribute__ ((aligned (4))) = { 0 };

#ifdef REAL_FFT
  fxp re_tmp[FFT_SIZE/2+1] __attribute__ ((aligned (4))) = { 0 };
  fxp im_tmp[FFT_SIZE/2+1] __attribute__ ((aligned (4))) = { 0 };
#endif // REAL_FFT

// one dim per core x n input values (data ptrs, constants, ...)
int32_t cgra_input[CGRA_N_COLS][CGRA_N_SLOTS][10] __attribute__ ((aligned (4))) = { 0 };
int8_t cgra_intr_flag;
// Nothing should be write here by the FFT kernel
// int32_t cgra_output[CGRA_N_COLS][CGRA_N_ROWS][10] __attribute__ ((aligned (4))) = { 0 };

/*----------------------------------------------------------------------------
                        INTERRUPTS
-----------------------------------------------------------------------------*/

// Interrupt controller variables
void handler_irq_cgra(uint32_t id) {
    cgra_intr_flag = 1;
}

cgra_t cgra;

/* --------------------------------------------------------------------------
 *                     main
 * --------------------------------------------------------------------------*/
int main(void) {

  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  PRINTF("\rdone\n");

  // Init the PLIC
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, handler_irq_cgra);

  // Enable interrupt on processor side
  // Enable global interrupt for machine-level interrupts
  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  // Set mie.MEIE bit to one to enable machine-level external interrupts
  const uint32_t mask = 1 << 11;//IRQ_EXT_ENABLE_OFFSET;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  
  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);


  //////////////////////////////////////////////////////////
  //
  // COMPLEX FFT radix-2 (Butterfy) implementation
  //
  //////////////////////////////////////////////////////////


  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  uint16_t numBits = NumberOfBitsNeeded ( FFT_SIZE );
  int8_t column_idx;

  // STEP 1: bit reverse
  // Select request slot of CGRA (2 slots)
  uint32_t cgra_slot = cgra_get_slot(&cgra);
  column_idx = 0;
  cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[column_idx][cgra_slot], column_idx);




  
  // input data ptr column 0
  cgra_input[column_idx][cgra_slot][0] = (int32_t)&input_signal[1]; // imaginary part is given second
  cgra_input[column_idx][cgra_slot][1] = (int32_t)&input_signal[0]; // imaginary part is given first
  cgra_input[column_idx][cgra_slot][2] = (int32_t)FFT_SIZE/2; // idx end
  cgra_input[column_idx][cgra_slot][3] = (int32_t)numBits;
  cgra_input[column_idx][cgra_slot][4] = (int32_t)&ImagOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][5] = (int32_t)&RealOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][6] = 0; // idx start
  
  // Print the configured CGRA input values
  printf("--- CGRA Input Configuration (Column: %d, Slot: %d) ---\n", column_idx, cgra_slot);
  printf("Slot [0] (Imaginary Part 2nd - Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][0]);
  printf("Slot [1] (Imaginary Part 1st - Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][1]);
  printf("Slot [2] (IDX End):                  %d\n", cgra_input[column_idx][cgra_slot][2]);
  printf("Slot [3] (Num Bits):                 %d\n", cgra_input[column_idx][cgra_slot][3]);
  printf("Slot [4] (ImagOut Ptr):              %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][4]);
  printf("Slot [5] (RealOut Ptr):              %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][5]);
  printf("Slot [6] (IDX Start):                %d\n", cgra_input[column_idx][cgra_slot][6]);
  printf("Run kernel: %d\n", CGRA_FTT_BITREV_ID);
  printf("-------------------------------------------------------\n");

  // Launch CGRA kernel
  //dump_on();
  cgra_set_kernel(&cgra, cgra_slot, CGRA_FTT_BITREV_ID);
  // Wait CGRA is done
  cgra_intr_flag=0;
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  //dump_off();

  printCGRA();
  cgra_perf_cnt_reset( &cgra );


  cgra_slot = cgra_get_slot(&cgra);
  column_idx = 0;
  cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[column_idx][cgra_slot], column_idx);

  // input data ptr column 0
  cgra_input[column_idx][cgra_slot][0] = (int32_t)&input_signal[FFT_SIZE/2+1]; // imaginary part is given second
  cgra_input[column_idx][cgra_slot][1] = (int32_t)&input_signal[FFT_SIZE/2]; // imaginary part is given first
  cgra_input[column_idx][cgra_slot][2] = (int32_t)FFT_SIZE; // idx end
  cgra_input[column_idx][cgra_slot][3] = (int32_t)numBits;
  cgra_input[column_idx][cgra_slot][4] = (int32_t)&ImagOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][5] = (int32_t)&RealOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][6] = FFT_SIZE/2; // idx start

// Print the configured CGRA input values (Second Half / High Indices)
  printf("--- CGRA Input Configuration (Column: %d, Slot: %d) ---\n", column_idx, cgra_slot);
  printf("Slot [0] (Imaginary Part 2nd - Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][0]);
  printf("Slot [1] (Imaginary Part 1st - Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][1]);
  printf("Slot [2] (IDX End):                  %d\n", cgra_input[column_idx][cgra_slot][2]);
  printf("Slot [3] (Num Bits):                 %d\n", cgra_input[column_idx][cgra_slot][3]);
  printf("Slot [4] (ImagOut Ptr):              %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][4]);
  printf("Slot [5] (RealOut Ptr):              %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][5]);
  printf("Slot [6] (IDX Start):                %d\n", cgra_input[column_idx][cgra_slot][6]);
  printf("Run kernel: %d\n", CGRA_FTT_BITREV_ID);
  printf("-------------------------------------------------------\n");

  // Launch CGRA kernel
  cgra_set_kernel(&cgra, cgra_slot, CGRA_FTT_BITREV_ID);
  

  // Wait CGRA is done
  cgra_intr_flag=0;
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  

  printCGRA();
  cgra_perf_cnt_reset( &cgra );

  // Step 2: complex-valued FFT computation
  PRINTF("Run a complex FFT of %d points on CGRA...\n", FFT_SIZE);

  cgra_slot = cgra_get_slot(&cgra);
  column_idx = 0;
  cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[column_idx][cgra_slot], column_idx);
  // cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_output[column_idx][cgra_slot], column_idx);

  // input data ptr column 0
  cgra_input[column_idx][cgra_slot][0] = (int32_t)&RealOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][1] = (int32_t)&f_real[0];
  cgra_input[column_idx][cgra_slot][2] = (int32_t)FFT_SIZE;

  // Print the configured CGRA input values
  /*printf("--- CGRA Input Configuration (Column: %d, Slot: %d) ---\n", column_idx, cgra_slot);
  printf("Slot [0] (RealOut_fft0 Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][0]);
  printf("Slot [1] (f_real Ptr):       %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][1]);
  printf("Slot [2] (FFT SIZE):         %d\n", cgra_input[column_idx][cgra_slot][2]);
*/
  column_idx = 1;
  cgra_set_read_ptr(&cgra, cgra_slot, (uint32_t) cgra_input[column_idx][cgra_slot], column_idx);
  // cgra_set_write_ptr(&cgra, cgra_slot, (uint32_t) cgra_output[column_idx][cgra_slot], column_idx);

  // input data ptr column 1
  cgra_input[column_idx][cgra_slot][0] = (int32_t)&f_imag[0];
  cgra_input[column_idx][cgra_slot][1] = (int32_t)&ImagOut_fft0_fxp[0];
  cgra_input[column_idx][cgra_slot][2] = (int32_t)numBits;

  // Print the configured CGRA input values
  /*printf("--- CGRA Input Configuration (Column: %d, Slot: %d) ---\n", column_idx, cgra_slot);
  printf("Slot [0] (f_imag Ptr):       %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][0]);
  printf("Slot [1] (ImagOut_fft0 Ptr): %p\n", (void*)(intptr_t)cgra_input[column_idx][cgra_slot][1]);
  printf("Slot [2] (Num Bits):         %d\n", cgra_input[column_idx][cgra_slot][2]);
  */

  // Launch CGRA kernel
    printf("Run kernel: %d\n", CGRA_FTT_CPLX_ID);
    printf("-------------------------------------------------------\n");
    dump_on();
    cgra_set_kernel(&cgra, cgra_slot, CGRA_FTT_CPLX_ID);

  // Wait CGRA is done
  cgra_intr_flag=0;
  while(cgra_intr_flag==0) {
    wait_for_interrupt();
  }
  dump_off();

  printCGRA();
  cgra_perf_cnt_reset( &cgra );


  int32_t errors=0;
  for (int i=0; i<FFT_SIZE; i++) {
    if(RealOut_fft0_fxp[i] != exp_output_real[i] ||
        ImagOut_fft0_fxp[i] != exp_output_imag[i]) {
          //printf("Real[%d] (out/expected) %08x != %08x)\n", i, RealOut_fft0_fxp[i], exp_output_real[i]);
          //printf("Imag[%d] (out/expected) %08x != %08x)\n", i, ImagOut_fft0_fxp[i], exp_output_imag[i]);
        errors++;
      }
  }

  printf("CGRA FFT %d errors\n", errors);

  return EXIT_SUCCESS;
}

void printCGRA(){
  int column_idx = 0;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));

  column_idx = 1;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));

  column_idx = 2;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));

  column_idx = 3;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));

}

uint16_t ReverseBits (uint16_t index, uint16_t numBits)
{
  uint16_t i, rev;

  for (i=rev=0; i<numBits; i++) {
    rev = (rev << 1) | (index & 1);
    index >>= 1;
  }

  return rev;
}

uint16_t NumberOfBitsNeeded (uint16_t powerOfTwo)
{
  uint16_t i;

  if (powerOfTwo < 2) {
   return 0; // should not happen
  }

  for (i=0;; i++) {
    if (powerOfTwo & (1 << i))
      return i;
  }
}

void dump_on(void)
{
  gpio_result_t gpio_res;
    gpio_cfg_t pin_cfg = {
        .pin = VCD_TRIGGER_GPIO,
        .mode = GpioModeOutPushPull
    };

    gpio_res = gpio_config (pin_cfg);

    if (gpio_res != GpioOk)
        printf("Gpio initialization failed!\n");


    gpio_write(VCD_TRIGGER_GPIO, true);

}

void dump_off(void)
{
    gpio_write(VCD_TRIGGER_GPIO, false);
}
