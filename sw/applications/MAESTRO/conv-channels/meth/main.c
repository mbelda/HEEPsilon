/*
                              *******************
******************************* C SOURCE FILE *******************************
** ******************* **
** **
** project  : 3-Loops Convolution Benchmark with Edge Accumulation          **
** author   : Maria Jose Belda (mbelda@ucm.es)                             **
** filename : main.c                                                       **
** version  : 3 (Hybrid CPU + CGRA Verification)                           **
** date     : 2026                                                         **
** **
*****************************************************************************
** **
** Copyright (c) UCM                                                       **
** All rights reserved.                                                    **
** **
*****************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "cgra_bitstream.h"
#include "cgra_x_heep.h"

// For interrupt handling
#include "csr.h"
#include "handler.h"
#include "rv_plic.h"
#include "rv_plic_regs.h"
#include "hart.h"

// Dataset generado dinámicamente por el script de Python
#include "dataset.h"

// Metricas cpu
# include "performance.h"

/****************************************************************************/
/* PROTOTYPES OF LOCAL FUNCTIONS                                            */
/****************************************************************************/
void handler_irq_cgra(uint32_t id);
void printMetrics();
void initCGRA();
void check_errors(int32_t total_obtained);
int32_t calcular_bordes_cpu();

/****************************************************************************/
/* GLOBAL VARIABLES                                                         */
/****************************************************************************/
volatile bool               cgra_intr_flag;
static cgra_t               cgra;
static uint8_t              cgra_slot;
int cgra_sum_out;

#define CGRA_COL_INPUT_SIZE 8
#define CGRA_COL_OUTPUT_SIZE 1
static int32_t cgra_input[CGRA_N_COLS][CGRA_COL_INPUT_SIZE] __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_N_COLS][CGRA_COL_OUTPUT_SIZE] __attribute__ ((aligned (4)));

/****************************************************************************/
/* MAIN EXECUTION                                                           */
/****************************************************************************/
int main()
{
  init_csr_counters();
  // Inicializar el hardware del CGRA
  initCGRA();

  // Habilitar y resetear contadores de rendimiento
  cgra_perf_cnt_enable(&cgra, 1);
  cgra_perf_cnt_reset( &cgra );

  printf("Running Hybrid 3-Loops Conv for C=%d, KH=%d, KW=%d, IH=%d, IW=%d...\n\r", 
          CHANNELS_PER_GROUP, KERNEL_H, KERNEL_W, INPUT_H, INPUT_W);

  // --- PASO 1: CÁLCULO DE LOS BORDES EN LA CPU ---
  // Ejecutamos el remanente de los bordes concurrentemente o antes de lanzar el CGRA
  printf("Computing edge elements on CPU...\n\r");
  reset_csr_counters();
  int32_t cpu_edge_sum = calcular_bordes_cpu();
  read_csr_counters();

  printf("Configure input values for CGRA...\n");
  reset_csr_counters();
  // --- PASO 2: CONFIGURACIÓN Y LANZAMIENTO DEL CGRA (ZONA CENTRAL) ---
  uint32_t first_addr_in = (uint32_t)&input_lider[0];
  uint32_t first_addr_w  = (uint32_t)&weights[0];

  uint32_t size_ch_im = 4 * TAM_CANAL_INPUT;
  uint32_t size_ch_f  = 4 * KERNEL_H * KERNEL_W;

  // Direcciones mapeadas
  uint32_t addr_Im_0  = first_addr_in + (0 * size_ch_im);
  uint32_t addr_Im_6  = first_addr_in + (6 * size_ch_im);
  uint32_t addr_Im_9  = first_addr_in + (9 * size_ch_im);
  uint32_t addr_Im_15 = first_addr_in + (15 * size_ch_im);

  uint32_t addr_F_2   = first_addr_w + (2 * size_ch_f);
  uint32_t addr_F_4   = first_addr_w + (4 * size_ch_f);
  uint32_t addr_F_11  = first_addr_w + (11 * size_ch_f);
  uint32_t addr_F_13  = first_addr_w + (13 * size_ch_f);

  int loopCIt  = (CHANNELS_PER_GROUP / 16) - 1;
  int loopKHIt = KERNEL_H - 1;
  int loopKWIt = KERNEL_W - 1;

  // Carga de configuración en las columnas del CGRA
  cgra_input[0][0] = addr_Im_0;
  cgra_input[0][1] = addr_F_4;
  cgra_input[0][2] = size_ch_f;
  cgra_input[0][3] = size_ch_im;
  cgra_input[0][4] = INPUT_H;
  cgra_input[0][5] = KERNEL_H;
  cgra_input[0][6] = KERNEL_W;
  cgra_input[0][7] = loopCIt;

  cgra_input[1][0] = size_ch_im;
  cgra_input[1][1] = size_ch_f;
  cgra_input[1][2] = addr_Im_9;
  cgra_input[1][3] = addr_F_13;
  cgra_input[1][4] = INPUT_W;
  cgra_input[1][5] = KERNEL_W;
  cgra_input[1][6] = INPUT_H;
  cgra_input[1][7] = KERNEL_H;

  cgra_input[2][0] = addr_F_2;
  cgra_input[2][1] = addr_Im_6;
  cgra_input[2][2] = size_ch_im;
  cgra_input[2][3] = size_ch_f;
  cgra_input[2][4] = KERNEL_H;
  cgra_input[2][5] = loopKHIt;
  cgra_input[2][6] = INPUT_W;
  cgra_input[2][7] = KERNEL_W;

  cgra_input[3][0] = size_ch_f;
  cgra_input[3][1] = size_ch_im;
  cgra_input[3][2] = addr_F_11;
  cgra_input[3][3] = addr_Im_15;
  cgra_input[3][4] = KERNEL_W;
  cgra_input[3][5] = loopKWIt;
  cgra_input[3][6] = KERNEL_H;
  cgra_input[3][7] = INPUT_W;

  for(int col_idx = 0 ; col_idx < CGRA_N_COLS ; col_idx++){
    cgra_set_read_ptr ( &cgra, cgra_slot, (uint32_t) cgra_input[col_idx],  col_idx );
    cgra_set_write_ptr( &cgra, cgra_slot, (uint32_t) cgra_output[col_idx], col_idx );
  }
  read_csr_counters();

  // Ejecución del CGRA
  cgra_intr_flag = 0;
  cgra_set_kernel( &cgra, cgra_slot, CONV_KERNEL_ID);

  // Esperar la interrupción de hardware del CGRA
  while(cgra_intr_flag == 0) {
    wait_for_interrupt();
  }

  cgra_sum_out = cgra_output[1][0];
  /*
  // --- PASO 3: FUSIÓN Y COMPROBACIÓN ---
  int32_t total_obtained = cgra_sum_out + cpu_edge_sum;
  
  //printMetrics();
  printf("CPU Edges Accumulation: %d\n\r", cpu_edge_sum);
  check_errors(total_obtained);
  */


  check_errors(cgra_sum_out);

  return EXIT_SUCCESS;
}

/****************************************************************************/
/* AUXILIARY FUNCTIONS                                                      */
/****************************************************************************/

/**
 * @brief Recorre las mismas dimensiones del patch simulando la lógica de bordes 
 *        utilizando las macros del dataset autogenerado.
 */
int32_t calcular_bordes_cpu() {
    int32_t sum_bordes = 0;

    // Asumimos un stride por defecto de 1 y pad dinámico para testeo unitario
    // (Ajusta estas variables si tu generador de Python altera strides/pads)
    int stride_h = 1; 
    int stride_w = 1;
    int pad_h = KERNEL_H / 2; // Estimación común de padding
    int pad_w = KERNEL_W / 2;

    int out_h = (INPUT_H + 2 * pad_h - KERNEL_H) / stride_h + 1;
    int out_w = (INPUT_W + 2 * pad_w - KERNEL_W) / stride_w + 1;

    // Límites de la zona segura del CGRA
    int oh_seguro_inicio = (pad_h + stride_h - 1) / stride_h;
    int oh_seguro_fin = (INPUT_H + pad_h - KERNEL_H) / stride_h + 1;
    if (oh_seguro_fin < oh_seguro_inicio) oh_seguro_fin = oh_seguro_inicio;

    int ow_seguro_inicio = (pad_w + stride_w - 1) / stride_w;
    int ow_seguro_fin = (INPUT_W + pad_w - KERNEL_W) / stride_w + 1;
    if (ow_seguro_fin < ow_seguro_inicio) ow_seguro_fin = ow_seguro_inicio;

    // Procesamos la matriz de salida completa buscando ÚNICAMENTE los elementos fuera de la zona segura
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            
            // Si cae DENTRO de la zona segura, lo ignoramos (esto lo hace el CGRA)
            if (oh >= oh_seguro_inicio && oh < oh_seguro_fin &&
                ow >= ow_seguro_inicio && ow < ow_seguro_fin) {
                continue; 
            }

            // Si es un BORDE, calculamos su acumulación en la CPU
            for (int icg = 0; icg < CHANNELS_PER_GROUP; ++icg) {
                for (int kh = 0; kh < KERNEL_H; ++kh) {
                    for (int kw = 0; kw < KERNEL_W; ++kw) {
                        
                        int ih = oh * stride_h + kh - pad_h;
                        int iw = ow * stride_w + kw - pad_w;

                        // Verificación estricta de zero-padding
                        if (ih < 0 || ih >= INPUT_H || iw < 0 || iw >= INPUT_W) {
                            continue; 
                        }

                        // Indexación plana compatible con los vectores lineales de dataset.h
                        int input_idx = icg * TAM_CANAL_INPUT + ih * INPUT_W + iw;
                        int weight_idx = (icg * KERNEL_H + kh) * KERNEL_W + kw;

                        sum_bordes += input_lider[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }
    return sum_bordes;
}

/**
 * @brief Comprueba el resultado del CGRA de forma aislada contra el valor seguro
 *        esperado generado por el script de Python.
 */
void check_errors(int32_t obtenido_cgra) {
    printf("\n\r=== VERIFICACIÓN AISLADA DEL HARDWARE (CGRA) ===\n\r");
    
    if(obtenido_cgra == EXPECTED_CGRA_SUM) {
        printf("OK [CGRA Core Pasó el Test]\n\r");
        printf("  -> Obtenido en Hardware: %d\n\r", obtenido_cgra);
        printf("  -> Esperado en CGRA (Zona Segura): %d\n\r", EXPECTED_CGRA_SUM);
        printf("SUCCESS!\n\r");
    } else {
        printf("FAIL [Discrepancia en el Hardware]\n\r");
        printf("  -> ERROR: Se esperaba %d pero el CGRA devolvió %d\n\r", EXPECTED_CGRA_SUM, obtenido_cgra);
        printf("  -> Nota: Comprueba que el mapeo de punteros coincida con los saltos de memoria.\n\r");
    }
}

/*
void check_errors(int32_t total_obtained) {
    printf("Check Unified (CGRA + CPU Edges) output result:\n\r");
    if(total_obtained == EXPECTED_SUM) {
        printf("OK (Obtained: %d | Expected: %d)\n\r", total_obtained, EXPECTED_SUM);
        printf("SUCCESS!\n\r");
    } else {
        printf("FAIL -> Expected: %d | Total Obtained: %d (CGRA: %d, CPU Edges: %d)\n\r", 
                EXPECTED_SUM, total_obtained, cgra_sum_out, total_obtained - cgra_sum_out);
    }
}
*/

void initCGRA(){
  plic_Init();
  plic_irq_set_priority(CGRA_INTR, 1);
  plic_irq_set_enabled(CGRA_INTR, kPlicToggleEnabled);
  plic_assign_external_irq_handler( CGRA_INTR, handler_irq_cgra);

  CSR_SET_BITS(CSR_REG_MSTATUS, 0x8);
  const uint32_t mask = 1 << 11;
  CSR_SET_BITS(CSR_REG_MIE, mask);
  cgra_intr_flag = 0;

  printf("Load CGRA instructions...\n");
  reset_csr_counters();
  cgra_cmem_init(cgra_imem_bitstream, cgra_kmem_bitstream);
  read_csr_counters();

  cgra.base_addr = mmio_region_from_addr((uintptr_t)CGRA_PERIPH_START_ADDRESS);
  cgra_slot = cgra_get_slot(&cgra);
}

void printMetrics(){
  printf("CGRA kernel executed: %d\n\r", cgra_perf_cnt_get_kernel(&cgra));
  int column_idx = 0;
  printf("CGRA column %d active cycles: %d\n\r", column_idx, cgra_perf_cnt_get_col_active(&cgra, column_idx));
  printf("CGRA column %d stall cycles : %d\n\r", column_idx, cgra_perf_cnt_get_col_stall(&cgra, column_idx));
}

void handler_irq_cgra(uint32_t id) {
  cgra_intr_flag = 1;
}