/*
                              *******************
******************************* C SOURCE FILE *******************************
**                            *******************                          **
**                                                                         **
** project  : HEEPsilon                                                  **
** filename : transformer.c                                                 **
** version  : 1                                                            **
** date     : 2023-10-25                                                       **
**                                                                         **
*****************************************************************************
**                                                                         **
** Copyright (c) EPFL                                                      **
** All rights reserved.                                                    **
**                                                                         **
*****************************************************************************
*/

/***************************************************************************/
/***************************************************************************/

/**
* @file   transformer.c
* @date   2023-10-25
* @brief  A description of the kernel...
*
*/

#define _TRANSFORMER_C

/****************************************************************************/
/**                                                                        **/
/*                             MODULES USED                                 */
/**                                                                        **/
/****************************************************************************/
#include <stdint.h>

#include "transformer.h"
#include "function.h"

/****************************************************************************/
/**                                                                        **/
/*                        DEFINITIONS AND MACROS                            */
/**                                                                        **/
/****************************************************************************/

#define CGRA_COLS       3



/****************************************************************************/
/**                                                                        **/
/*                      PROTOTYPES OF LOCAL FUNCTIONS                       */
/**                                                                        **/
/****************************************************************************/

static void        config  (void);
static void        software(void);
static uint32_t    check   (void);

/****************************************************************************/
/**                                                                        **/
/*                            GLOBAL VARIABLES                              */
/**                                                                        **/
/****************************************************************************/

const uint32_t  cgra_imem_transformer[CGRA_IMEM_SIZE] = {  0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0xc80000, 0xc80000, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xab0000, 0x30d0000, 0x30f0000, 0xa80000, 0x17180000, 0x1090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,   };
static uint32_t cgra_kmem_transformer[CGRA_KMEM_SIZE] = {  0x0, 0x700e, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,  };

static int32_t cgra_input[CGRA_COLS][BLOCK_SIZE*BLOCK_SIZE]     __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_COLS][BLOCK_SIZE*BLOCK_SIZE]   __attribute__ ((aligned (4)));

static int16_t outSW[ROWS_A*COLS_B];
static int16_t outCGRA[ROWS_A*COLS_B];

    static uint32_t	i_index_soft;
    static uint32_t	i_index_cgra;
    static uint32_t	i_NumBits_soft;
    static uint32_t	i_NumBits_cgra;

    static uint32_t	o_ret_soft;
    static uint32_t	o_ret_cgra;


/****************************************************************************/
/**                                                                        **/
/*                           EXPORTED VARIABLES                             */
/**                                                                        **/
/****************************************************************************/

extern kcom_kernel_t transformer_kernel = {
    .kmem   = cgra_kmem_transformer,
    .imem   = cgra_imem_transformer,
    .col_n  = CGRA_N_COLS,
    .in_n   = BLOCK_SIZE*BLOCK_SIZE,
    .out_n  = BLOCK_SIZE*BLOCK_SIZE,
    .input  = cgra_input,
    .output = cgra_output,
    .config = config,
    .func   = software,
    .check  = check,
    .name   = "transformer",
};

/****************************************************************************/
/**                                                                        **/
/*                            LOCAL FUNCTIONS                               */
/**                                                                        **/
/****************************************************************************/

static int16_t MatrixA[ROWS_A*COLS_A] = { 1,2,3, 4,5,6, 7,8,9};
static int16_t MatrixB[ROWS_B*COLS_B] = { 10,10,10, 20,20,20, 30,30,30};


void config()
{
    // First the matrixA
    int cont = 0;
    for(int16_t i = 0; i < ROWS_A; i++, cont++){
        cgra_input[0][cont] = MatrixA[i*3];
        cgra_input[1][cont] = MatrixA[i*3+1];
        cgra_input[2][cont] = MatrixA[i*3+2];
    }

    // Then the matrixB
    for(int16_t i = 0; i < ROWS_A; i++){
        for(int j = 0; j < 3; j++, cont++){
            cgra_input[0][cont] = MatrixA[i];                // [0][0], [1][0], [2][0]
            cgra_input[1][cont] = MatrixA[(i+1)%COLS_A + 1]; // [1][1], [2][1], [0][1]
            cgra_input[2][cont] = MatrixA[(i+2)%COLS_A + 2]; // [2][2], [0][2], [1][2]
        }
    }
    
}

void software(void)
{
    transformer( outSW );
}

uint32_t check(void)
{
    uint32_t errors = 0;

    for(int16_t i = 0; i < ROWS_A; i++) {
        for(int16_t j=0; j<COLS_B; j++){
            outCGRA[i*COLS_B+j] = cgra_output[i][j];
        }
    }

    for( uint16_t i = 0; i < BLOCK_SIZE*BLOCK_SIZE; i++ ){
            PRINTF("[%d] %d != %d\n", i, outSW[i],outCGRA[i] );
        // if( outImg[i] != Gold_Out_Img[i] ){
        //     errors++;
        // }
    }

    o_ret_cgra = outCGRA[4]; // ??

    return errors;
}

/****************************************************************************/
/**                                                                        **/
/**                                EOF                                     **/
/**                                                                        **/
/****************************************************************************/