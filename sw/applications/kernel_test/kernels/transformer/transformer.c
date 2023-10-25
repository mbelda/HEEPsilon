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

#define CGRA_COLS       4
#define IN_VAR_DEPTH    9 // Numero de filas de la matriz
#define OUT_VAR_DEPTH   9




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

const uint32_t  cgra_imem_transformer[CGRA_IMEM_SIZE] = {  0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0xc80000, 0xc80000, 0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0xa90000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa090009, 0x6a091fff, 0x0, 0x0, 0x6a091fff, 0x0, 0x0, 0x6a091fff, 0x60880006, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x4a090000, 0x6a180002, 0x6a080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x4a090000, 0x6a180002, 0x14080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5080000, 0x14080000, 0x13080000, 0x5080000, 0x14080000, 0x13080000, 0x5080000, 0x14080000, 0x13080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5080000, 0x14080000, 0x0, 0x5080000, 0x14080000, 0x0, 0x5080000, 0x14080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5080000, 0x14080000, 0x0, 0x5080000, 0x14080000, 0x0, 0x5080000, 0x14080000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2a080000, 0x13080000, 0x10b00000, 0x2a080000, 0x13080000, 0x10b00000, 0x2a080000, 0x13080000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,  };
static uint32_t cgra_kmem_transformer[CGRA_KMEM_SIZE] = {  0x0, 0xf00d, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };

static int32_t cgra_input[CGRA_COLS][IN_VAR_DEPTH]     __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_COLS][OUT_VAR_DEPTH]   __attribute__ ((aligned (4)));

static int16_t outSW[SEQ_LEN*OUTPUT_SIZE];
static int16_t outCGRA[SEQ_LEN*OUTPUT_SIZE];

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
    .in_n   = IN_VAR_DEPTH,
    .out_n  = OUT_VAR_DEPTH,
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

static int16_t blockedA[SEQ_LEN*INPUT_SIZE] = { {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4},{ 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8}, {9,9,9,9, 10,10,10,10, 11,11,11,11} };
static int16_t blockedB[INPUT_SIZE*OUTPUT_SIZE] = { {1,1,1,1,1, 1,1,1,1,1, 1, 2,2,2,2,2}, {2,2,2,2,2, 2, 3,3,3,3,3, 3,3,3,3,3}, {3, 4,4,4,4,4, 4,4,4,4,4, 4} };

#define ROWS_A     11
#define COLS_A      4
#define ROWS_B      4
#define COLS_B      11

void config()
{
    //Para cada bloque
    //Ojo!!! This only works if the output matrix is squared, aka, the number of rows is equal to the number of columns
    int numBlocks = (ROWS_A*COLS_A) / (BLOCK_SIZE*BLOCK_SIZE);
    for(int block = 0; block < numBlocks; block++){
        int contA = 0, contB = 0;
        int tam_block = (block == numBlocks - 1) ? ((ROWS_A*COLS_A) % (BLOCK_SIZE*BLOCK_SIZE)) : BLOCK_SIZE;

        for(int16_t i = 0; i < tam_block; i++){
            if (i == 0 || i >= tam_block*2/numBlocks) {
                for(int n = 0; n < NUM_COLS_CGRA; n++){
                    cgra_input[n][i] = blockedA[block][contA*NColsA + i + n];
                }
                contA++;
                
            } else {
                for(int n = 0; n < NUM_COLS_CGRA; n++){
                    cgra_input[n][i] = blockedB[block][contB + n];
                }
                contB++;
            }
        }
    }

    /*
    cgra_input[0][0] = blockA[contA*NColsA + i];
    cgra_input[1][0] = blockA[contA*NColsA + i + 1];
    ...
    cgra_input[3][0] = blockA[contA*NColsA + i + 3];
    contA++;
    cgra_input[0][1] = blockB[contB];
    cgra_input[1][1] = blockB[contB + 1];
    ...
    cgra_input[3][1] = blockB[contB + 3];
    contB++;
    cgra_input[0][2] = blockB[contB++];
    cgra_input[0][3] = blockB[contB++];
    cgra_input[0][4] = blockB[contB++];
    cgra_input[0][5] = blockA[contA++*NColsA + i];
    cgra_input[0][6] = blockA[contA++*NColsA + i];
    cgra_input[0][7] = blockA[contA++*NColsA + i];
    */
    
}

void software(void)
{
    transformer( outSW );
}

uint32_t check(void)
{
    uint32_t errors = 0;

    for(int16_t i = 0; i < IN_VAR_DEPTH; i++) {
        outCGRA[4 + i*3] = cgra_output[3][i];
    }

    for( uint16_t i = 0; i < IMG_DIM; i++ ){
            PRINTF("[%d] %d != %d\n", i, outSW[i],outCGRA[i] );
        // if( outImg[i] != Gold_Out_Img[i] ){
        //     errors++;
        // }


    }

    o_ret_cgra = outCGRA[4];

    return errors;
}

/****************************************************************************/
/**                                                                        **/
/**                                EOF                                     **/
/**                                                                        **/
/****************************************************************************/