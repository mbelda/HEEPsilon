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
#define CGRA_ROWS       4


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

const uint32_t  cgra_imem_transformer[CGRA_IMEM_SIZE] = {  0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0xc80000, 0xc80000, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0xa90000, 0x30b0000, 0x30d0000, 0x30f0000, 0xa80000, 0x16180000, 0x1090000, 0xa80000, 0x17180000, 0x61090000, 0xa80000, 0x18180000, 0x61090000, 0xa80000, 0x19180000, 0x61090000, 0x10b00000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,  };
static uint32_t cgra_kmem_transformer[CGRA_KMEM_SIZE] = {  0x0, 0xf012, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,    };

#define INPUT_MAT_A ROWS_A
#define INPUT_MAT_B ROWS_B*CGRA_ROWS

static int32_t cgra_input[CGRA_COLS][INPUT_MAT_A+INPUT_MAT_B]     __attribute__ ((aligned (4)));
static int32_t cgra_output[CGRA_COLS][ROWS_A*COLS_B]   __attribute__ ((aligned (4)));

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
    .in_n   = INPUT_MAT_A+INPUT_MAT_B,
    .out_n  = ROWS_A*COLS_B,
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


//static int16_t MatrixA[ROWS_A*COLS_A] = { 1,2,3, 4,5,6, 7,8,9};
//static int16_t MatrixB[ROWS_B*COLS_B] = { 10,10,10, 20,20,20, 30,30,30};


void config()
{
    /*
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
    */
    // TODO: Parse from csv
    // Matrix A 
    /* {1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16}
    */
    cgra_input[0][0] = 1; /*a00*/   cgra_input[1][0] = 2; /*a01*/   cgra_input[2][0] = 3; /*a02*/   cgra_input[3][0] = 4; /*a03*/
    cgra_input[0][1] = 5; /*a10*/   cgra_input[1][1] = 6;           cgra_input[2][1] = 7;           cgra_input[3][1] = 8;
    cgra_input[0][2] = 9; /*a20*/   cgra_input[1][2] = 10;          cgra_input[2][2] = 11;          cgra_input[3][2] = 12;
    cgra_input[0][3] = 13;/*a30*/   cgra_input[1][3] = 14;          cgra_input[2][3] = 15;          cgra_input[3][3] = 16;
    // Matrix B 
    /* {1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16}
    */
    cgra_input[0][4] = 1; /*b00*/   cgra_input[1][4] = 6; /*b11*/   cgra_input[2][4] = 11; /*b22*/   cgra_input[3][4] = 16; /*b33*/
    cgra_input[0][5] = 1; /*b00*/   cgra_input[1][5] = 6;           cgra_input[2][5] = 11;           cgra_input[3][5] = 16;
    cgra_input[0][6] = 1; /*b00*/   cgra_input[1][6] = 6;           cgra_input[2][6] = 11;           cgra_input[3][6] = 16;
    cgra_input[0][7] = 1; /*b00*/   cgra_input[1][7] = 6;           cgra_input[2][7] = 11;           cgra_input[3][7] = 16;

    cgra_input[0][8] = 5; /*b10*/   cgra_input[1][8] = 10; /*b21*/   cgra_input[2][8] = 15; /*b32*/  cgra_input[3][8] = 4; /*b03*/
    cgra_input[0][9] = 5; /*b10*/   cgra_input[1][9] = 10;           cgra_input[2][9] = 15;          cgra_input[3][9] = 4;
    cgra_input[0][10] = 5; /*b10*/  cgra_input[1][10] = 10;          cgra_input[2][10] = 15;         cgra_input[3][10] = 4;
    cgra_input[0][11] = 5; /*b10*/  cgra_input[1][11] = 10;          cgra_input[2][11] = 15;         cgra_input[3][11] = 4;

    cgra_input[0][12] = 9; /*b20*/  cgra_input[1][12] = 14; /*b31*/  cgra_input[2][12] = 3; /*b02*/  cgra_input[3][12] = 8; /*b13*/
    cgra_input[0][13] = 9; /*b20*/  cgra_input[1][13] = 14;          cgra_input[2][13] = 3;          cgra_input[3][13] = 8;
    cgra_input[0][14] = 9; /*b20*/  cgra_input[1][14] = 14;          cgra_input[2][14] = 3;          cgra_input[3][14] = 8;
    cgra_input[0][15] = 9; /*b20*/  cgra_input[1][15] = 14;          cgra_input[2][15] = 3;          cgra_input[3][15] = 8;

    cgra_input[0][16] = 13; /*b30*/  cgra_input[1][16] = 2; /*b01*/  cgra_input[2][16] = 7;  /*b12*/ cgra_input[3][16] = 12; /*b23*/
    cgra_input[0][17] = 13;          cgra_input[1][17] = 2;          cgra_input[2][17] = 7;          cgra_input[3][17] = 12;
    cgra_input[0][18] = 13;          cgra_input[1][18] = 2;          cgra_input[2][18] = 7;          cgra_input[3][18] = 12;
    cgra_input[0][19] = 13;          cgra_input[1][19] = 2;          cgra_input[2][19] = 7;          cgra_input[3][19] = 12;
    
    /*int cont = 0;
    for(int i=0; i < INPUT_MAT_A+INPUT_MAT_B; i++,cont++){
        cgra_input[0][cont] = 1;   cgra_input[1][cont] = 1;   cgra_input[2][cont] = 1;  cgra_input[3][cont] = 1;
    }*/

}

void software(void)
{
    transformer( outSW );
}

uint32_t check(void)
{
    uint32_t errors = 0;
    //This gives the result matrix transposed 'cause the RC compute their transposed position
    /*
    int cont = 0;
    for(int16_t i = 0; i < CGRA_COLS; i++) {
        for(int16_t j=0; j< CGRA_COLS; j++, cont++){
            outCGRA[cont] = cgra_output[i][j];
            PRINTF("cgra[%d,%d]=%d\n", i, j, cgra_output[i][j]);
        }
    }
    */

    int cont = 0;
    for(int16_t i = 0; i < CGRA_COLS; i++) {
        for(int16_t j=0; j< CGRA_COLS; j++, cont++){
            outCGRA[cont] = cgra_output[j][i];
        }
    }

    for( uint16_t i = 0; i < ROWS_A*COLS_B; i++ ){
        
        if(outSW[i]!=outCGRA[i]){
            errors++;
            PRINTF("[%d] %d != %d\n", i, outSW[i],outCGRA[i] );
        }
    }

    o_ret_cgra = outCGRA[4]; // ??

    return errors;
}

/****************************************************************************/
/**                                                                        **/
/**                                EOF                                     **/
/**                                                                        **/
/****************************************************************************/