#ifndef _STIMULI_H_
#define _STIMULI_H_

#include <stdint.h>
#include "cgra.h"

 

#define ROWS_A 120    // Multiplo de CGRA_N_ROWS (4)
#define COLS_A 3    // Multiplo de BLOCK_SIZE (3)
#define ROWS_B 3
#define COLS_B 120   // Multiplo de CGRA_N_COLS*(CGRA_N_ROWS-1) (12)
#define ROWS_C ROWS_A
#define COLS_C COLS_B
#define BLOCK_SIZE 3


#endif // _STIMULI_H_
