#include "performance.h"

#define BLOCK_SIZE 4

// Multiply the matrix in the cgra
void multiply_cgra(int * matrixA, int rowsA, int colsA, int * matrixB, int colsB, int * matrixC);
void countersInit();
void initCGRA();