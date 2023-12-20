
#include "cgra.h"

// Matrix multiplication using the standard three loops
void mmulSoftware(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);
// Multiply the matrix in the cgra
void multiply_cgra(int32_t * matrixC, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB, cgra_t * cgra, uint8_t cgra_slot);