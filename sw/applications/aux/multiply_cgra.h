// Sizes of the input and ouput buffers of the CGRA
#define CGRA_COL_INPUT_SIZE 6
#define MAX_ROWS_A 121
#define CGRA_COL_OUTPUT_SIZE CGRA_N_COLS*(CGRA_N_ROWS-1)*(MAX_ROWS_A/CGRA_N_COLS)
#define BLOCK_SIZE 3


// Matrix multiplication using the standard three loops
void mmulSoftware(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);
// Fill output buffers with zeroes
void fillOutputZeroes(int32_t * output, int rows, int cols);
// Handler for the CGRA interruption
void handler_irq_ext(uint32_t id);
// Process extra A rows
void processExtraARows(int rB, int cB, int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);
// Process extra A cols
void processExtraACols(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);
// Process extra B cols
void processExtraBCols(int rB, int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);

void multiply_cgra(int32_t * out, int32_t * matrixA, int rowsA, int colsA, int32_t * matrixB, int colsB);