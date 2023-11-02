#define ROWS_A       4
#define COLS_A       4
#define ROWS_B       4
#define COLS_B       4
#define NUM_FRACTION_BITS 12

// MatrixA dims 11x4
// MatrixB dims 4x11

//3x3
//static int16_t MatrixA[ROWS_A*COLS_A] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
//static int16_t MatrixB[ROWS_B*COLS_B] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};

//4x4
static int16_t matrixA[ROWS_A*COLS_A] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
static int16_t matrixB[ROWS_B*COLS_B] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
/*
void mmul ( int16_t * output )
{
  for (int16_t i  = 0; i < SEQ_LEN; i+=BLOCK_SIZE){
    int16_t block_size_i = i + BLOCK_SIZE <= SEQ_LEN ? BLOCK_SIZE : SEQ_LEN - i;
    for (int16_t j = 0; j < OUTPUT_SIZE; j+=BLOCK_SIZE){
      int16_t block_size_j = j + BLOCK_SIZE <= OUTPUT_SIZE ? BLOCK_SIZE : OUTPUT_SIZE - j;
      // Multiply the block
      for (int16_t ii = i; (ii < i + block_size_i) && (ii < SEQ_LEN); ii++){
        for (int16_t jj = j; (jj < j + block_size_j) && (jj < OUTPUT_SIZE); jj++){
          int32_t sum = 0;
          for(int16_t k = 0; k < INPUT_SIZE; k++){
              sum += MatrixA[ii*INPUT_SIZE + k] * MatrixB[k*OUTPUT_SIZE + jj];
          }
          output[ii*OUTPUT_SIZE + jj] = (int16_t)(sum >> NUM_FRACTION_BITS);
        }
      }
    }
  }
}*/

void mmul (int16_t * output ){
  for(int row = 0; row < ROWS_A; row++){
    for(int col = 0; col < COLS_B; col++){
      int32_t sum = 0;
      for(int k = 0; k < COLS_A; k++){
        sum += matrixA[row*COLS_A + k] * matrixB[k*COLS_B + col];
      }
      output[row*COLS_B + col] = (int16_t)(sum);
    }
  }
}