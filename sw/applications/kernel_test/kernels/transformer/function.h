#define SEQ_LEN     11
#define INPUT_SIZE  4
#define OUTPUT_SIZE 11
#define BLOCK_SIZE  4
#define NUM_FRACTION_BITS 12

// MatrixA dims 11x4
// MatrixB dims 4x11

static int16_t MatrixA[SEQ_LEN*INPUT_SIZE] = { 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11 };
static int16_t MatrixB[INPUT_SIZE*OUTPUT_SIZE] = { 1,1,1,1,1, 1,1,1,1,1, 1, 2,2,2,2,2, 2,2,2,2,2, 2, 3,3,3,3,3, 3,3,3,3,3, 3, 4,4,4,4,4, 4,4,4,4,4, 4 };


void transformer ( int16_t * output )
{
  for (int16_t i  = 0; i < SEQ_LEN; i+=BLOCK_SIZE){
    int16_t block_size_i = i + BLOCK_SIZE <= SEQ_LEN ? BLOCK_SIZE : seqSEQ_LEN_len - i;
    for (int16_t j = 0; j < OUTPUT_SIZE; j+=BLOCK_SIZE){
      int16_t block_size_j = j + BLOCK_SIZE <= OUTPUT_SIZE ? BLOCK_SIZE : OUTPUT_SIZE - j;
      // Multiply the block
      for (int16_t ii = i; (ii < i + block_size_i) && (ii < SEQ_LEN); ii++){
        for (int16_t jj = j; (jj < j + block_size_j) && (jj < OUTPUT_SIZE); jj++){
          int32_t sum = 0;
          for(int16_t k = 0; k < INPUT_SIZE; k++){
              sum += input[ii*INPUT_SIZE + k] * weight[k*OUTPUT_SIZE + jj];
          }
          output[ii*OUTPUT_SIZE + jj] = (int16_t)(sum >> NUM_FRACTION_BITS);
        }
      }
    }
  }
}