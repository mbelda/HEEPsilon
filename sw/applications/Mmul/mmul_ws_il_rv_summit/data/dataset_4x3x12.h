#include <stdint.h>
#define ROWS_A 4
#define COLS_A 3
#define COLS_B 12
#define ROWS_B COLS_A
#define ROWS_C ROWS_A
#define COLS_C COLS_B
int32_t matrixA[ROWS_A*COLS_A] __attribute__((section(".xheep_data_interleaved"))) = 
{
34, -46, -25, -10, -33, -46, -31, 29, 15, 18, 49, 35
};
int32_t matrixB[COLS_A*COLS_B] __attribute__((section(".xheep_data_interleaved"))) = 
{
-30, -49, 22, -42, -27, 21, 26, -5, -7, 15, -29, -29, -19, -12, -10, 21, 4, -32, 8, -45, -25, -32, 47, 21, 22, 23, -17, 2, 39, -8, -27, 46, -2, -20, 14, -32
};
volatile int32_t matrixC[ROWS_A*COLS_B] __attribute__((section(".xheep_data_interleaved"))) = 
{
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
int32_t cpu_out[ROWS_A*COLS_B] = 
{
-696, -1689, 1633, -2444, -2077, 2386, 1191, 750, 962, 2482, -3498, -1152, -85, -172, 892, -365, -1656, 1214, 718, -581, 987, 1826, -1905, 1069, 709, 1516, -1227, 1941, 1538, -1699, -979, -460, -538, -1693, 2472, 1028, -701, -665, -689, 343, 1075, -1470, -85, -685, -1421, -1998, 2271, -613
};
