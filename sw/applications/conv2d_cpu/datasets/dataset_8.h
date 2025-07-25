#include <stdint.h>
#define IM_HEIGHT 8
#define IM_WIDTH 8
volatile int32_t image[IM_HEIGHT*IM_WIDTH] __attribute__((section(".xheep_data_interleaved"))) = 
{
35, 15, -47, -1, 0, -35, 27, 15, -29, 36, -38, -16, -49, -25, -30, -24, 46, -30, -8, 46, -44, -17, -48, -39, -8, 36, 20, -38, 41, 17, 35, -13, -23, -1, 44, -23, -31, -40, 0, -43, -31, -3, -47, -12, -38, 11, 28, 43, 36, 15, -38, -42, 39, 1, 41, 32, -44, 40, -45, 3, 31, -49, -30, 8
};
volatile int32_t output[IM_HEIGHT*IM_WIDTH] __attribute__((section(".xheep_data_interleaved"))) = 
{
0
};
volatile int32_t filter[9] = 
{
-3, -3, -4, 1, -1, -4, 4, 1, -2
};
volatile int32_t expected_output[(IM_HEIGHT -1)*(IM_WIDTH -1)] = 
{
0, 0, 0, 0, 0, 0, 0, 0, 0, 295, 18, 420, 450, -4, 27, 0, 0, 203, 104, 440, 308, 618, 577, 0, 0, -321, 184, 171, -128, 95, 311, 0, 0, -395, -4, -43, -17, -502, 14, 0, 0, 291, 161, -94, 173, 127, 84, 0, 0, 417, 528, -62, 162, -22, -699, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
