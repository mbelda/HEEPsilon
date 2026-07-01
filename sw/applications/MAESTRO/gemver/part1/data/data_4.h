#ifndef DATA_GEMVER_P1_H
#define DATA_GEMVER_P1_H

#include <stdint.h>

/* Dataset Size: 4 */
#define N 4

int A[16] = {
    5, 3, 4, -2, 3, 3, -5, 0, -2, 5, 4, 4, 5, 0, 2, 1
};

int u1[4] = {
    -5, -1, 2, 3
};

int v1[4] = {
    5, -4, 5, 1
};

int u2[4] = {
    -3, -3, -4, -2
};

int v2[4] = {
    5, 0, 3, -4
};

int A_expected[16] = {
    -35, 23, -30, 5, -17, 7, -19, 11, -12, -3, 2, 22, 10, -12, 11, 12
};

#endif
