//
// Created by alireza on 10/6/23.
// Approx version by Hossein Taji and Francesco Poluzzi on 25/2/25.
//

#ifndef FVLLMONTITRANSFORMER_SOFTMAXC_H
#define FVLLMONTITRANSFORMER_SOFTMAXC_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "../param.h"

void computeSoftmax(int16_t* input, size_t seq_len);

#define FIXED_MAX(a, b) ((a) > (b) ? (a) : (b))

#endif //FVLLMONTITRANSFORMER_SOFTMAXC_H
