//
// Created by Hossein Taji on 25/2/25.
//
#ifndef GELU_H
#define GELU_H

#include <stdlib.h>
#include "dense_layerC.h"

void gelu(Dense *dense, size_t length, int16_t *input, int16_t *output);

#endif // GELU_H