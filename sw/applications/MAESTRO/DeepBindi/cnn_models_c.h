#ifndef CNN_MODELS_C_H
#define CNN_MODELS_C_H

#include "nn_runtime.h"

Tensor *run_cnn_2d_v1(void);
Tensor *run_cnn_2d_v2(void);
Tensor *run_cnn_2d_v3(void);
Tensor *run_cnn_1d_v1(void);
Tensor *run_cnn_1d_v2(void);
Tensor *run_cnn_1d_v3(void);
Tensor *run_mobilenet_v3_custom(void);
Tensor *run_cnn_1d_tensorflow_sigmoid(void);
Tensor *run_cnn_1d_tensorflow_softmax(void);
Tensor *run_cnn_2d_tensorflow_softmax(void);

#endif
