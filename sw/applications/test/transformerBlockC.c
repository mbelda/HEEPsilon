//
// Created by alireza on 10/6/23.
//

#include <stdio.h>
#include "transformerBlockC.h"
#include "multiply_cgra.h"

SingleHeadSelfAttn global_selfatten [NUM_LAYERS * NUM_HEAD];
Dense global_query_layer[NUM_LAYERS * NUM_HEAD];
Dense global_key_layer[NUM_LAYERS * NUM_HEAD];
Dense global_value_layer[NUM_LAYERS * NUM_HEAD];

Dense global_condense[NUM_LAYERS];
Dense global_patch;
Dense global_FF[NUM_LAYERS * 2];
Dense global_mlp;

TransformerBlock  global_transformer_block;
TokenPosEmbedding  global_token_embedding;

TransformerBlock* createTransformerBlock(size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, size_t num_heads, size_t ff_size, int32_t** weightVector, int32_t** biasVector, int32_t* clsTokenVector, int32_t* posMatrix) {
    TransformerBlock* transformerBlock = &global_transformer_block;
    transformerBlock->num_heads_ = num_heads;
    transformerBlock->head_hidden_size_ = head_hidden_size;
    transformerBlock->input_dim_ = input_dim;
    transformerBlock->ff_size_ = ff_size;

    transformerBlock->addNorm = createAddNormalize(pre_seq_len, D_EMBEDDING, weightVector[0], biasVector[0]);
    transformerBlock->patchEmbedding = &global_patch;
    createDense(transformerBlock->patchEmbedding, D_EMBEDDING, D_MODEL, weightVector[1], biasVector[1]);
    transformerBlock->addNorm2 = createAddNormalize(pre_seq_len, D_MODEL, weightVector[2], biasVector[2]);
    transformerBlock->token = &global_token_embedding;
    createTokenPosEmbedding(transformerBlock->token, posMatrix, clsTokenVector, pre_seq_len, input_dim, D_SEQ + 1);

    for (int l = 0; l < 4; l++) {
        transformerBlock->transformer_layer_0_addNorm[l] = createAddNormalize((pre_seq_len + 1), D_MODEL, weightVector[l * 17 + 3], biasVector[l * 17 + 3]);

        for (int n = 0; n < num_heads; n++) {
            transformerBlock->selfatten[l * num_heads + n] = &global_selfatten[l * num_heads + n];
            transformerBlock->selfatten[l * num_heads + n]->query_layer = &global_query_layer[l * num_heads + n];
            transformerBlock->selfatten[l * num_heads + n]->key_layer = &global_key_layer[l * num_heads + n];
            transformerBlock->selfatten[l * num_heads + n]->value_layer = &global_value_layer[l * num_heads + n];

            create_SingleHeadSelfAttn(transformerBlock->selfatten[l * num_heads + n], (pre_seq_len + 1), input_dim, head_hidden_size, weightVector + l * 17 + 4 + n * 3);
        }

        transformerBlock->condense[l] = &global_condense[l];
        createDense(transformerBlock->condense[l], num_heads * head_hidden_size, input_dim, weightVector[l * 17 + num_heads * 3 + 4], biasVector[l * 17 + num_heads * 3 + 4]);

        transformerBlock->transformer_layer_1_addNorm[l] = createAddNormalize((pre_seq_len + 1), input_dim, weightVector[l * 17 + num_heads * 3 + 5], biasVector[l * 17 + num_heads * 3 + 5]);

        transformerBlock->feedForward0[l] = &global_FF[2*l];
        createDense(transformerBlock->feedForward0[l], input_dim, ff_size, weightVector[l * 17 + num_heads * 3 + 6], biasVector[l * 17 + num_heads * 3 + 6]);

        transformerBlock->feedForward1[l] = &global_FF[2*l + 1];
        createDense(transformerBlock->feedForward1[l], ff_size, input_dim, weightVector[l * 17 + num_heads * 3 + 7], biasVector[l * 17 + num_heads * 3 + 7]);
    }

    transformerBlock->mlp_head_norm = createAddNormalize(1, D_MODEL, weightVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 8], biasVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 8]);

    transformerBlock->mlp_head_linear = &global_mlp;
    createDense(transformerBlock->mlp_head_linear, D_MODEL, D_MODEL, weightVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 9], biasVector[(NUM_LAYERS - 1) * 17 + NUM_HEAD * 3 + 9]);

    return transformerBlock;
}


void destroyTransformerBlock(TransformerBlock* transformerBlock) {
    // Free dynamically allocated memory

    free(transformerBlock);
}

void computeFixedPoint(TransformerBlock* transformerBlock, size_t seq_len, quant_bit_width * input,
                       quant_bit_width * input_normalized, quant_bit_width * output,
                       quant_bit_width* intermediate, quant_bit_width* qkv, void * kperf) {

    //12x16x12
    printf("\rMultiply_cgra\n");
    multiply_cgra((kcom_perf_t *) kperf, input, 12, 16, input, 16, 12, output, 12, 12); 
    printf("\rEnd multiply_cgra\n");
}

