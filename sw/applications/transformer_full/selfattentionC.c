//
// Created by alireza on 10/6/23.
//

#include <stdio.h>
#include "selfattentionC.h"


void create_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn, size_t pre_seq_len, size_t input_dim, size_t head_hidden_size, int32_t** weightVector) {
    self_attn->pre_seq_len = pre_seq_len;
    self_attn->head_hidden_size = head_hidden_size;
    createDense(self_attn->query_layer, input_dim, head_hidden_size, weightVector[0], NULL);
    createDense(self_attn->key_layer, input_dim, head_hidden_size, weightVector[1], NULL);
    createDense(self_attn->value_layer, input_dim, head_hidden_size, weightVector[2], NULL);
}

void destroy_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn) {
    free(self_attn->query_layer_out);
    free(self_attn->key_layer_out);
    free(self_attn->key_transposed_layer_out);
    free(self_attn->value_layer_out);
    free(self_attn->attention_scores);

    destroyDense(self_attn->query_layer);
    destroyDense(self_attn->key_layer);
    destroyDense(self_attn->value_layer);

    free(self_attn);
}

void compute_SingleHeadSelfAttn(SingleHeadSelfAttn* self_attn, int32_t* input, int32_t* output, int32_t* qkv, int32_t* intermediate, cgra_t * cgra, uint8_t cgra_slot) {
    self_attn->query_layer_out = qkv;
    self_attn->key_layer_out = qkv + self_attn->pre_seq_len * self_attn->head_hidden_size;
    self_attn->value_layer_out = qkv + 2 * self_attn->pre_seq_len * self_attn->head_hidden_size;
    self_attn->key_transposed_layer_out = qkv + 3 * self_attn->pre_seq_len * self_attn->head_hidden_size;

    // This 3 mmul need to think they have 124 rows instead of 121
    // TODO: Check that the outputs have enough space for the extra 3 rows
    computeDense(self_attn->query_layer, self_attn->pre_seq_len +3, input, self_attn->query_layer_out, cgra, cgra_slot);
    computeDense(self_attn->key_layer, self_attn->pre_seq_len +3, input, self_attn->key_layer_out, cgra, cgra_slot);
    computeDense(self_attn->value_layer, self_attn->pre_seq_len +3, input, self_attn->value_layer_out, cgra, cgra_slot);

    transpose_quant(self_attn->key_layer_out, self_attn->key_transposed_layer_out, self_attn->pre_seq_len +3, self_attn->head_hidden_size);
    MatMul_scale(self_attn->key_transposed_layer_out, 1, self_attn->pre_seq_len * self_attn->head_hidden_size);

    MatMul_multiply(self_attn->pre_seq_len +3, self_attn->query_layer_out, self_attn->key_transposed_layer_out, intermediate, self_attn->head_hidden_size, self_attn->pre_seq_len +3, cgra, cgra_slot);
    // 124x124 -> 124x121 Llevar les columnes extra de deveres
    int32_t* auxIntermediate = intermediate + 124*124;
    transpose_quant(intermediate, auxIntermediate, self_attn->pre_seq_len +3, self_attn->pre_seq_len +3); // 124x124
    transpose_quant(auxIntermediate, intermediate, self_attn->pre_seq_len, self_attn->pre_seq_len +3); // 121x124
    // Now intermediate is 124x121
    computeSoftmax(intermediate, self_attn->pre_seq_len);
    
    MatMul_multiply(self_attn->pre_seq_len +3, intermediate, self_attn->value_layer_out, output, self_attn->pre_seq_len, self_attn->head_hidden_size, cgra, cgra_slot);
}
