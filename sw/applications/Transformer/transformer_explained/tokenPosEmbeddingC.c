//
// Created by alireza on 10/6/23.
//

#include "tokenPosEmbeddingC.h"


void createTokenPosEmbedding(TokenPosEmbedding* tokenPosEmbedding, quant_bit_width* pos_matrix, quant_bit_width* cls_token_vector, size_t seq_len, size_t input_dim, size_t pos_matrix_dim) {
    tokenPosEmbedding->cls_token_vector_ = cls_token_vector;
    tokenPosEmbedding->pos_matrix_ = pos_matrix;
    tokenPosEmbedding->seq_len_ = seq_len;
    tokenPosEmbedding->input_dim_ = input_dim;
}

/**
 * @brief Concatenate the cls_token_ with the input array column-wise.
 * @param tpe Pointer to the TokenPosEmbedding structure containing the cls_token_ and dimensions.
 * @param input Pointer to the input array of shape (seq_len_, input_dim_).
 * @param concatenated_input Pointer to the output array of shape (seq_len_ + 1, input_dim_) where the result will be stored.
 */
void clsConcatenate(TokenPosEmbedding* tpe, quant_bit_width* input, quant_bit_width* concatenated_input) {
    // Can be done in CGRA
    // Copy cls_token_ into the concatenated array column-wise at the beginning
    for (size_t i = 0; i < tpe->input_dim_; ++i) {
        concatenated_input[i] = tpe->cls_token_vector_[i];
    }
    // Copy the input array into the concatenated array
    for (size_t i = 0; i < tpe->seq_len_ * tpe->input_dim_; ++i) {
        concatenated_input[i + tpe->input_dim_] = input[i];
    }
}

/**
 * @brief Add positional embeddings to the input array.
 * @param tpe Pointer to the TokenPosEmbedding structure containing the pos_matrix_ and dimensions.
 * @param input Pointer to the input array of shape (seq_len_ + 1, input_dim_) where positional embeddings will be added.
 * Done in-place
 */
void posEmbedding(TokenPosEmbedding* tpe, quant_bit_width* input) {
    // Can be done in CGRA
    for (size_t i = 0; i < (tpe->seq_len_ + 1); ++i) { // Including cls token
        for (size_t j = 0; j < tpe->input_dim_; ++j) {
            input[i * tpe->input_dim_+ j] += tpe->pos_matrix_[i * tpe->input_dim_ + j];
        }
    }
}

