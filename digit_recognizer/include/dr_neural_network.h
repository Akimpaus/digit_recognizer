#ifndef DR_NEURAL_NETWORK_H
#define DR_NEURAL_NETWORK_H

#include "dr_matrix.h"

typedef struct {
    size_t layers_count;
    dr_matrix* layers;
    size_t connections_count;
    dr_matrix* connections;
} dr_neural_network;

dr_neural_network dr_neural_network_create(const size_t layers_count, const size_t* layers_sizes); // test

void dr_neural_network_free(dr_neural_network* neural_network); // test

#endif // DR_NEURAL_NETWORK_H