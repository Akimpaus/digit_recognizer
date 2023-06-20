#include "dr_neural_network.h"

dr_neural_network dr_neural_network_create(const size_t layers_count, const size_t* layers_sizes) {
    dr_neural_network nn;
    nn.layers_count      = layers_count;
    nn.connections_count = layers_count - 1;

    nn.layers = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.layers_count);
    DR_ASSERT_MSG(nn.layers, "alloc layers error");
    for (size_t i = 0; i < nn.layers_count; ++i) {
        nn.layers[i] = dr_matrix_create_filled(1, layers_sizes[i], 0);
    }

    nn.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.connections_count);
    DR_ASSERT_MSG(nn.connections, "alloc connections error");
    for (size_t i = 0; i < nn.connections_count; ++i) {
        nn.connections[i] = dr_matrix_create_filled(layers_sizes[i + 1], layers_sizes[i], 0);
    }

    return nn;
}

void dr_neural_network_free(dr_neural_network* neural_network) {
    for (size_t i = 0; i < neural_network->layers_count; ++i) {
        dr_matrix_free(neural_network->layers + i);
    }
    free(neural_network->layers);
    neural_network->layers_count = 0;

    for (size_t i = 0; i < neural_network->connections_count; ++i) {
        dr_matrix_free(neural_network->connections + i);
    }
    free(neural_network->connections);
    neural_network->connections_count = 0;
}