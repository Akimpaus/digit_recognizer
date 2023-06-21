#include "dr_neural_network.h"
#include <string.h>

bool dr_neural_network_valid(const dr_neural_network neural_network) {
    return (neural_network.layers_count >= 2) &&
        (neural_network.connections_count == neural_network.layers_count - 1) &&
        neural_network.layers &&
        neural_network.connections;
}

dr_neural_network dr_neural_network_create(const size_t layers_count, const size_t* layers_sizes) {
    DR_ASSERT_MSG(layers_count >= 2, "neural network must contain 2 or more layers");
    DR_ASSERT_MSG(layers_sizes, "neural network layers sizes array cannot be NULL");

    dr_neural_network nn;
    nn.layers_count      = layers_count;
    nn.connections_count = layers_count - 1; // <- number of connections less than count of layers by 1

    nn.layers = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.layers_count);
    DR_ASSERT_MSG(nn.layers, "alloc neural network layers error");
    for (size_t i = 0; i < nn.layers_count; ++i) {
        const size_t layer_size = layers_sizes[i];
        DR_ASSERT_MSG(layer_size > 0, "layer size of neural network must be more than zero");
        nn.layers[i] = dr_matrix_create_filled(1, layer_size, 0);
    }

    nn.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.connections_count);
    DR_ASSERT_MSG(nn.connections, "alloc neural network connections error");
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
    neural_network->layers       = NULL;

    for (size_t i = 0; i < neural_network->connections_count; ++i) {
        dr_matrix_free(neural_network->connections + i);
    }
    free(neural_network->connections);
    neural_network->connections_count = 0;
    neural_network->connections       = NULL;
}

void dr_neural_network_unchecked_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        dr_matrix_unchecked_fill_random(neural_network.connections + i, min, max);
    }
}

void dr_neural_network_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to randomize weights for a not valid neural network");
    // the matrices are not checked when filling in
    dr_neural_network_unchecked_randomize_weights(neural_network, min, max);
}

void dr_neural_network_print(const dr_neural_network neural_network) {
    const char layer_str[]          = "layer";
    const char connection_str[]     = "connection";
    const size_t layer_str_len      = strlen(layer_str);
    const size_t connection_str_len = strlen(connection_str);
    const size_t max_str_len        = layer_str_len > connection_str_len ? layer_str_len : connection_str_len;
    const size_t layer_count_len    = dr_size_t_len(neural_network.layers_count);
    // 1 is length of symbol '='
    char* str_buffer = (char*)DR_MALLOC(sizeof(char) * (neural_network.layers_count + layer_count_len + 1));
    DR_ASSERT_MSG(str_buffer, "alloc str buffer error when printing neural network");

    printf("%s\n", "[");
    const size_t space = 4;
    for (size_t i = 0; i < neural_network.layers_count; ++i) {
        sprintf(str_buffer, "%s=%zu", layer_str, i + 1);
        dr_matrix_print_name_space(neural_network.layers[i], str_buffer, space, space, space);
        if (i < neural_network.connections_count) {
            sprintf(str_buffer, "%s=%zu", connection_str, i + 1);
            dr_matrix_print_name_space(neural_network.connections[i], str_buffer, space, space, space);
        }
    }
    printf("%s\n", "]");
    DR_FREE(str_buffer);
}

void dr_neural_network_print_name(const dr_neural_network neural_network, const char* name) {
    printf("%s: ", name);
    dr_neural_network_print(neural_network);
}