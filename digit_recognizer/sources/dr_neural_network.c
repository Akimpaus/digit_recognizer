#include "dr_neural_network.h"
#include <string.h>

bool dr_neural_network_valid(const dr_neural_network neural_network) {
    return (neural_network.layers_count >= 2) &&
        neural_network.layers &&
        (neural_network.connections_count == neural_network.layers_count - 1) &&
        neural_network.connections &&
        neural_network.activation_functions &&
        neural_network.activation_functions_derivatives;
}

dr_neural_network dr_neural_network_create(
    const size_t* layers_sizes, const size_t layers_count,
    const dr_activation_function* activation_functions,
    const dr_activation_function* activation_functions_derivatives) {
    DR_ASSERT_MSG(layers_count >= 2, "neural network must contain 2 or more layers");
    DR_ASSERT_MSG(layers_sizes, "neural network layers sizes array cannot be NULL");
    DR_ASSERT_MSG(activation_functions, "neural network activation functions cannot be NULL");
    DR_ASSERT_MSG(activation_functions_derivatives, "neural network activation functions derivatives cannot be NULL");

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

    nn.activation_functions = (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * nn.connections_count);
    DR_ASSERT_MSG(nn.activation_functions, "alloc neural network activation functions error");
    nn.activation_functions_derivatives =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * nn.connections_count);
    DR_ASSERT_MSG(nn.activation_functions_derivatives, "alloc neural network activation functions derivatives error");
    nn.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.connections_count);
    DR_ASSERT_MSG(nn.connections, "alloc neural network connections error");
    for (size_t i = 0; i < nn.connections_count; ++i) {
        nn.activation_functions[i] = activation_functions[i];
        nn.activation_functions_derivatives[i] = activation_functions_derivatives[i];
        nn.connections[i] = dr_matrix_create_filled(layers_sizes[i], layers_sizes[i + 1], 0);
    }

    return nn;
}

void dr_neural_network_free(dr_neural_network* neural_network) {
    DR_FREE(neural_network->activation_functions);
    neural_network->activation_functions = NULL;
    DR_FREE(neural_network->activation_functions_derivatives);
    neural_network->activation_functions_derivatives = NULL;

    for (size_t i = 0; i < neural_network->layers_count; ++i) {
        dr_matrix_free(neural_network->layers + i);
    }
    DR_FREE(neural_network->layers);
    neural_network->layers_count = 0;
    neural_network->layers       = NULL;

    for (size_t i = 0; i < neural_network->connections_count; ++i) {
        dr_matrix_free(neural_network->connections + i);
    }
    DR_FREE(neural_network->connections);
    neural_network->connections_count = 0;
    neural_network->connections       = NULL;
}

void dr_neural_network_unchecked_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        dr_matrix_unchecked_fill_random(neural_network.connections[i], min, max);
    }
}

void dr_neural_network_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to randomize weights for a not valid neural network");
    // the matrices are not checked when filling in
    dr_neural_network_unchecked_randomize_weights(neural_network, min, max);
}

size_t dr_neural_network_unchecked_input_size(const dr_neural_network neural_network) {
    return neural_network.layers[0].height;
}

size_t dr_neural_network_input_size(const dr_neural_network neural_network) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to get input size of a not valid neural network");
    return dr_neural_network_unchecked_input_size(neural_network);
}

size_t dr_neural_network_unchecked_output_size(const dr_neural_network neural_network) {
    return neural_network.layers[neural_network.layers_count - 1].height;
}

size_t dr_neural_network_output_size(const dr_neural_network neural_network) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to get output size of a not valid neural network");
    return dr_neural_network_unchecked_output_size(neural_network);
}

void dr_neural_network_unchecked_set_input(dr_neural_network neural_network, const DR_FLOAT_TYPE* input) {
    dr_matrix_unchecked_copy_array(neural_network.layers[0], input);
}

void dr_neural_network_set_input(dr_neural_network neural_network, const DR_FLOAT_TYPE* input) {
    DR_ASSERT_MSG(input, "attempt to set a NULL input array for a neural network");
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to set input for a not valid neural network");
    dr_neural_network_unchecked_set_input(neural_network, input);
}

void dr_neural_network_unchecked_get_output(const dr_neural_network neural_network, DR_FLOAT_TYPE* output) {
    dr_matrix_copy_to_array(neural_network.layers[neural_network.layers_count - 1], output);
}

void dr_neural_network_get_output(const dr_neural_network neural_network, DR_FLOAT_TYPE* output) {
    DR_ASSERT_MSG(output, "attempt to copy an output layer of a neural network to a NULL output array");
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to get output for a not valid neural network");
    dr_neural_network_unchecked_get_output(neural_network, output);
}

void dr_neural_network_unchecked_forward_propagation(dr_neural_network neural_network) {
    for (size_t i = 1; i < neural_network.layers_count; ++i) {
        const size_t prev_index    = i - 1;
        const dr_matrix connection = neural_network.connections[prev_index];
        const dr_matrix layer      = neural_network.layers[prev_index];
        dr_matrix result_layer     = *(neural_network.layers + i);
        dr_matrix_unchecked_dot_write(connection, layer, result_layer);
        // activating
        const size_t result_layer_size = dr_matrix_unchecked_size(result_layer);
        for (size_t j = 0; j < result_layer_size; ++j) {
            DR_FLOAT_TYPE* element = result_layer.elements + j;
            *element = neural_network.activation_functions[prev_index](*element);
        }
    }
}

void dr_neural_network_forward_propagation(dr_neural_network neural_network) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to call a forward propagation on a not valid neural network");
    dr_neural_network_unchecked_forward_propagation(neural_network);
}

dr_matrix dr_neural_network_unchecked_activation_functions_derivatives_for_layer_matrix_create(
    const dr_neural_network neural_network, const size_t layer_index) {
    const dr_matrix layer       = neural_network.layers[layer_index];
    dr_matrix result            = dr_matrix_alloc(layer.width, layer.height);
    const size_t layer_size     = dr_matrix_unchecked_size(layer);
    dr_activation_function func = neural_network.activation_functions_derivatives[layer_index - 1];
    for (size_t i = 0; i < layer_size; ++i) {
        result.elements[i] = func(layer.elements[i]);
    }
    return result;
}

dr_matrix dr_neural_network_activation_functions_derivatives_for_layer_matrix_create(
    const dr_neural_network neural_network, const size_t layer_index) {
    dr_neural_network_valid(neural_network);
    DR_ASSERT_MSG(layer_index > 0 && layer_index < neural_network.layers_count,
        "invalid index is specified when trying to call "
        "dr_neural_network_activation_functions_derivatives_for_layer_matrix_create, "
        "it must be more zero and less than layers_count");
    return dr_neural_network_unchecked_activation_functions_derivatives_for_layer_matrix_create(
        neural_network, layer_index);
}

void dr_neural_network_unchecked_back_propagation( // TODO make all unchecked here
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const dr_matrix output_error_matrix) {
    
}

void dr_neural_network_print(const dr_neural_network neural_network) {
    const char layer_str[]          = "layer";
    const char connection_str[]     = "connection";
    const size_t layer_str_len      = DR_ARRAY_LENGTH(layer_str);
    const size_t connection_str_len = DR_ARRAY_LENGTH(connection_str);
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