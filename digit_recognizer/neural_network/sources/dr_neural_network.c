#include <dr_neural_network.h>

DR_FLOAT_TYPE dr_sigmoid(const DR_FLOAT_TYPE value) {
    return 1.0 / (1.0 + exp(-value));
}

DR_FLOAT_TYPE dr_sigmoid_derivative(const DR_FLOAT_TYPE value) {
    return value * (1.0 - value);
}

DR_FLOAT_TYPE dr_tanh(const DR_FLOAT_TYPE value) {
    return tanh(value);
}

DR_FLOAT_TYPE dr_tanh_derivative(const DR_FLOAT_TYPE value) {
    return 1.0 - pow(value, 2);
}

DR_FLOAT_TYPE dr_relu(const DR_FLOAT_TYPE value) {
    return value < 0.0 ? 0 : value;
}

DR_FLOAT_TYPE dr_relu_derivative(const DR_FLOAT_TYPE value) {
    return value > 0;
}

char* dr_default_activation_function_to_string(const dr_activation_function activation_function) {
    DR_ASSERT_MSG(activation_function, "attempt to convert a NULL activation function to the string");
    if (activation_function == &dr_sigmoid) {
        return dr_str_alloc(DR_SIGMOID_STR);
    } else if (activation_function == &dr_tanh) {
        return dr_str_alloc(DR_TANH_STR);
    } else if (activation_function == &dr_relu) {
        return dr_str_alloc(DR_RELU_STR);
    } else {
        return NULL;
    }
}

dr_activation_function dr_default_activation_function_from_string(const char* string) {
    DR_ASSERT_MSG(string, "attempt to convert a NULL string to the activation function");
    if (strcmp(string, DR_SIGMOID_STR) == 0) {
        return &dr_sigmoid;
    } else if (strcmp(string, DR_TANH_STR) == 0) {
        return &dr_tanh;
    } else if (strcmp(string, DR_RELU_STR) == 0) {
        return &dr_relu;
    } else {
        return NULL;
    }
}

char* dr_default_activation_function_derivative_to_string(const dr_activation_function activation_function_derivative) {
    DR_ASSERT_MSG(activation_function_derivative,
        "attempt to convert a NULL activation function derivative to the string");
    if (activation_function_derivative == &dr_sigmoid_derivative) {
        return dr_str_alloc(DR_SIGMOID_DERIVATIVE_STR);
    } else if (activation_function_derivative == &dr_tanh_derivative) {
        return dr_str_alloc(DR_TANH_DERIVATIVE_STR);
    } else if (activation_function_derivative == &dr_relu_derivative) {
        return dr_str_alloc(DR_RELU_DERIVATIVE_STR);
    } else {
        return NULL;
    }
}

dr_activation_function dr_default_activation_function_derivative_from_string(const char* string) {
    DR_ASSERT_MSG(string, "attempt to convert a NULL string to the activation function derivative");
    if (strcmp(string, DR_SIGMOID_DERIVATIVE_STR) == 0) {
        return &dr_sigmoid_derivative;
    } else if (strcmp(string, DR_TANH_DERIVATIVE_STR) == 0) {
        return &dr_tanh_derivative;
    } else if (strcmp(string, DR_RELU_DERIVATIVE_STR) == 0) {
        return &dr_relu_derivative;
    } else {
        return NULL;
    }
}

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

    nn.activation_functions = (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * nn.connections_count);
    DR_ASSERT_MSG(nn.activation_functions, "alloc neural network activation functions error");
    nn.activation_functions_derivatives =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * nn.connections_count);
    DR_ASSERT_MSG(nn.activation_functions_derivatives, "alloc neural network activation functions derivatives error");

    nn.layers = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.layers_count);
    DR_ASSERT_MSG(nn.layers, "alloc neural network layers error");
    nn.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * nn.connections_count);
    DR_ASSERT_MSG(nn.connections, "alloc neural network connections error");

    nn.layers[0] = dr_matrix_create_filled(1, layers_sizes[0], 0);
    for (size_t i = 0; i < nn.connections_count; ++i) {
        const size_t layer_index = i + 1;
        nn.activation_functions[i] = activation_functions[i];
        nn.activation_functions_derivatives[i] = activation_functions_derivatives[i];
        nn.connections[i] = dr_matrix_create_filled(layers_sizes[i], layers_sizes[layer_index], 0);
        nn.layers[layer_index] = dr_matrix_create_filled(1, layers_sizes[layer_index], 0);
    }

    return nn;
}

dr_neural_network dr_neural_network_unchecked_copy_create(const dr_neural_network neural_network) {
    dr_neural_network new_neural_network;
    new_neural_network.layers_count      = neural_network.layers_count;
    new_neural_network.connections_count = neural_network.connections_count;

    new_neural_network.activation_functions =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * new_neural_network.connections_count);
    DR_ASSERT_MSG(new_neural_network.activation_functions,
        "alloc neural network activation functions error when copying");

    new_neural_network.activation_functions_derivatives =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * new_neural_network.connections_count);
    DR_ASSERT_MSG(new_neural_network.activation_functions_derivatives,
        "alloc neural network activation functions derivatives error when copying");

    new_neural_network.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * new_neural_network.connections_count);
    DR_ASSERT_MSG(new_neural_network.connections, "alloc neural network connections error when copying");

    new_neural_network.layers = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * new_neural_network.layers_count);
    DR_ASSERT_MSG(new_neural_network.layers, "alloc neural network layers error when copying");

    new_neural_network.layers[0] = dr_matrix_unchecked_copy_create(neural_network.layers[0]);

    for (size_t i = 0; i < new_neural_network.connections_count; ++i) {
        const size_t layer_index = i + 1;
        new_neural_network.connections[i] = dr_matrix_unchecked_copy_create(neural_network.connections[i]);
        new_neural_network.activation_functions[i]             = neural_network.activation_functions[i];
        new_neural_network.activation_functions_derivatives[i] = neural_network.activation_functions_derivatives[i];
        new_neural_network.layers[layer_index] = dr_matrix_unchecked_copy_create(neural_network.layers[layer_index]);
    }

    return new_neural_network;
}

dr_neural_network dr_neural_network_copy_create(const dr_neural_network neural_network) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to create a copy of a not valid neural network");
    return dr_neural_network_unchecked_copy_create(neural_network);
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

static inline dr_matrix dr_neural_network_details_activation_functions_derivatives_for_layer_matrix_create(
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

static inline void dr_neural_network_details_update_E_W_next_output_layer(const dr_neural_network neural_network,
    const DR_FLOAT_TYPE* output_errors, const dr_matrix W, dr_matrix* E, dr_matrix* W_next) {
    const size_t last_layer_size = neural_network.layers[neural_network.layers_count - 1].height;
    *E      = dr_matrix_create_from_array(output_errors, 1, last_layer_size);
    *W_next = dr_matrix_unchecked_copy_create(W);
}

static inline void dr_neural_network_details_update_E_W_next_hidden_layer(
    const dr_matrix W, dr_matrix* E, dr_matrix* W_next) {
    dr_matrix W_next_T = dr_matrix_unchecked_transpose_create(*W_next);
    dr_matrix_unchecked_free(W_next);
    const dr_matrix E_new = dr_matrix_unchecked_dot_create(W_next_T, *E);
    dr_matrix_unchecked_free(E);
    *E = E_new;
    dr_matrix_unchecked_free(&W_next_T);
    *W_next = dr_matrix_unchecked_copy_create(W);
}

static inline void dr_neural_network_details_update_E_W_next(const dr_neural_network neural_network,
    const DR_FLOAT_TYPE* output_errors, const dr_matrix W, dr_matrix* E, dr_matrix* W_next, const size_t layer_index) {
    if (layer_index - 1 == neural_network.connections_count - 1) {
        dr_neural_network_details_update_E_W_next_output_layer(neural_network, output_errors, W, E, W_next);
    } else {
        dr_neural_network_details_update_E_W_next_hidden_layer(W, E, W_next);
    }
}

static inline dr_matrix dr_neural_network_details_W_delta_create(
    const dr_neural_network neural_network, const dr_matrix E,
    const DR_FLOAT_TYPE learning_rate, const size_t layer_index) {
    dr_matrix AFD = dr_neural_network_details_activation_functions_derivatives_for_layer_matrix_create(
        neural_network, layer_index);

    dr_matrix AFD_mult_E = dr_matrix_unchecked_multiplication_create(AFD, E);
    dr_matrix_unchecked_free(&AFD);

    const dr_matrix O = neural_network.layers[layer_index - 1];
    dr_matrix O_T     = dr_matrix_unchecked_transpose_create(O);

    dr_matrix W_delta = dr_matrix_unchecked_dot_create(AFD_mult_E, O_T);
    dr_matrix_unchecked_scale_write(W_delta, learning_rate, W_delta);
    dr_matrix_unchecked_free(&AFD_mult_E);
    dr_matrix_unchecked_free(&O_T);

    return W_delta;
}

static inline void dr_neural_network_details_apply_W_delta(const dr_neural_network neural_network, const dr_matrix E,
    dr_matrix W, const DR_FLOAT_TYPE learning_rate, const size_t layer_index) {
    dr_matrix W_delta = dr_neural_network_details_W_delta_create(neural_network, E, learning_rate, layer_index);
    dr_matrix_unchecked_addition_write(W, W_delta, W);
    dr_matrix_unchecked_free(&W_delta);
}

void dr_neural_network_unchecked_back_propagation(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const DR_FLOAT_TYPE* output_errors) {
    dr_matrix E      = dr_matrix_create_empty();
    dr_matrix W_next = dr_matrix_create_empty();

    for (size_t layer_index = neural_network.layers_count - 1; layer_index > 0; --layer_index) {
        dr_matrix W = neural_network.connections[layer_index - 1];
        dr_neural_network_details_update_E_W_next(neural_network, output_errors, W, &E, &W_next, layer_index);
        dr_neural_network_details_apply_W_delta(neural_network, E, W, learning_rate, layer_index);
    }

    dr_matrix_unchecked_free(&E);
    dr_matrix_unchecked_free(&W_next);
}

void dr_neural_network_back_propagation(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const DR_FLOAT_TYPE* output_errors) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to call a back propagation for a not valid neural network");
    DR_ASSERT_MSG(output_errors, "attempt to call back_propagation for the neural network with NULL output_errors");
    dr_neural_network_unchecked_back_propagation(neural_network, learning_rate, output_errors);
}

void dr_neural_network_unchecked_train(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const size_t epochs,
    const DR_FLOAT_TYPE** train_inputs, const DR_FLOAT_TYPE** train_outputs, const size_t train_count) {
    const size_t neural_network_output_size = dr_neural_network_unchecked_output_size(neural_network);

    DR_FLOAT_TYPE* errors      = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * neural_network_output_size);
    DR_ASSERT_MSG(errors, "buffer for errors alloc error, when training neural network");
    DR_FLOAT_TYPE* real_output = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * neural_network_output_size);
    DR_ASSERT_MSG(real_output, "buffer for real output alloc error, when training neural network");

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t data_index = 0; data_index < train_count; ++data_index) {
            const DR_FLOAT_TYPE* input         = train_inputs[data_index];
            const DR_FLOAT_TYPE* target_output = train_outputs[data_index];

            dr_neural_network_unchecked_set_input(neural_network, input);
            dr_neural_network_unchecked_forward_propagation(neural_network);
            dr_neural_network_unchecked_get_output(neural_network, real_output);

            for (size_t i = 0; i < neural_network_output_size; ++i) {
                errors[i] = target_output[i] - real_output[i];
            }

            dr_neural_network_unchecked_back_propagation(neural_network, learning_rate, errors);
        }
    }

    DR_FREE(errors);
    DR_FREE(real_output);
}

void dr_neural_network_train(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const size_t epochs,
    const DR_FLOAT_TYPE** train_inputs, const DR_FLOAT_TYPE** train_outputs, const size_t train_count) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to train a not valid neural network");
    DR_ASSERT_MSG(train_inputs, "attempt to train a neural network with a null train_inputs");
    DR_ASSERT_MSG(train_inputs, "attempt to train a neural network with a null train_outputs");
    DR_ASSERT_MSG(epochs > 0, "attempt to train a neural network with zero epochs");
    DR_ASSERT_MSG(train_count > 0, "attempt to train a neural network with empty train data");
    dr_neural_network_unchecked_train(neural_network, learning_rate, epochs, train_inputs, train_outputs, train_count);
}

void dr_neural_network_unchecked_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction) {
    dr_neural_network_unchecked_set_input(neural_network, input);
    dr_neural_network_unchecked_forward_propagation(neural_network);
    const dr_matrix output_layer = neural_network.layers[neural_network.layers_count - 1];
    const size_t output_size     = output_layer.width * output_layer.height;
    for (size_t i = 0; i < output_size; ++i) {
        prediction.elements[i] = output_layer.elements[i];
    }
}

void dr_neural_network_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to write a prediction with a not valid neural network");
    DR_ASSERT_MSG(input, "attempt to write a neural network prediction with a NULL input");
    DR_ASSERT_MSG(prediction.elements,
        "attempt to write a neural network prediction to a NULL matrix");
    DR_ASSERT_MSG(prediction.width == 1 && prediction.height == dr_neural_network_unchecked_output_size(neural_network),
        "attempt to write a neural network prediction to a matrix with wrong size");
    dr_neural_network_unchecked_prediction_write(neural_network, input, prediction);
}

dr_matrix dr_neural_network_unchecked_prediction_create(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input) {
    const size_t output_size = dr_neural_network_output_size(neural_network);
    dr_matrix prediction = dr_matrix_alloc(1, output_size);
    dr_neural_network_unchecked_prediction_write(neural_network, input, prediction);
    return prediction;
}

dr_matrix dr_neural_network_prediction_create(const dr_neural_network neural_network, const DR_FLOAT_TYPE* input) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network),
        "attempt to create a prediction with a not valid neural network");
    DR_ASSERT_MSG(input, "attempt to create a neural network prediction with a NULL input");
    return dr_neural_network_unchecked_prediction_create(neural_network, input);
}

bool dr_neural_network_save_to_file_custom_activation_function_transformer(const dr_neural_network neural_network,
    const dr_activation_function_to_string_callback activation_function_to_string_callback,
    const dr_activation_function_to_string_callback activation_function_derivative_to_string_callback,
    const char* file_path) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to save the not valid neural network to the file");
    DR_ASSERT_MSG(activation_function_to_string_callback,
        "can't save the neural network to a file: activation_function_to_string_callback was NULL");
    DR_ASSERT_MSG(activation_function_derivative_to_string_callback,
        "can't save the neural network to a file: activation_function_derivative_to_string_callback was NULL");
    DR_ASSERT_MSG(file_path, "attempt to save the neural netowrk with NULL file_path");

    FILE* file = fopen(file_path, "w");
    if (!file) {
        return false;
    }

    fprintf(file, "%s\n", DR_NEURAL_NETWORK_BEGIN_STR);
    fprintf(file, "%zu\n", neural_network.layers_count);
    fprintf(file, "%zu\n", neural_network.layers[0].height);

    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        const dr_matrix connection = neural_network.connections[i];

        fprintf(file, "%zu %zu\n", connection.width, connection.height);

        for (size_t row = 0; row < connection.height; ++row) {
            for (size_t column = 0; column < connection.width; ++column) {
                fprintf(file, "%f ", dr_matrix_unchecked_get_element(connection, column, row));
            }
            fprintf(file, "\n");
        }

        fprintf(file, "%zu\n", neural_network.layers[i + 1].height);

        char* activation_function_str = activation_function_to_string_callback(neural_network.activation_functions[i]);
        if (!activation_function_str) {
            return false;
        }
        fprintf(file, "%s\n", activation_function_str);
        DR_FREE(activation_function_str);

        char* activation_function_derivative_str =
            activation_function_derivative_to_string_callback(neural_network.activation_functions_derivatives[i]);
        if (!activation_function_derivative_str) {
            return false;
        }
        fprintf(file, "%s\n", activation_function_derivative_str);
        DR_FREE(activation_function_derivative_str);
    }

    fprintf(file, "%s", DR_NEURAL_NETWORK_END_STR);
    fclose(file);
    return true;
}

bool dr_neural_network_save_to_file(const dr_neural_network neural_network, const char* file_path) {
    return dr_neural_network_save_to_file_custom_activation_function_transformer(neural_network,
        &dr_default_activation_function_to_string, &dr_default_activation_function_derivative_to_string, file_path);
}

dr_neural_network dr_neural_network_load_from_file_custom_activation_function_transformer(
    const dr_activation_function_from_string_callback activation_function_from_string_callback,
    const dr_activation_function_from_string_callback activation_function_derivative_from_string_callback,
    const char* file_path) {
    DR_ASSERT_MSG(activation_function_from_string_callback,
        "can't save the neural network to a file: activation_function_from_string_callback was NULL");
    DR_ASSERT_MSG(activation_function_derivative_from_string_callback,
        "can't save the neural network to a file: activation_function_derivative_from_string_callback was NULL");
    DR_ASSERT_MSG(file_path, "attempt to load a neural netowrk from a file with NULL file_path");

    dr_neural_network neural_network = { 0 };

    FILE* file = fopen(file_path, "r");
    if (!file) {
        return neural_network;
    }

    char str_buffer[256] = { 0 };

    fscanf(file, "%s", str_buffer);
    if (strcmp(str_buffer, DR_NEURAL_NETWORK_BEGIN_STR) != 0) {
        return neural_network;
    }

    fscanf(file, "%zu", &neural_network.layers_count);
    neural_network.connections_count = neural_network.layers_count - 1;

    neural_network.activation_functions =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * neural_network.connections_count);
    DR_ASSERT_MSG(neural_network.activation_functions,
        "neural network activation functions alloc error, when loading from file");
    neural_network.activation_functions_derivatives =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * neural_network.connections_count);
    DR_ASSERT_MSG(neural_network.activation_functions_derivatives,
        "neural network activation functions derivatives alloc error, when loading from file");
    neural_network.layers      = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * neural_network.layers_count);
    DR_ASSERT_MSG(neural_network.layers,
        "neural network layers alloc error, when loading from file");
    neural_network.connections = (dr_matrix*)DR_MALLOC(sizeof(dr_matrix) * neural_network.connections_count);
    DR_ASSERT_MSG(neural_network.activation_functions,
        "neural network connections alloc error, when loading from file");

    size_t input_layer_height = 0;
    fscanf(file, "%zu", &input_layer_height);
    neural_network.layers[0] = dr_matrix_create_filled(1, input_layer_height, 0);


    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        size_t connection_width  = 0;
        size_t connection_height = 0;
        fscanf(file, "%zu %zu", &connection_width, &connection_height);
        dr_matrix* connection = neural_network.connections + i;
        *connection = dr_matrix_alloc(connection_width, connection_height);

        for (size_t row = 0; row < connection_height; ++row) {
            for (size_t column = 0; column < connection_width; ++column) {
                DR_FLOAT_TYPE element = 0;
                fscanf(file, "%f", &element);
                dr_matrix_unchecked_set_element(*connection, column, row, element);
            }
        }

        size_t layer_height = 0;
        fscanf(file, "%zu", &layer_height);
        neural_network.layers[i + 1] = dr_matrix_create_filled(1, layer_height, 0);

        fscanf(file, "%s", str_buffer);
        dr_activation_function activation_function = activation_function_from_string_callback(str_buffer);
        DR_ASSERT_MSG(activation_function, "neural_network error loading the activation function from a file");
        neural_network.activation_functions[i] = activation_function;

        fscanf(file, "%s", str_buffer);
        dr_activation_function activation_function_derivative =
            activation_function_derivative_from_string_callback(str_buffer);
        DR_ASSERT_MSG(activation_function_derivative,
            "neural_network error loading the activation function derivative from a file");
        neural_network.activation_functions_derivatives[i] = activation_function_derivative;
    }

    fscanf(file, "%s", str_buffer);
    if (strcmp(str_buffer, DR_NEURAL_NETWORK_END_STR) != 0) {
        dr_neural_network_free(&neural_network);
        return neural_network;
    }

    fclose(file);
    return neural_network;
}

dr_neural_network dr_neural_network_load_from_file(const char* file_path) {
    return dr_neural_network_load_from_file_custom_activation_function_transformer(
        dr_default_activation_function_from_string, dr_default_activation_function_derivative_from_string, file_path);
}

void dr_neural_network_print(const dr_neural_network neural_network) {
    DR_ASSERT_MSG(dr_neural_network_valid(neural_network), "attempt to print the not valid neural netowrk");

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