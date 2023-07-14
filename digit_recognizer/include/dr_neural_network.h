#ifndef DR_NEURAL_NETWORK_H
#define DR_NEURAL_NETWORK_H

#include <math.h>
#include "dr_matrix.h"

typedef DR_FLOAT_TYPE(*dr_activation_function)(DR_FLOAT_TYPE);

typedef struct {
    size_t layers_count;
    dr_matrix* layers;
    size_t connections_count;
    dr_matrix* connections;
    dr_activation_function* activation_functions;
    dr_activation_function* activation_functions_derivatives;
} dr_neural_network;

static inline DR_FLOAT_TYPE dr_sigmoid(const DR_FLOAT_TYPE value) {
    return 1.0 / (1.0 + exp(-value));
}

static inline DR_FLOAT_TYPE dr_sigmoid_derivative(const DR_FLOAT_TYPE value) {
    return value * (1.0 - value);
}

static inline DR_FLOAT_TYPE dr_relu(const DR_FLOAT_TYPE value) {
    return value < 0.0 ? 0 : value;
}

static inline DR_FLOAT_TYPE dr_relu_derivative(const DR_FLOAT_TYPE value) {
    return value > 0;
}

static inline DR_FLOAT_TYPE dr_tanh(const DR_FLOAT_TYPE value) {
    return tanh(value);
}

static inline DR_FLOAT_TYPE dr_tanh_derivative(const DR_FLOAT_TYPE value) {
    return 1 - pow(value, 2);
}

bool dr_neural_network_valid(const dr_neural_network neural_network);

dr_neural_network dr_neural_network_create(
    const size_t* layers_sizes, const size_t layers_count,
    const dr_activation_function* activation_functions,
    const dr_activation_function* activation_functions_derivatives);

void dr_neural_network_free(dr_neural_network* neural_network);

void dr_neural_network_unchecked_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max);

void dr_neural_network_randomize_weights(
    dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max);

size_t dr_neural_network_unchecked_input_size(const dr_neural_network neural_network);

size_t dr_neural_network_input_size(const dr_neural_network neural_network);

size_t dr_neural_network_unchecked_output_size(const dr_neural_network neural_network);

size_t dr_neural_network_output_size(const dr_neural_network neural_network);

void dr_neural_network_unchecked_set_input(dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

void dr_neural_network_set_input(dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

void dr_neural_network_unchecked_get_output(const dr_neural_network neural_network, DR_FLOAT_TYPE* output);

void dr_neural_network_get_output(const dr_neural_network neural_network, DR_FLOAT_TYPE* output);

void dr_neural_network_unchecked_forward_propagation(dr_neural_network neural_network);

void dr_neural_network_forward_propagation(dr_neural_network neural_network);

void dr_neural_network_unchecked_back_propagation(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const DR_FLOAT_TYPE* output_errors);

void dr_neural_network_back_propagation(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const DR_FLOAT_TYPE* output_errors);

void dr_neural_network_unchecked_train(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const size_t epochs,
    const DR_FLOAT_TYPE** inputs, const DR_FLOAT_TYPE** outputs, const size_t count);

void dr_neural_network_train(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const size_t epochs,
    const DR_FLOAT_TYPE** inputs, const DR_FLOAT_TYPE** outputs, const size_t count); // test

void dr_neural_network_unchecked_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction);

void dr_neural_network_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction);

dr_matrix dr_neural_network_unchecked_prediction_create(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

dr_matrix dr_neural_network_prediction_create(const dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

void dr_neural_network_print(const dr_neural_network neural_network);

void dr_neural_network_print_name(const dr_neural_network neural_network, const char* name);

#endif // DR_NEURAL_NETWORK_H