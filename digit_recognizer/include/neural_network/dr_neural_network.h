#ifndef DR_NEURAL_NETWORK_H
#define DR_NEURAL_NETWORK_H

#include <math.h>
#include "dr_matrix.h"

typedef DR_FLOAT_TYPE(*dr_activation_function)(DR_FLOAT_TYPE);
typedef char*(*dr_activation_function_to_string_callback)(const dr_activation_function);
typedef dr_activation_function(*dr_activation_function_from_string_callback)(const char*);

typedef struct {
    size_t layers_count;
    dr_matrix* layers;
    size_t connections_count;
    dr_matrix* connections;
    dr_activation_function* activation_functions;
    dr_activation_function* activation_functions_derivatives;
} dr_neural_network;

static const char DR_NEURAL_NETWORK_BEGIN_STR[] = "DR_NEURAL_NETWORK_BEGIN";
static const char DR_NEURAL_NETWORK_END_STR[]   = "DR_NEURAL_NETWORK_END";

static const char DR_SIGMOID_STR[] = "DR_SIGMOID";
DR_FLOAT_TYPE dr_sigmoid(const DR_FLOAT_TYPE value);

static const char DR_SIGMOID_DERIVATIVE_STR[] = "DR_SIGMOID_DERIVATIVE";
DR_FLOAT_TYPE dr_sigmoid_derivative(const DR_FLOAT_TYPE value);

static const char DR_TANH_STR[] = "DR_TANH";
DR_FLOAT_TYPE dr_tanh(const DR_FLOAT_TYPE value);

static const char DR_TANH_DERIVATIVE_STR[] = "DR_TANH_DERIVATIVE";
DR_FLOAT_TYPE dr_tanh_derivative(const DR_FLOAT_TYPE value);

static const char DR_RELU_STR[] = "DR_RELU";
DR_FLOAT_TYPE dr_relu(const DR_FLOAT_TYPE value);

static const char DR_RELU_DERIVATIVE_STR[] = "DR_RELU_DERIVATIVE";
DR_FLOAT_TYPE dr_relu_derivative(const DR_FLOAT_TYPE value);

char* dr_default_activation_function_to_string(const dr_activation_function activation_function);

dr_activation_function dr_default_activation_function_from_string(const char* string);

char* dr_default_activation_function_derivative_to_string(const dr_activation_function activation_function_derivative);

dr_activation_function dr_default_activation_function_derivative_from_string(const char* string);

bool dr_neural_network_valid(const dr_neural_network neural_network);

dr_neural_network dr_neural_network_create(
    const size_t* layers_sizes, const size_t layers_count,
    const dr_activation_function* activation_functions,
    const dr_activation_function* activation_functions_derivatives);

dr_neural_network dr_neural_network_unchecked_copy_create(const dr_neural_network neural_network);

dr_neural_network dr_neural_network_copy_create(const dr_neural_network neural_network);

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
    const DR_FLOAT_TYPE** trains_inputs, const DR_FLOAT_TYPE** trains_outputs, const size_t train_count);

void dr_neural_network_train(
    dr_neural_network neural_network, const DR_FLOAT_TYPE learning_rate, const size_t epochs,
    const DR_FLOAT_TYPE** train_inputs, const DR_FLOAT_TYPE** train_outputs, const size_t train_count);

void dr_neural_network_unchecked_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction);

void dr_neural_network_prediction_write(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input, dr_matrix prediction);

dr_matrix dr_neural_network_unchecked_prediction_create(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

dr_matrix dr_neural_network_prediction_create(const dr_neural_network neural_network, const DR_FLOAT_TYPE* input);

bool dr_neural_network_save_to_file_custom_activation_function_transformer(const dr_neural_network neural_network,
    const dr_activation_function_to_string_callback activation_function_to_string_callback,
    const dr_activation_function_to_string_callback activation_function_derivative_to_string_callback,
    const char* file_path);

bool dr_neural_network_save_to_file(const dr_neural_network neural_network, const char* file_path);

dr_neural_network dr_neural_network_load_from_file_custom_activation_function_transformer(
    const dr_activation_function_from_string_callback activation_function_from_string_callback,
    const dr_activation_function_from_string_callback activation_function_derivative_from_string_callback,
    const char* file_path);

dr_neural_network dr_neural_network_load_from_file(const char* file_path); // test

void dr_neural_network_print(const dr_neural_network neural_network);

void dr_neural_network_print_name(const dr_neural_network neural_network, const char* name);

#endif // DR_NEURAL_NETWORK_H