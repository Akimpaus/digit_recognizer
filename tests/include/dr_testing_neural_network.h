#ifndef DR_TESTING_NEURAL_NETWORK_H
#define DR_TESTING_NEURAL_NETWORK_H

#include "dr_testing_matrix.h"
#include <neural_network/dr_neural_network.h>

static dr_activation_function dr_testing_neural_network_activation_functions_plug[] = {
    &dr_tanh,
    &dr_relu,
    &dr_sigmoid
};
#define DR_TESTING_NN_AF_PLUG dr_testing_neural_network_activation_functions_plug

static dr_activation_function dr_testing_neural_network_activation_functions_derivatives_plug[] = {
    &dr_tanh_derivative,
    &dr_relu_derivative,
    &dr_sigmoid_derivative
};
#define DR_TESTING_NN_AFD_PLUG dr_testing_neural_network_activation_functions_derivatives_plug

static bool dr_testing_neural_network_randomized_weights(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        if (!dr_testing_matrix_filled_random(neural_network.connections[i], min, max)) {
            return false;
        }
    }
    return true;
}

static DR_FLOAT_TYPE dr_testing_neural_network_func_nothing(const DR_FLOAT_TYPE val) {
    return val;
}

static DR_FLOAT_TYPE dr_testing_neural_network_func_double(const DR_FLOAT_TYPE val) {
    return val * 2;
}

static DR_FLOAT_TYPE dr_testing_neural_network_func_triple(const DR_FLOAT_TYPE val) {
    return val * 3;
}

#endif // DR_TESTING_NEURAL_NETWORK_H