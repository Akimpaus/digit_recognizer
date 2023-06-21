#ifndef DR_TESTING_NEURAL_NETWORK_H
#define DR_TESTING_NEURAL_NETWORK_H

#include "dr_testing_matrix.h"
#include <dr_neural_network.h>

static bool dr_testing_neural_network_randomized_weights(
    const dr_neural_network neural_network, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    for (size_t i = 0; i < neural_network.connections_count; ++i) {
        if (!dr_testing_matrix_filled_random(neural_network.connections[i], min, max)) {
            return false;
        }
    }
    return true;
}

#endif // DR_TESTING_NEURAL_NETWORK_H