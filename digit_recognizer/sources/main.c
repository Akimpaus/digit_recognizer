#include "dr_neural_network.h"


int main() {
    srand(time(NULL));

    const size_t layers[]     = { 2, 2, 2, 10 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[] = { &dr_sigmoid, &dr_sigmoid, &dr_sigmoid };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions);


    dr_neural_network_randomize_weights(nn, 0, 1);

    DR_FLOAT_TYPE input[] = { 1, 0.5 };
    dr_neural_network_set_input(nn, input);
    dr_neural_network_unchecked_forward_propagation(nn);

    dr_neural_network_print_name(nn, "Neural network");

    dr_neural_network_free(&nn);
    return 0;
}