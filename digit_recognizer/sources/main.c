#include "dr_neural_network.h"

int main() {
    srand(time(NULL));

    const size_t layers[]     = { 1, 2, 3 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
    dr_neural_network_randomize_weights(nn, 0, 1);

    dr_neural_network_print_name(nn, "Neural network");

    dr_neural_network_free(&nn);
    return 0;
}