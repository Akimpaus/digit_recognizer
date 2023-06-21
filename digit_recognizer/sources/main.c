#include "dr_neural_network.h"

int main() {
    const size_t layers[] = { 2, 3 };
    dr_neural_network nn = dr_neural_network_create(2, layers);

    dr_neural_network_print(nn);

    dr_neural_network_free(&nn);
    return 0;
}