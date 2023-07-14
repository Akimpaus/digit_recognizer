#include <time.h>
#include "dr_neural_network.h"

int main() {
    srand(time(NULL));


    const size_t train_data_height = 4;

    DR_FLOAT_TYPE** inputs  = dr_array_2d_alloc(2, train_data_height);
    DR_FLOAT_TYPE** outputs = dr_array_2d_alloc(1, train_data_height);

    inputs[0][0] = 1; inputs[0][1] = 0;
    inputs[1][0] = 1; inputs[1][1] = 1;
    inputs[2][0] = 0; inputs[2][1] = 1;
    inputs[3][0] = 0; inputs[3][1] = 0;

    outputs[0][0] = 1;
    outputs[1][0] = 0;
    outputs[2][0] = 1;
    outputs[3][0] = 0;

    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_sigmoid };
    dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative, &dr_sigmoid_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    dr_neural_network_randomize_weights(nn, 0, 1);

    dr_neural_network_train(nn, 0.01, 1000000, (const DR_FLOAT_TYPE**)inputs, (const DR_FLOAT_TYPE**)outputs, 4);

    for (size_t i = 0; i < 4; ++i) {
        dr_neural_network_set_input(nn, inputs[i]);
        dr_neural_network_forward_propagation(nn);
        printf("%f -> %d\n", nn.layers[nn.layers_count - 1].elements[0], (int)roundf(nn.layers[nn.layers_count - 1].elements[0]));
    }

    dr_neural_network_free(&nn);
    dr_array_2d_free(inputs, train_data_height);
    dr_array_2d_free(outputs, train_data_height);
    return 0;
}