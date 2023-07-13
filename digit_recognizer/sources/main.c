#include "dr_neural_network.h"

int main() {
    srand(time(NULL));

    DR_FLOAT_TYPE inputs[4][2] = {
        { 1, 0 },
        { 1, 1 },
        { 0, 1 },
        { 0, 0 }
    };

    DR_FLOAT_TYPE outputs[4] = {
        1,
        0,
        1,
        0
    };

    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_sigmoid };
    dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative, &dr_sigmoid_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    dr_neural_network_randomize_weights(nn, 0, 1);

    for (size_t i = 0; i < 1000000; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            dr_neural_network_set_input(nn, inputs[j]);
            dr_neural_network_unchecked_forward_propagation(nn);

            dr_matrix expected_output = dr_matrix_alloc(1, 1);
            dr_matrix real_output     = dr_matrix_alloc(1, 1);
            dr_matrix error           = dr_matrix_alloc(1, 1);

            expected_output.elements[0] = outputs[j];
            dr_neural_network_get_output(nn, real_output.elements);
            dr_matrix_subtraction_write(expected_output, real_output, error);
            const DR_FLOAT_TYPE err_sq = pow(error.elements[0], 2);
            //printf("error:%f - error^2:%f\n", error.elements[0], err_sq);
            //error.elements[0] = err_sq;

            dr_neural_network_unchecked_back_propagation(nn, 0.01, error);

            dr_matrix_free(&expected_output);
            dr_matrix_free(&real_output);
            dr_matrix_free(&error);
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        dr_neural_network_set_input(nn, inputs[i]);
        dr_neural_network_forward_propagation(nn);
        printf("%f -> %d\n", nn.layers[nn.layers_count - 1].elements[0], (int)roundf(nn.layers[nn.layers_count - 1].elements[0]));
    }

    dr_neural_network_free(&nn);
    return 0;
}