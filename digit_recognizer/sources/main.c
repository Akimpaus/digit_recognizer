#include "dr_neural_network.h"

int main() {
    srand(time(NULL));

    DR_FLOAT_TYPE inputs[][2] = {
        { 1, 0 },
        { 0, 1 },
        { 1, 1 },
        { 0, 0 }
    };

    DR_FLOAT_TYPE outputs[][2] = {
        { 1 },
        { 1 },
        { 0 },
        { 0 }
    };

    const size_t layers[]     = { 1, 10, 10, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_sigmoid, &dr_sigmoid };
    dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative, &dr_sigmoid_derivative, &dr_sigmoid_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    dr_neural_network_randomize_weights(nn, 0, 1);

    // test
    dr_neural_network_unchecked_forward_propagation(nn);
    dr_matrix error           = dr_matrix_create_filled(1, layers[DR_ARRAY_LENGTH(layers) - 1], 1);
    dr_neural_network_unchecked_back_propagation(nn, 0.01, error);
    dr_neural_network_print_name(nn, "Neural network");
    return 0;
    //test 


    for (size_t i = 0; i < 100000; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            dr_neural_network_set_input(nn, inputs[j]);
            dr_neural_network_unchecked_forward_propagation(nn);

            dr_matrix expected_output = dr_matrix_create_from_array(outputs[j], 1, 1);
            dr_matrix real_output     = dr_matrix_alloc(1, 1);
            dr_matrix error           = dr_matrix_alloc(1, 1);

            dr_neural_network_get_output(nn, real_output.elements);
            dr_matrix_subtraction_write(expected_output, real_output, error);
            const size_t error_size = dr_matrix_size(error);
            for (size_t i = 0; i < error_size; ++i) {
                error.elements[i] = pow(error.elements[i], 2);
            }

            dr_neural_network_unchecked_back_propagation(nn, 0.01, error);
            dr_matrix_free(&expected_output);
            dr_matrix_free(&real_output);
            dr_matrix_free(&error);
        }
    }

    dr_neural_network_set_input(nn, inputs[2]);
    dr_neural_network_unchecked_forward_propagation(nn);
    dr_neural_network_print_name(nn, "Neural network");

    dr_neural_network_free(&nn);
    return 0;
}