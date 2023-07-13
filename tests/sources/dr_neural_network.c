#include <utest.h>
#include "dr_testing_neural_network.h"

UTEST(dr_neural_network, dr_sigmoid) {
    EXPECT_NEAR(dr_sigmoid(-10), 0.00004539786870227497, 0.0000001);
    EXPECT_NEAR(dr_sigmoid(0), 0.5, 0.001);
    EXPECT_NEAR(dr_sigmoid(10), 0.9999546021312978, 0.0000001);
}

UTEST(dr_neural_network, dr_sigmoid_derivative) {
    EXPECT_EQ(dr_sigmoid_derivative(-10), -110);
    EXPECT_EQ(dr_sigmoid_derivative(0), 0);
    EXPECT_EQ(dr_sigmoid_derivative(10), -90);
}

UTEST(dr_neural_network, dr_relu) {
    EXPECT_EQ(dr_relu(-10), 0);
    EXPECT_EQ(dr_relu(0), 0);
    EXPECT_EQ(dr_relu(10), 10);
}

UTEST(dr_neural_network, dr_relu_derivative) {
    EXPECT_EQ(dr_relu_derivative(-10), 0);
    EXPECT_EQ(dr_relu_derivative(0), 0);
    EXPECT_EQ(dr_relu_derivative(10), 1);
}

UTEST(dr_neural_network, dr_tanh) {
    EXPECT_EQ(dr_tanh(-10), -1);
    EXPECT_EQ(dr_tanh(0), 0);
    EXPECT_EQ(dr_tanh(10), 1);
}

UTEST(dr_neural_network, dr_tanh_derivative) {
    EXPECT_EQ(dr_tanh_derivative(-10), -99);
    EXPECT_EQ(dr_tanh_derivative(0), 1);
    EXPECT_EQ(dr_tanh_derivative(5), -24);
}

UTEST(dr_neural_network, valid) {
    {
        dr_neural_network nn;
        nn.layers_count         = 0;
        nn.layers               = NULL;
        nn.connections_count    = 0;
        nn.connections          = NULL;
        nn.activation_functions = NULL;
        nn.activation_functions_derivatives = NULL;
        EXPECT_FALSE(dr_neural_network_valid(nn));
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(
            layers, layers_count, DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_TRUE(dr_neural_network_valid(nn));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, create_free) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function funcs[] = {
            &dr_testing_neural_network_func_double
        };
        dr_activation_function funcs_d[] = {
            &dr_testing_neural_network_func_triple
        };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, funcs, funcs_d);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);
        EXPECT_EQ(nn.activation_functions[0](1), 2);
        EXPECT_EQ(nn.activation_functions_derivatives[0](1), 3);

        // TEST LAYERS
        dr_matrix mat_layer_1      = nn.layers[0];
        dr_matrix mat_layer_2      = nn.layers[1];
        dr_matrix mat_connection_1 = nn.connections[0];
        const DR_FLOAT_TYPE expected_arr_layer_1[] = {
            0
        };
        const DR_FLOAT_TYPE expected_arr_layer_2[] = {
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, expected_arr_layer_1, 1, 1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, expected_arr_layer_2, 1, 1));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, expected_arr_connection_1, 1, 1));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
        EXPECT_FALSE(nn.activation_functions);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function funcs[] = {
            &dr_testing_neural_network_func_double
        };
        dr_activation_function funcs_d[] = {
            &dr_testing_neural_network_func_triple
        };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, funcs, funcs_d);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);
        EXPECT_EQ(nn.activation_functions[0](1), 2);
        EXPECT_EQ(nn.activation_functions_derivatives[0](1), 3);

        // TEST LAYERS
        dr_matrix mat_layer_1      = nn.layers[0];
        dr_matrix mat_layer_2      = nn.layers[1];
        dr_matrix mat_connection_1 = nn.connections[0];
        const DR_FLOAT_TYPE expected_arr_layer_1[] = {
            0,
            0
        };
        const DR_FLOAT_TYPE expected_arr_layer_2[] = {
            0,
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, expected_arr_layer_1, 1, 2));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, expected_arr_layer_2, 1, 2));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0, 0,
            0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, expected_arr_connection_1, 2, 2));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
        EXPECT_FALSE(nn.activation_functions);
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function funcs[] = {
            &dr_testing_neural_network_func_double,
            &dr_testing_neural_network_func_triple
        };
        dr_activation_function funcs_d[] = {
            &dr_testing_neural_network_func_triple,
            &dr_testing_neural_network_func_nothing
        };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, funcs, funcs_d);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);
        EXPECT_EQ(nn.activation_functions[0](1), 2);
        EXPECT_EQ(nn.activation_functions[1](1), 3);
        EXPECT_EQ(nn.activation_functions_derivatives[0](1), 3);
        EXPECT_EQ(nn.activation_functions_derivatives[1](1), 1);

        // TEST LAYERS
        dr_matrix mat_layer_1      = nn.layers[0];
        dr_matrix mat_layer_2      = nn.layers[1];
        dr_matrix mat_layer_3      = nn.layers[2];
        dr_matrix mat_connection_1 = nn.connections[0];
        dr_matrix mat_connection_2 = nn.connections[1];
        const DR_FLOAT_TYPE expected_arr_layer_1[] = {
            0
        };
        const DR_FLOAT_TYPE expected_arr_layer_2[] = {
            0,
            0
        };
        const DR_FLOAT_TYPE expected_arr_layer_3[] = {
            0,
            0,
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, expected_arr_layer_1, 1, 1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, expected_arr_layer_2, 1, 2));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_3, expected_arr_layer_3, 1, 3));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0,
            0
        };
        const DR_FLOAT_TYPE expected_arr_connection_2[] = {
            0, 0,
            0, 0,
            0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, expected_arr_connection_1, 1, 2));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_2, expected_arr_connection_2, 2, 3));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
        EXPECT_FALSE(nn.activation_functions);
    }
}

UTEST(dr_neural_network, randomize_weights) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(
            layers, layers_count, DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE min   = 0;
        const DR_FLOAT_TYPE max   = 0;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(
            layers, layers_count, DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE min   = 1.5;
        const DR_FLOAT_TYPE max   = 5;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(
            layers, layers_count, DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE min   = -0.2;
        const DR_FLOAT_TYPE max   = 0.5;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, input_size) {
    {
        const size_t layers[]       = { 1, 1 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_input_size(nn), 1);
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]       = { 2, 1 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_input_size(nn), 2);
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_input_size(nn), 4);
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, output_size) {
    {
        const size_t layers[]       = { 1, 1 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_output_size(nn), 1);
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]       = { 1, 2 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_output_size(nn), 2);
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        EXPECT_EQ(dr_neural_network_output_size(nn), 3);
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, set_input) {
    {
        const size_t layers[]       = { 1, 1 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE input[] = { 10 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 1));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]       = { 2, 2 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE input[] = { 1, 2 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 2));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE input[] = { 0, 2, 7, 10 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 4));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, get_output) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        dr_matrix output_layer    = nn.layers[nn.layers_count - 1];

        DR_FLOAT_TYPE output[1];
        const DR_FLOAT_TYPE expected_output[] = { 10 };
        dr_matrix_copy_array(output_layer, expected_output);
        dr_neural_network_get_output(nn, output);

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 1));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        dr_matrix output_layer    = nn.layers[nn.layers_count - 1];

        DR_FLOAT_TYPE output[2];
        const DR_FLOAT_TYPE expected_output[] = { 1, 2 };
        dr_matrix_copy_array(output_layer, expected_output);
        dr_neural_network_get_output(nn, output);

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 2));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        dr_matrix output_layer    = nn.layers[nn.layers_count - 1];

        DR_FLOAT_TYPE output[3];
        const DR_FLOAT_TYPE expected_output[] = { 0, 2, 7 };
        dr_matrix_copy_array(output_layer, expected_output);
        dr_neural_network_get_output(nn, output);

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 3));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, forward_propagation) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.layers[0].elements[0]      = 0;
        nn.connections[0].elements[0] = 2;
        dr_neural_network_forward_propagation(nn);

        EXPECT_EQ(nn.layers[1].elements[0], 0.5);

        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.layers[0].elements[0] = 1;
        nn.layers[0].elements[1] = 0.5;

        nn.connections[0].elements[0] = 0.9;
        nn.connections[0].elements[1] = 0.3;
        nn.connections[0].elements[2] = 0.2;
        nn.connections[0].elements[3] = 0.8;

        dr_neural_network_forward_propagation(nn);

        EXPECT_NEAR(nn.layers[1].elements[0], 0.740775, 0.00001);
        EXPECT_NEAR(nn.layers[1].elements[1], 0.645656, 0.00001);

        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 2, 3, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid, &dr_tanh, &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.layers[0].elements[0] = 1;
        nn.layers[0].elements[1] = 0.5;

        nn.connections[0].elements[0] = 0.9;
        nn.connections[0].elements[1] = 0.3;
        nn.connections[0].elements[2] = 0.2;
        nn.connections[0].elements[3] = 0.8;

        nn.connections[1].elements[0] = 0.5;
        nn.connections[1].elements[1] = 0.2;
        nn.connections[1].elements[2] = 1;
        nn.connections[1].elements[3] = 1;
        nn.connections[1].elements[4] = 0.4;
        nn.connections[1].elements[5] = 0.6;

        nn.connections[2].elements[0] = 1;
        nn.connections[2].elements[1] = 3;
        nn.connections[2].elements[2] = 0;

        dr_neural_network_forward_propagation(nn);

        EXPECT_NEAR(nn.layers[2].elements[0], 0.4617386, 0.00001);
        EXPECT_NEAR(nn.layers[2].elements[1], 0.8823832, 0.00001);
        EXPECT_NEAR(nn.layers[2].elements[2], 0.5939218, 0.00001);

        EXPECT_NEAR(nn.layers[3].elements[0], 0.957257, 0.0001);

        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, back_propagation) {
    
}