#include <dr_testing_neural_network.h>

UTEST(dr_neural_network, dr_sigmoid) {
    EXPECT_NEAR(dr_sigmoid(-10), 0.00004539786870227497, 0.0000001);
    EXPECT_NEAR(dr_sigmoid(0), 0.5, 0.001);
    EXPECT_NEAR(dr_sigmoid(10), 0.9999546021312978, 0.0000001);
}

UTEST(dr_neural_network, dr_sigmoid_derivative) {
    EXPECT_NEAR(dr_sigmoid_derivative(-10), -110, 0.001);
    EXPECT_NEAR(dr_sigmoid_derivative(0), 0, 0.001);
    EXPECT_NEAR(dr_sigmoid_derivative(10), -90, 0.001);
}

UTEST(dr_neural_network, dr_tanh) {
    EXPECT_NEAR(dr_tanh(-10), -1, 0.001);
    EXPECT_NEAR(dr_tanh(0), 0, 0.001);
    EXPECT_NEAR(dr_tanh(10), 1, 0.001);
}

UTEST(dr_neural_network, dr_tanh_derivative) {
    EXPECT_NEAR(dr_tanh_derivative(-10), -99, 0.001);
    EXPECT_NEAR(dr_tanh_derivative(0), 1, 0.001);
    EXPECT_NEAR(dr_tanh_derivative(5), -24, 0.001);
}

UTEST(dr_neural_network, dr_relu) {
    EXPECT_NEAR(dr_relu(-10), 0, 0.001);
    EXPECT_NEAR(dr_relu(0), 0, 0.001);
    EXPECT_NEAR(dr_relu(10), 10, 0.001);
}

UTEST(dr_neural_network, dr_relu_derivative) {
    EXPECT_NEAR(dr_relu_derivative(-10), 0, 0.001);
    EXPECT_NEAR(dr_relu_derivative(0), 0, 0.001);
    EXPECT_NEAR(dr_relu_derivative(10), 1, 0.001);
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
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_1, expected_arr_layer_1, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_2, expected_arr_layer_2, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_connection_1, expected_arr_connection_1, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));

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
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_1, expected_arr_layer_1, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_2, expected_arr_layer_2, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0, 0,
            0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_connection_1, expected_arr_connection_1, 2, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));

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
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_1, expected_arr_layer_1, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_2, expected_arr_layer_2, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_layer_3, expected_arr_layer_3, 1, 3, DR_TESTING_MATRIX_EQUALS_EPSILON));

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
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_connection_1, expected_arr_connection_1, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));
        EXPECT_TRUE(dr_matrix_equals_to_array(
            mat_connection_2, expected_arr_connection_2, 2, 3, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
        EXPECT_FALSE(nn.activation_functions);
    }
}

UTEST(dr_neural_network, copy_create) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function acitvation_funcs[]   = { dr_tanh , dr_sigmoid };
        dr_activation_function acitvation_funcs_d[] = { dr_tanh_derivative , dr_sigmoid_derivative };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, acitvation_funcs, acitvation_funcs_d);

        nn.layers[0].elements[0] = 0.3;
        nn.connections[0].elements[0] = 5;

        dr_neural_network copy_nn = dr_neural_network_copy_create(nn);
        EXPECT_TRUE(dr_testing_neural_network_equals(nn, copy_nn, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        dr_neural_network_free(&copy_nn);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function acitvation_funcs[]   = { dr_tanh , dr_sigmoid };
        dr_activation_function acitvation_funcs_d[] = { dr_tanh_derivative , dr_sigmoid_derivative };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, acitvation_funcs, acitvation_funcs_d);
        dr_neural_network_randomize_weights(nn, 0, 1);

        dr_neural_network copy_nn = dr_neural_network_copy_create(nn);
        EXPECT_TRUE(dr_testing_neural_network_equals(nn, copy_nn, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        dr_neural_network_free(&copy_nn);
    }

    {
        const size_t layers[]     = { 4, 6, 4, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function acitvation_funcs[]   = { dr_tanh , dr_sigmoid, dr_relu };
        dr_activation_function acitvation_funcs_d[] =
            { dr_tanh_derivative , dr_sigmoid_derivative, dr_relu_derivative };
        dr_neural_network nn = dr_neural_network_create(layers, layers_count, acitvation_funcs, acitvation_funcs_d);
        dr_neural_network_randomize_weights(nn, 0, 1);

        dr_neural_network copy_nn = dr_neural_network_copy_create(nn);
        EXPECT_TRUE(dr_testing_neural_network_equals(nn, copy_nn, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        dr_neural_network_free(&copy_nn);
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
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]       = { 2, 2 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE input[] = { 1, 2 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count,
            DR_TESTING_NN_AF_PLUG, DR_TESTING_NN_AFD_PLUG);
        const DR_FLOAT_TYPE input[] = { 0, 2, 7, 10 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 4, DR_TESTING_MATRIX_EQUALS_EPSILON));
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

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 1, DR_TESTING_MATRIX_EQUALS_EPSILON));
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

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 2, DR_TESTING_MATRIX_EQUALS_EPSILON));
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

        EXPECT_TRUE(dr_matrix_equals_to_array(output_layer, expected_output, 1, 3, DR_TESTING_MATRIX_EQUALS_EPSILON));
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
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_sigmoid };
        dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, activation_functions_d);

        const DR_FLOAT_TYPE learning_rate = 0.1;
        const DR_FLOAT_TYPE error_arr[] = {
            0.2
        };

        // input
        nn.layers[0].elements[0] = 2;

        // connection input - output
        nn.connections[0].elements[0] = 0.3;

        // output
        nn.layers[1].elements[0] = 3;

        dr_neural_network_back_propagation(nn, learning_rate, error_arr);

        EXPECT_NEAR(nn.connections[0].elements[0], 0.0600, 0.0001);

        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 3, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_sigmoid };
        dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative, &dr_sigmoid_derivative };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, activation_functions_d);

        const DR_FLOAT_TYPE learning_rate = 0.1;
        const DR_FLOAT_TYPE error_arr[] = {
            0.2,
            0.5
        };

        // input
        nn.layers[0].elements[0] = 1;
        nn.layers[0].elements[1] = 2;

        // connection input - hidden
        nn.connections[0].elements[0] = 0.1;
        nn.connections[0].elements[1] = 0.2;
        nn.connections[0].elements[2] = 0.3;
        nn.connections[0].elements[3] = 0.4;
        nn.connections[0].elements[4] = 0.5;
        nn.connections[0].elements[5] = 0.6;

        // hidden
        nn.layers[1].elements[0] = 1;
        nn.layers[1].elements[1] = 2;
        nn.layers[1].elements[2] = 3;

        // connection hidden - output
        nn.connections[1].elements[0] = 0.6;
        nn.connections[1].elements[1] = 0.5;
        nn.connections[1].elements[2] = 0.4;
        nn.connections[1].elements[3] = 0.3;
        nn.connections[1].elements[4] = 0.2;
        nn.connections[1].elements[5] = 0.1;

        // output
        nn.layers[2].elements[0] = 2;
        nn.layers[2].elements[1] = 10;

        dr_neural_network_back_propagation(nn, learning_rate, error_arr);

        // connection hidden - output
        EXPECT_NEAR(nn.connections[1].elements[0], 0.56, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[1], 0.42, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[2], 0.28, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[3], -4.2, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[4], -8.8, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[5], -13.4, 0.0001);

        // connection input - hidden
        EXPECT_NEAR(nn.connections[0].elements[0], 0.1, 0.0001);
        EXPECT_NEAR(nn.connections[0].elements[1], 0.2, 0.0001);
        EXPECT_NEAR(nn.connections[0].elements[2], 0.26, 0.0001);
        EXPECT_NEAR(nn.connections[0].elements[3], 0.32, 0.0001);
        EXPECT_NEAR(nn.connections[0].elements[4], 0.422, 0.0001);
        EXPECT_NEAR(nn.connections[0].elements[5], 0.444, 0.0001);

        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 3, 3, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = {
            &dr_tanh, &dr_tanh, &dr_sigmoid };
        dr_activation_function activation_functions_d[] = {
            &dr_tanh_derivative, &dr_tanh_derivative, &dr_sigmoid_derivative };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, activation_functions_d);

        const DR_FLOAT_TYPE learning_rate = 0.3;
        const DR_FLOAT_TYPE error_arr[] = {
            5,
            -0.6
        };

        // input
        nn.layers[0].elements[0] = 1;
        nn.layers[0].elements[1] = 2;
        nn.layers[0].elements[2] = 3;
        nn.layers[0].elements[3] = 4;

        // connection input - hidden_1
        nn.connections[0].elements[0] = 0.1;
        nn.connections[0].elements[1] = 1;
        nn.connections[0].elements[2] = 2;
        nn.connections[0].elements[3] = 0.4;
        nn.connections[0].elements[4] = 0.5;
        nn.connections[0].elements[5] = 0;
        nn.connections[0].elements[6] = 0.7;
        nn.connections[0].elements[7] = 0.8;
        nn.connections[0].elements[8] = 0.9;
        nn.connections[0].elements[9] = -0.1;
        nn.connections[0].elements[10] = -0.2;
        nn.connections[0].elements[11] = -0.3;

        // hidden_1
        nn.layers[1].elements[0] = 3;
        nn.layers[1].elements[1] = 2;
        nn.layers[1].elements[2] = 2;

        // connection hidden_1 - hidden_2
        nn.connections[1].elements[0] = 0.6;
        nn.connections[1].elements[1] = 0.5;
        nn.connections[1].elements[2] = 0.4;
        nn.connections[1].elements[3] = 0.3;
        nn.connections[1].elements[4] = 0.2;
        nn.connections[1].elements[5] = 0.1;
        nn.connections[1].elements[6] = 0.2;
        nn.connections[1].elements[7] = 0.2;
        nn.connections[1].elements[8] = 0.1;

        // hidden_2
        nn.layers[2].elements[0] = 1;
        nn.layers[2].elements[1] = 2;
        nn.layers[2].elements[2] = 3;

        // connection hidden_2 - output
        nn.connections[2].elements[0] = 1;
        nn.connections[2].elements[1] = 1;
        nn.connections[2].elements[2] = -0.3;
        nn.connections[2].elements[3] = 1;
        nn.connections[2].elements[4] = 0;
        nn.connections[2].elements[5] = 0.4;

        // output
        nn.layers[3].elements[0] = 10;
        nn.layers[3].elements[1] = -5;

        dr_neural_network_back_propagation(nn, learning_rate, error_arr);

        // connection hidden_2 - output
        EXPECT_NEAR(nn.connections[2].elements[0], -134, 0.0001);
        EXPECT_NEAR(nn.connections[2].elements[1], -269, 0.0001);
        EXPECT_NEAR(nn.connections[2].elements[2], -405.3, 0.0001);
        EXPECT_NEAR(nn.connections[2].elements[3], 6.4, 0.0001);
        EXPECT_NEAR(nn.connections[2].elements[4], 10.8, 0.0001);
        EXPECT_NEAR(nn.connections[2].elements[5], 16.6, 0.0001);

        // connection hidden_1 - hidden_2
        EXPECT_NEAR(nn.connections[1].elements[0], 0.6, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[1], 0.5, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[2], 0.4, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[3], -13.2, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[4], -8.8, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[5], -8.9, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[6], 12.728, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[7], 8.552, 0.0001);
        EXPECT_NEAR(nn.connections[1].elements[8], 8.452, 0.0001);

        // connection input - hidden_1
        EXPECT_NEAR(nn.connections[0].elements[0], -9.001, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[1], -17.202, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[2], -25.302, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[3], -36.003, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[4], -2.067, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[5], -5.134, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[6], -7.000, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[7], -9.467, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[8], -0.977, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[9], -3.855, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[10], -5.832, 0.001);
        EXPECT_NEAR(nn.connections[0].elements[11], -7.81, 0.001);

        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, prediction_write) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.connections[0].elements[0] = 2;

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE));
        input[0]             = 0;
        DR_FLOAT_TYPE* prediction = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE));
        dr_neural_network_prediction_write(nn, input, prediction);

        EXPECT_EQ(prediction[0], 0.5);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.connections[0].elements[0] = 0.9;
        nn.connections[0].elements[1] = 0.3;
        nn.connections[0].elements[2] = 0.2;
        nn.connections[0].elements[3] = 0.8;

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * 2);
        input[0]             = 1;
        input[1]             = 0.5;
        DR_FLOAT_TYPE* prediction = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * 2);
        dr_neural_network_prediction_write(nn, input, prediction);

        EXPECT_NEAR(prediction[0], 0.740775, 0.00001);
        EXPECT_NEAR(prediction[1], 0.645656, 0.00001);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }

    {
        const size_t layers[]     = { 2, 2, 3, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid, &dr_tanh, &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

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

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * 2);
        input[0]             = 1;
        input[1]             = 0.5;
        DR_FLOAT_TYPE* prediction = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE));
        dr_neural_network_prediction_write(nn, input, prediction);

        EXPECT_NEAR(prediction[0], 0.957257, 0.0001);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }
}

UTEST(dr_neural_network, prediction_create) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.connections[0].elements[0] = 2;

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE));
        input[0]             = 0;
        DR_FLOAT_TYPE* prediction = dr_neural_network_prediction_create(nn, input);

        EXPECT_EQ(prediction[0], 0.5);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

        nn.connections[0].elements[0] = 0.9;
        nn.connections[0].elements[1] = 0.3;
        nn.connections[0].elements[2] = 0.2;
        nn.connections[0].elements[3] = 0.8;

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * 2);
        input[0]             = 1;
        input[1]             = 0.5;
        DR_FLOAT_TYPE* prediction = dr_neural_network_prediction_create(nn, input);

        EXPECT_NEAR(prediction[0], 0.740775, 0.00001);
        EXPECT_NEAR(prediction[1], 0.645656, 0.00001);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }

    {
        const size_t layers[]     = { 2, 2, 3, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[] = { &dr_sigmoid, &dr_tanh, &dr_sigmoid };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, DR_TESTING_NN_AFD_PLUG);

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

        DR_FLOAT_TYPE* input = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * 2);
        input[0]             = 1;
        input[1]             = 0.5;
        DR_FLOAT_TYPE* prediction = dr_neural_network_prediction_create(nn, input);

        EXPECT_NEAR(prediction[0], 0.957257, 0.0001);

        dr_neural_network_free(&nn);
        DR_FREE(prediction);
        DR_FREE(input);
    }
}

UTEST(dr_neural_network, train) {
    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_tanh, &dr_tanh };
    dr_activation_function activation_functions_d[] = { &dr_tanh_derivative, &dr_tanh_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    nn.connections[0].elements[0] = 1;
    nn.connections[0].elements[1] = 0.4;
    nn.connections[0].elements[2] = 0.1;
    nn.connections[0].elements[3] = -0.3;
    nn.connections[0].elements[4] = 0.1;
    nn.connections[0].elements[5] = 0.1;

    nn.connections[1].elements[0] = 0;
    nn.connections[1].elements[1] = -0.1;
    nn.connections[1].elements[2] = 0.7;

    const size_t train_data_size = 4;
    DR_FLOAT_TYPE** inputs       = dr_array_2d_float_alloc(2, train_data_size);
    DR_FLOAT_TYPE** outputs      = dr_array_2d_float_alloc(1, train_data_size);

    inputs[0][0] = 1; inputs[0][1] = 1;
    inputs[1][0] = 0; inputs[1][1] = 0;
    inputs[2][0] = 1; inputs[2][1] = 0;
    inputs[3][0] = 0; inputs[3][1] = 1;

    outputs[0][0] = 0;
    outputs[1][0] = 0;
    outputs[2][0] = 1;
    outputs[3][0] = 1;

    dr_neural_network_train(nn, 0.3, 100,
        (const DR_FLOAT_TYPE**)inputs, (const DR_FLOAT_TYPE**)outputs, train_data_size);
    DR_FLOAT_TYPE* prediction_1 = dr_neural_network_prediction_create(nn, inputs[0]);
    DR_FLOAT_TYPE* prediction_2 = dr_neural_network_prediction_create(nn, inputs[1]);
    DR_FLOAT_TYPE* prediction_3 = dr_neural_network_prediction_create(nn, inputs[2]);
    DR_FLOAT_TYPE* prediction_4 = dr_neural_network_prediction_create(nn, inputs[3]);

    EXPECT_NEAR(roundf(prediction_1[0]), 0.0, 0.001);
    EXPECT_NEAR(roundf(prediction_2[0]), 0.0, 0.001);
    EXPECT_NEAR(roundf(prediction_3[0]), 1.0, 0.001);
    EXPECT_NEAR(roundf(prediction_4[0]), 1.0, 0.001);

    dr_neural_network_free(&nn);
    dr_array_2d_float_free(inputs, train_data_size);
    dr_array_2d_float_free(outputs, train_data_size);
    DR_FREE(prediction_1);
    DR_FREE(prediction_2);
    DR_FREE(prediction_3);
    DR_FREE(prediction_4);
}

UTEST(dr_neural_network, dr_default_activation_function_to_string) {
    {
        char* str = dr_default_activation_function_to_string(&dr_sigmoid);
        EXPECT_EQ(strcmp(str, DR_SIGMOID_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_to_string(&dr_tanh);
        EXPECT_EQ(strcmp(str, DR_TANH_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_to_string(&dr_relu);
        EXPECT_EQ(strcmp(str, DR_RELU_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_to_string(&dr_testing_neural_network_func_nothing);
        EXPECT_EQ(str, NULL);
    }
}

UTEST(dr_neural_network, dr_default_activation_function_from_string) {
    EXPECT_EQ(dr_default_activation_function_from_string(DR_SIGMOID_STR), &dr_sigmoid);
    EXPECT_EQ(dr_default_activation_function_from_string(DR_TANH_STR), &dr_tanh);
    EXPECT_EQ(dr_default_activation_function_from_string(DR_RELU_STR), &dr_relu);
    EXPECT_EQ(dr_default_activation_function_from_string("NO"), NULL);
}

UTEST(dr_neural_network, dr_default_activation_function_derivative_to_string) {
    {
        char* str = dr_default_activation_function_derivative_to_string(&dr_sigmoid_derivative);
        EXPECT_EQ(strcmp(str, DR_SIGMOID_DERIVATIVE_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_derivative_to_string(&dr_tanh_derivative);
        EXPECT_EQ(strcmp(str, DR_TANH_DERIVATIVE_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_derivative_to_string(&dr_relu_derivative);
        EXPECT_EQ(strcmp(str, DR_RELU_DERIVATIVE_STR), 0);
        DR_FREE(str);
    }

    {
        char* str = dr_default_activation_function_derivative_to_string(&dr_testing_neural_network_func_nothing);
        EXPECT_EQ(str, NULL);
    }
}

UTEST(dr_neural_network, dr_default_activation_function_derivative_from_string) {
    EXPECT_EQ(dr_default_activation_function_derivative_from_string(DR_SIGMOID_DERIVATIVE_STR), &dr_sigmoid_derivative);
    EXPECT_EQ(dr_default_activation_function_derivative_from_string(DR_TANH_DERIVATIVE_STR), &dr_tanh_derivative);
    EXPECT_EQ(dr_default_activation_function_derivative_from_string(DR_RELU_DERIVATIVE_STR), &dr_relu_derivative);
    EXPECT_EQ(dr_default_activation_function_derivative_from_string("NO"), NULL);
}

bool custom_activation_function_transformer_called = false;
bool custom_activation_function_derivative_transformer_called = false;

char* dr_testing_details_activation_function_to_string(const dr_activation_function activation_function) {
    custom_activation_function_transformer_called = true;
    return dr_str_alloc("TEST");
}

char* dr_testing_details_activation_function_derivative_to_string(const dr_activation_function activation_function) {
    custom_activation_function_derivative_transformer_called = true;
    return dr_str_alloc("TEST");
}

UTEST(dr_neural_network, save_to_file_custom_activation_function_transformer) {
    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_tanh, &dr_tanh };
    dr_activation_function activation_functions_d[] = { &dr_tanh_derivative, &dr_tanh_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    EXPECT_FALSE(custom_activation_function_transformer_called);
    EXPECT_FALSE(custom_activation_function_derivative_transformer_called);

    const bool res = dr_neural_network_save_to_file_custom_activation_function_transformer(nn,
        dr_testing_details_activation_function_to_string,
        dr_testing_details_activation_function_derivative_to_string,
        "test_save_to_file_custom_activation_function_transformer.txt");

    EXPECT_TRUE(res);
    EXPECT_TRUE(custom_activation_function_transformer_called);
    EXPECT_TRUE(custom_activation_function_derivative_transformer_called);

    dr_neural_network_free(&nn);

    custom_activation_function_transformer_called = false;
    custom_activation_function_derivative_transformer_called = false;
}

UTEST(dr_neural_network, save_to_file) {
    const char* file_path = "test_save_to_file.txt";

    {
        const size_t layers[]     = { 2, 3, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_tanh };
        dr_activation_function activation_functions_d[] = { &dr_sigmoid_derivative, &dr_tanh_derivative };
        dr_neural_network nn =
            dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

        nn.connections[0].elements[0] = -1;
        nn.connections[0].elements[1] = 2;
        nn.connections[0].elements[2] = -3;
        nn.connections[0].elements[3] = 4;
        nn.connections[0].elements[4] = 5;
        nn.connections[0].elements[5] = 10;

        nn.connections[1].elements[0] = 0;
        nn.connections[1].elements[1] = 0.6;
        nn.connections[1].elements[2] = -0.2;

        const bool res = dr_neural_network_save_to_file(nn, file_path);
        EXPECT_TRUE(res);

        FILE* file = fopen(file_path, "r");
        char buffer[256] = { 0 };
        fscanf(file, "%s", buffer);
        EXPECT_TRUE(strcmp(buffer, DR_NEURAL_NETWORK_BEGIN_STR) == 0);
        size_t read_layers_count = 0;
        fscanf(file, "%zu", &read_layers_count);
        EXPECT_EQ(read_layers_count, layers_count);
        size_t read_layer_size = 0;
        fscanf(file, "%zu", &read_layer_size);
        EXPECT_EQ(read_layer_size, layers[0]);

        const size_t connection_count = read_layers_count - 1;
        for (size_t i = 0; i < connection_count; ++i) {
            const dr_matrix connection = nn.connections[i];
            size_t read_width  = 0;
            size_t read_height = 0;
            fscanf(file, "%zu %zu", &read_width, &read_height);
            EXPECT_EQ(read_width, connection.width);
            EXPECT_EQ(read_height, connection.height);
            for (size_t row = 0; row < read_height; ++row) {
                for (size_t column = 0; column < read_width; ++column) {
                    const float val = dr_matrix_unchecked_get_element(connection, column, row);
                    float read_val = 0;
                    fscanf(file, "%f", &read_val);
                    EXPECT_NEAR(read_val, val, 0.001);
                }
            }
            fscanf(file, "%zu", &read_layer_size);
            EXPECT_EQ(read_layer_size, nn.layers[i + 1].height);
            char* expected_func_name   = dr_default_activation_function_to_string(nn.activation_functions[i]);
            char* expected_func_d_name =
                dr_default_activation_function_derivative_to_string(nn.activation_functions_derivatives[i]);
            fscanf(file, "%s", buffer);
            EXPECT_EQ(strcmp(buffer, expected_func_name), 0);
            fscanf(file, "%s", buffer);
            EXPECT_EQ(strcmp(buffer, expected_func_d_name), 0);
            DR_FREE(expected_func_name);
            DR_FREE(expected_func_d_name);
        }
        fscanf(file, "%s", buffer);
        EXPECT_EQ(strcmp(buffer, DR_NEURAL_NETWORK_END_STR), 0);

        fclose(file);
        dr_neural_network_free(&nn);
    }


    {
        const size_t layers[]     = { 4, 5, 2, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_relu, &dr_tanh };
        dr_activation_function activation_functions_d[] =
            { &dr_sigmoid_derivative, &dr_relu_derivative, &dr_tanh_derivative };
        dr_neural_network nn =
            dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

        dr_neural_network_randomize_weights(nn, 0, 1);

        const bool res = dr_neural_network_save_to_file(nn, file_path);
        EXPECT_TRUE(res);

        FILE* file = fopen(file_path, "r");
        char buffer[256] = { 0 };
        fscanf(file, "%s", buffer);
        EXPECT_TRUE(strcmp(buffer, DR_NEURAL_NETWORK_BEGIN_STR) == 0);
        size_t read_layers_count = 0;
        fscanf(file, "%zu", &read_layers_count);
        EXPECT_EQ(read_layers_count, layers_count);
        size_t read_layer_size = 0;
        fscanf(file, "%zu", &read_layer_size);
        EXPECT_EQ(read_layer_size, layers[0]);

        const size_t connection_count = read_layers_count - 1;
        for (size_t i = 0; i < connection_count; ++i) {
            const dr_matrix connection = nn.connections[i];
            size_t read_width  = 0;
            size_t read_height = 0;
            fscanf(file, "%zu %zu", &read_width, &read_height);
            EXPECT_EQ(read_width, connection.width);
            EXPECT_EQ(read_height, connection.height);
            for (size_t row = 0; row < read_height; ++row) {
                for (size_t column = 0; column < read_width; ++column) {
                    const float val = dr_matrix_unchecked_get_element(connection, column, row);
                    float read_val = 0;
                    fscanf(file, "%f", &read_val);
                    EXPECT_NEAR(read_val, val, 0.001);
                }
            }
            fscanf(file, "%zu", &read_layer_size);
            EXPECT_EQ(read_layer_size, nn.layers[i + 1].height);
            char* expected_func_name   = dr_default_activation_function_to_string(nn.activation_functions[i]);
            char* expected_func_d_name =
                dr_default_activation_function_derivative_to_string(nn.activation_functions_derivatives[i]);
            fscanf(file, "%s", buffer);
            EXPECT_EQ(strcmp(buffer, expected_func_name), 0);
            fscanf(file, "%s", buffer);
            EXPECT_EQ(strcmp(buffer, expected_func_d_name), 0);
            DR_FREE(expected_func_name);
            DR_FREE(expected_func_d_name);
        }
        fscanf(file, "%s", buffer);
        EXPECT_EQ(strcmp(buffer, DR_NEURAL_NETWORK_END_STR), 0);

        fclose(file);
        dr_neural_network_free(&nn);
    }
}

static inline dr_activation_function dr_testing_details_activation_function_from_string(const char* string) {
    custom_activation_function_transformer_called = true;
    return dr_tanh;
}

static inline dr_activation_function dr_testing_details_activation_function_derivative_from_string(const char* string) {
    custom_activation_function_derivative_transformer_called = true;
    return dr_tanh_derivative;
}

UTEST(dr_neural_network, load_from_file_custom_activation_function_transformer) {
    const char* file_path = "test_load_from_file_custom_activation_function_transformer.txt";

    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    dr_activation_function activation_functions[]   = { &dr_tanh, &dr_tanh };
    dr_activation_function activation_functions_d[] = { &dr_tanh_derivative, &dr_tanh_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, activation_functions, activation_functions_d);

    EXPECT_FALSE(custom_activation_function_transformer_called);
    EXPECT_FALSE(custom_activation_function_derivative_transformer_called);

    dr_neural_network_save_to_file(nn, file_path);
    dr_neural_network_load_from_file_custom_activation_function_transformer(
        dr_testing_details_activation_function_from_string,
        dr_testing_details_activation_function_derivative_from_string,
        file_path);

    EXPECT_TRUE(custom_activation_function_transformer_called);
    EXPECT_TRUE(custom_activation_function_derivative_transformer_called);

    dr_neural_network_free(&nn);
}

UTEST(dr_neural_network, load_from_file) {
    const char* file_path = "test_load_from_file.txt";

    {
        const size_t layers[]     = { 2, 3, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_tanh, &dr_tanh };
        dr_activation_function activation_functions_d[] = { &dr_tanh_derivative, &dr_tanh_derivative };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, activation_functions_d);

        dr_neural_network_randomize_weights(nn, 0, 1);

        const bool save_res = dr_neural_network_save_to_file(nn, file_path);
        EXPECT_TRUE(save_res);

        dr_neural_network copy_nn = dr_neural_network_copy_create(nn);
        dr_neural_network_free(&nn);

        nn = dr_neural_network_load_from_file(file_path);

        EXPECT_TRUE(dr_neural_network_valid(nn));
        EXPECT_TRUE(dr_testing_neural_network_equals(copy_nn, nn, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        dr_neural_network_free(&copy_nn);
    }

    {
        const size_t layers[]     = { 4, 5, 2, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_activation_function activation_functions[]   = { &dr_sigmoid, &dr_tanh, &dr_relu };
        dr_activation_function activation_functions_d[] =
            { &dr_sigmoid_derivative, &dr_tanh_derivative, &dr_relu_derivative };
        dr_neural_network nn = dr_neural_network_create(
            layers, layers_count, activation_functions, activation_functions_d);

        dr_neural_network_randomize_weights(nn, 0, 1);

        const bool save_res = dr_neural_network_save_to_file(nn, file_path);
        EXPECT_TRUE(save_res);

        dr_neural_network copy_nn = dr_neural_network_copy_create(nn);
        dr_neural_network_free(&nn);

        nn = dr_neural_network_load_from_file(file_path);

        EXPECT_TRUE(dr_neural_network_valid(nn));
        EXPECT_TRUE(dr_testing_neural_network_equals(copy_nn, nn, DR_TESTING_MATRIX_EQUALS_EPSILON));

        dr_neural_network_free(&nn);
        dr_neural_network_free(&copy_nn);
    }
}