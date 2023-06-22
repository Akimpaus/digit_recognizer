#include <utest.h>
#include "dr_testing_neural_network.h"

UTEST(dr_neural_network, valid) {
    {
        dr_neural_network nn;
        nn.layers_count      = 0;
        nn.layers            = NULL;
        nn.connections_count = 0;
        nn.connections       = NULL;
        EXPECT_FALSE(dr_neural_network_valid(nn));
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        EXPECT_TRUE(dr_neural_network_valid(nn));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, create_free) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);

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
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);

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
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        EXPECT_EQ(nn.layers_count, layers_count);
        EXPECT_EQ(nn.connections_count, layers_count - 1);

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
            0, 0
        };
        const DR_FLOAT_TYPE expected_arr_connection_2[] = {
            0, 0, 0,
            0, 0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, expected_arr_connection_1, 2, 1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_2, expected_arr_connection_2, 3, 2));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
    }
}

UTEST(dr_neural_network, randomize_weights) {
    {
        const size_t layers[]     = { 1, 1 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE min   = 0;
        const DR_FLOAT_TYPE max   = 0;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 2, 2 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE min   = 1.5;
        const DR_FLOAT_TYPE max   = 5;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 1, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE min   = -0.2;
        const DR_FLOAT_TYPE max   = 0.5;
        dr_neural_network_randomize_weights(nn, min, max);
        EXPECT_TRUE(dr_testing_neural_network_randomized_weights(nn, min, max));
        dr_neural_network_free(&nn);
    }
}

UTEST(dr_neural_network, set_input) {
    {
        const size_t layers[]       = { 1, 1 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn        = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE input[] = { 10 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 1));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]       = { 2, 2 };
        const size_t layers_count   = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn        = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE input[] = { 1, 2 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 2));
        dr_neural_network_free(&nn);
    }

    {
        const size_t layers[]     = { 4, 2, 3 };
        const size_t layers_count = DR_ARRAY_LENGTH(layers);
        dr_neural_network nn      = dr_neural_network_create(layers, layers_count);
        const DR_FLOAT_TYPE input[] = { 0, 2, 7, 10 };
        dr_neural_network_set_input(nn, input);
        EXPECT_TRUE(dr_matrix_equals_to_array(nn.layers[0], input, 1, 4));
        dr_neural_network_free(&nn);
    }
}