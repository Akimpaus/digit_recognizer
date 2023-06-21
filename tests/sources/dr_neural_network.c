#include <utest.h>
#include <dr_neural_network.h>

UTEST(dr_neural_network, create_free) {
    {
        const size_t layers[]    = { 1, 1 };
        const size_t layer_count = sizeof(layers) / sizeof(size_t);
        dr_neural_network nn     = dr_neural_network_create(layer_count, layers);
        EXPECT_EQ(nn.layers_count, layer_count);
        EXPECT_EQ(nn.connections_count, layer_count - 1);

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
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, 1, 1, expected_arr_layer_1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, 1, 1, expected_arr_layer_2));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, 1, 1, expected_arr_connection_1));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
    }

    {
        const size_t layers[]    = { 2, 2 };
        const size_t layer_count = sizeof(layers) / sizeof(size_t);
        dr_neural_network nn     = dr_neural_network_create(layer_count, layers);
        EXPECT_EQ(nn.layers_count, layer_count);
        EXPECT_EQ(nn.connections_count, layer_count - 1);

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
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, 1, 2, expected_arr_layer_1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, 1, 2, expected_arr_layer_2));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0, 0,
            0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, 2, 2, expected_arr_connection_1));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
    }

    {
        const size_t layers[]    = { 1, 2, 3 };
        const size_t layer_count = sizeof(layers) / sizeof(size_t);
        dr_neural_network nn     = dr_neural_network_create(layer_count, layers);
        EXPECT_EQ(nn.layers_count, layer_count);
        EXPECT_EQ(nn.connections_count, layer_count - 1);

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
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_1, 1, 1, expected_arr_layer_1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_2, 1, 2, expected_arr_layer_2));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_layer_3, 1, 3, expected_arr_layer_3));

        // TEST CONNECTIONS
        const DR_FLOAT_TYPE expected_arr_connection_1[] = {
            0, 0
        };
        const DR_FLOAT_TYPE expected_arr_connection_2[] = {
            0, 0, 0,
            0, 0, 0
        };
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_1, 2, 1, expected_arr_connection_1));
        EXPECT_TRUE(dr_matrix_equals_to_array(mat_connection_2, 3, 2, expected_arr_connection_2));

        dr_neural_network_free(&nn);
        EXPECT_EQ(nn.layers_count, 0);
        EXPECT_EQ(nn.connections_count, 0);
        EXPECT_FALSE(nn.layers);
        EXPECT_FALSE(nn.connections);
    }
}