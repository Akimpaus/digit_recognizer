#include <application/dr_application.h>
#include <neural_network/dr_neural_network.h>

int main() {
    const size_t layers[]     = { 2, 3, 1 };
    const size_t layers_count = DR_ARRAY_LENGTH(layers);
    const dr_activation_function funcs[]   = { &dr_sigmoid, &dr_sigmoid };
    const dr_activation_function funcs_d[] = { &dr_tanh_derivative, &dr_relu_derivative };
    dr_neural_network nn = dr_neural_network_create(layers, layers_count, funcs, funcs_d);
    dr_neural_network_randomize_weights(nn, 0, 1);

    const bool save_res = dr_neural_network_save_to_file(nn, "test.txt");
    printf("%d\n", save_res);

    dr_neural_network_free(&nn);

    return 0;
}