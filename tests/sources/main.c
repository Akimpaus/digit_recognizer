#include <utest.h>
#include <dr_neural_network.h>

UTEST_STATE();
int main(const int argc, const char* argv[]) {

    const size_t layers[] = { 2, 2 };
    dr_neural_network nn = dr_neural_network_create(sizeof(layers) / sizeof(size_t), layers);

    dr_neural_network_free(&nn);

    return utest_main(argc, argv);
}