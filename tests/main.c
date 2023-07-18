#include <utest.h>

UTEST_STATE();
int main(const int argc, const char* argv[]) {
    srand(time(NULL));
    return utest_main(argc, argv);
}