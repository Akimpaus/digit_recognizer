#include <utest.h>
#include <dr_utils.h>

UTEST(dr_utils, size_t_len) {
    EXPECT_EQ(size_t_len(0), 1);
    EXPECT_EQ(size_t_len(11), 2);
    EXPECT_EQ(size_t_len(245), 3);
}