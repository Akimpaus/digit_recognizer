#include <utest.h>
#include <dr_utils.h>

UTEST(dr_utils, dr_size_t_len) {
    EXPECT_EQ(dr_size_t_len(0), 1);
    EXPECT_EQ(dr_size_t_len(1), 1);
    EXPECT_EQ(dr_size_t_len(11), 2);
    EXPECT_EQ(dr_size_t_len(245), 3);
    EXPECT_EQ(dr_size_t_len(1000), 4);
}

UTEST(dr_utils, dr_random_float) {
    {
        const DR_FLOAT_TYPE min = 0;
        const DR_FLOAT_TYPE max = 0;
        const DR_FLOAT_TYPE val = dr_random_float(min, max);
        EXPECT_TRUE(val >= min && val <= max);
    }

    {
        const DR_FLOAT_TYPE min = 1;
        const DR_FLOAT_TYPE max = 1;
        const DR_FLOAT_TYPE val = dr_random_float(min, max);
        EXPECT_TRUE(val >= min && val <= max);
    }

    {
        const DR_FLOAT_TYPE min = 0.2;
        const DR_FLOAT_TYPE max = 0.4;
        const DR_FLOAT_TYPE val = dr_random_float(min, max);
        EXPECT_TRUE(val >= min && val <= max);
    }

    {
        const DR_FLOAT_TYPE min = 0.2;
        const DR_FLOAT_TYPE max = 0.4;
        const DR_FLOAT_TYPE val = dr_random_float(max, min);
        EXPECT_TRUE(val >= min && val <= max);
    }

    {
        const DR_FLOAT_TYPE min = -10;
        const DR_FLOAT_TYPE max = -5;
        const DR_FLOAT_TYPE val = dr_random_float(min, max);
        EXPECT_TRUE(val >= min && val <= max);
    }

    {
        const DR_FLOAT_TYPE min = -10;
        const DR_FLOAT_TYPE max = -5;
        const DR_FLOAT_TYPE val = dr_random_float(max, min);
        EXPECT_TRUE(val >= min && val <= max);
    }
}

UTEST(dr_utils, dr_array_2d_float_alloc) {
    const size_t width  = 2;
    const size_t height = 2;
    DR_FLOAT_TYPE** array = dr_array_2d_float_alloc(2, 2);

    DR_FLOAT_TYPE val = 0;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            array[y][x] = val++;
        }
    }

    EXPECT_EQ(array[0][0], 0);
    EXPECT_EQ(array[0][1], 1);
    EXPECT_EQ(array[1][0], 2);
    EXPECT_EQ(array[1][1], 3);

    dr_array_2d_float_free(array, height);
}