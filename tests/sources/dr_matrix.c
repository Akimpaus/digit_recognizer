#include <utest.h>
#include "dr_testing_matrix.h"

UTEST(dr_matrix, dr_matrix_correct_sizes) {
    EXPECT_TRUE(dr_matrix_correct_sizes(1, 1));
    EXPECT_TRUE(dr_matrix_correct_sizes(1, 2));
    EXPECT_TRUE(dr_matrix_correct_sizes(2, 1));
    EXPECT_FALSE(dr_matrix_correct_sizes(0, 1));
    EXPECT_FALSE(dr_matrix_correct_sizes(1, 0));
    EXPECT_TRUE(dr_matrix_correct_sizes(0, 0));
}

UTEST(dr_matrix, alloc_free) {
    {
        dr_matrix matrix = dr_matrix_alloc(0, 0);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);

        dr_matrix_free(&matrix);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(3, 1);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 3);
        EXPECT_EQ(matrix.height, 1);

        dr_matrix_free(&matrix);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(1, 3);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 1);
        EXPECT_EQ(matrix.height, 3);

        dr_matrix_free(&matrix);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }
}

UTEST(dr_matrix_fill, fill) {
    {
        dr_matrix matrix = dr_matrix_alloc(0, 0);
        dr_matrix_fill(matrix, 0);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(1, 1);
        const DR_FLOAT_TYPE fill_val = 0;
        dr_matrix_fill(matrix, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 1);
        EXPECT_EQ(matrix.height, 1);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(3, 5);
        const DR_FLOAT_TYPE fill_val = 1;
        dr_matrix_fill(matrix, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 3);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(5, 3);
        const DR_FLOAT_TYPE fill_val = 2;
        dr_matrix_fill(matrix, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 3);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(5, 5);
        const DR_FLOAT_TYPE fill_val = 3;
        dr_matrix_fill(matrix, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, fill_random) {
    {
        dr_matrix matrix = dr_matrix_alloc(0, 0);
        const DR_FLOAT_TYPE min = 0;
        const DR_FLOAT_TYPE max = 0;
        dr_matrix_fill_random(matrix, min, max);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(1, 1);
        const DR_FLOAT_TYPE min = 0.3;
        const DR_FLOAT_TYPE max = 1.5;
        dr_matrix_fill_random(matrix, min, max);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 1);
        EXPECT_EQ(matrix.height, 1);
        EXPECT_TRUE(dr_testing_matrix_filled_random(matrix, min, max));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(3, 5);
        const DR_FLOAT_TYPE min = -1.1;
        const DR_FLOAT_TYPE max = 5.6;
        dr_matrix_fill_random(matrix, min, max);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 3);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled_random(matrix, min, max));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(5, 3);
        const DR_FLOAT_TYPE min = 2.1;
        const DR_FLOAT_TYPE max = 101.6;
        dr_matrix_fill_random(matrix, min, max);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 3);
        EXPECT_TRUE(dr_testing_matrix_filled_random(matrix, min, max));
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_alloc(5, 5);
        const DR_FLOAT_TYPE min = -10.01;
        const DR_FLOAT_TYPE max = -4.3;
        dr_matrix_fill_random(matrix, min, max);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled_random(matrix, min, max));
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, copy_to_array) {
    {
        const size_t width  = 0;
        const size_t height = 0;
        const size_t size   = width * height;
        const DR_FLOAT_TYPE matrix_arr[] = {};
        dr_matrix matrix = dr_matrix_create_empty();
        DR_FLOAT_TYPE result_arr[size];
        dr_matrix_copy_to_array(matrix, result_arr);
        EXPECT_TRUE(dr_testing_matrix_array_equals(size, matrix_arr, result_arr));
    }

    {
        const size_t width  = 1;
        const size_t height = 1;
        const size_t size   = width * height;
        const DR_FLOAT_TYPE matrix_arr[] = { 9 };
        dr_matrix matrix = dr_matrix_create_from_array(matrix_arr, width, height);
        DR_FLOAT_TYPE result_arr[size];
        dr_matrix_copy_to_array(matrix, result_arr);
        EXPECT_TRUE(dr_testing_matrix_array_equals(size, matrix_arr, result_arr));
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 3;
        const size_t height = 3;
        const size_t size   = width * height;
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        dr_matrix matrix = dr_matrix_create_from_array(matrix_arr, width, height);
        DR_FLOAT_TYPE result_arr[size];
        dr_matrix_copy_to_array(matrix, result_arr);
        EXPECT_TRUE(dr_testing_matrix_array_equals(size, matrix_arr, result_arr));
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 5;
        const size_t height = 3;
        const size_t size   = width * height;
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15
        };
        dr_matrix matrix = dr_matrix_create_from_array(matrix_arr, width, height);
        DR_FLOAT_TYPE result_arr[size];
        dr_matrix_copy_to_array(matrix, result_arr);
        EXPECT_TRUE(dr_testing_matrix_array_equals(size, matrix_arr, result_arr));
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 3;
        const size_t height = 5;
        const size_t size   = width * height;
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15
        };
        dr_matrix matrix = dr_matrix_create_from_array(matrix_arr, width, height);
        DR_FLOAT_TYPE result_arr[size];
        dr_matrix_copy_to_array(matrix, result_arr);
        EXPECT_TRUE(dr_testing_matrix_array_equals(size, matrix_arr, result_arr));
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, copy_array) {
    {
        const size_t width  = 0;
        const size_t height = 0;
        const DR_FLOAT_TYPE arr[] = {};
        dr_matrix matrix = dr_matrix_alloc(width, height);
        dr_matrix_copy_array(matrix, arr);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, width);
        EXPECT_EQ(matrix.height, height);
    }

    {
        const size_t width  = 1;
        const size_t height = 1;
        const DR_FLOAT_TYPE arr[] = {
            1
        };
        dr_matrix matrix = dr_matrix_create_filled(width, height, 0);
        dr_matrix_copy_array(matrix, arr);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, arr, width, height));
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, width);
        EXPECT_EQ(matrix.height, height);
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 3;
        const size_t height = 3;
        const DR_FLOAT_TYPE arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        dr_matrix matrix = dr_matrix_create_filled(width, height, 0);
        dr_matrix_copy_array(matrix, arr);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, arr, width, height));
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, width);
        EXPECT_EQ(matrix.height, height);
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 5;
        const size_t height = 3;
        const DR_FLOAT_TYPE arr[] = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15
        };
        dr_matrix matrix = dr_matrix_create_filled(width, height, 0);
        dr_matrix_copy_array(matrix, arr);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, arr, width, height));
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, width);
        EXPECT_EQ(matrix.height, height);
        dr_matrix_free(&matrix);
    }

    {
        const size_t width  = 3;
        const size_t height = 5;
        const DR_FLOAT_TYPE arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15
        };
        dr_matrix matrix = dr_matrix_create_filled(width, height, 0);
        dr_matrix_copy_array(matrix, arr);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, arr, width, height));
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, width);
        EXPECT_EQ(matrix.height, height);
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, create_empty) {
    dr_matrix matrix = dr_matrix_create_empty();
    EXPECT_FALSE(matrix.elements);
    EXPECT_EQ(matrix.width, 0);
    EXPECT_EQ(matrix.height, 0);
}

UTEST(dr_matrix, create_filled) {
    {
        dr_matrix matrix = dr_matrix_create_filled(0, 0, 0);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        const DR_FLOAT_TYPE fill_val = 0;
        dr_matrix matrix = dr_matrix_create_filled(1, 1, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 1);
        EXPECT_EQ(matrix.height, 1);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        const DR_FLOAT_TYPE fill_val = 0;
        dr_matrix matrix = dr_matrix_create_filled(3, 5, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 3);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        const DR_FLOAT_TYPE fill_val = 0;
        dr_matrix matrix = dr_matrix_create_filled(5, 3, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 3);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }

    {
        const DR_FLOAT_TYPE fill_val = 0;
        dr_matrix matrix = dr_matrix_create_filled(5, 5, fill_val);
        EXPECT_TRUE(matrix.elements);
        EXPECT_EQ(matrix.width, 5);
        EXPECT_EQ(matrix.height, 5);
        EXPECT_TRUE(dr_testing_matrix_filled(matrix, fill_val));
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, create_from_array) {
    {
        const DR_FLOAT_TYPE* array = NULL;
        dr_matrix matrix = dr_matrix_create_from_array(array, 0, 0);
        EXPECT_FALSE(matrix.elements);
        EXPECT_EQ(matrix.width, 0);
        EXPECT_EQ(matrix.height, 0);
    }

    {
        const DR_FLOAT_TYPE array[] = {
            1
        };
        dr_matrix matrix = dr_matrix_create_from_array(array, 1, 1);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, array, 1, 1));
        dr_matrix_free(&matrix);
    }

    {
        const DR_FLOAT_TYPE array[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        dr_matrix matrix = dr_matrix_create_from_array(array, 3, 3);
        EXPECT_TRUE(dr_matrix_equals_to_array(matrix, array, 3, 3));
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, get_element) {
    dr_matrix matrix = dr_matrix_create_filled(2, 2, 0);
    const size_t size = matrix.width * matrix.height;
    for (size_t i = 0; i < size; ++i) {
        matrix.elements[i] = i;
    }
    EXPECT_EQ(dr_matrix_get_element(matrix, 0, 0), 0);
    EXPECT_EQ(dr_matrix_get_element(matrix, 1, 0), 1);
    EXPECT_EQ(dr_matrix_get_element(matrix, 0, 1), 2);
    EXPECT_EQ(dr_matrix_get_element(matrix, 1, 1), 3);
    dr_matrix_free(&matrix);
}

UTEST(dr_matrix, set_element) {
    dr_matrix matrix = dr_matrix_create_filled(2, 2, 0);
    dr_matrix_set_element(matrix, 0, 0, 0);
    dr_matrix_set_element(matrix, 1, 0, 1);
    dr_matrix_set_element(matrix, 0, 1, 2);
    dr_matrix_set_element(matrix, 1, 1, 3);
    EXPECT_EQ(dr_matrix_get_element(matrix, 0, 0), 0);
    EXPECT_EQ(dr_matrix_get_element(matrix, 1, 0), 1);
    EXPECT_EQ(dr_matrix_get_element(matrix, 0, 1), 2);
    EXPECT_EQ(dr_matrix_get_element(matrix, 1, 1), 3);
    dr_matrix_free(&matrix);
}

UTEST(dr_matrix, size) {
    {
        dr_matrix matrix = dr_matrix_create_empty();
        EXPECT_EQ(dr_matrix_size(matrix), 0);
    }

    {
        dr_matrix matrix = dr_matrix_create_filled(1, 1, 0);
        EXPECT_EQ(dr_matrix_size(matrix), 1);
        dr_matrix_free(&matrix);
    }

    {
        dr_matrix matrix = dr_matrix_create_filled(3, 5, 0);
        EXPECT_EQ(dr_matrix_size(matrix), 15);
        dr_matrix_free(&matrix);
    }
}

UTEST(dr_matrix, multiplication_write) {
    {
        const DR_FLOAT_TYPE left_arr[] = {
            0
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 1, 1);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_multiplication_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            5, 6,
            7, 8
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            19, 22,
            43, 50
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_multiplication_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            5,
            6
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            17,
            39
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 2);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_multiplication_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, multiplication_create) {
    {
        const DR_FLOAT_TYPE left_arr[] = {
            0
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 1, 1);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_multiplication_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            5, 6,
            7, 8
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            19, 22,
            43, 50
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_multiplication_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            5,
            6
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            17,
            39
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 2);
        dr_matrix result          = dr_matrix_multiplication_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, scale_write) {
    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            2
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_alloc(matrix.width, matrix.height);

        dr_matrix_scale_write(matrix, 2, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            3, 6,
            9, 12
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_alloc(matrix.width, matrix.height);

        dr_matrix_scale_write(matrix, 3, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4,
            5, 6
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            4, 8,
            12, 16,
            20, 24
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 3);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_alloc(matrix.width, matrix.height);

        dr_matrix_scale_write(matrix, 4, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, scale_create) {
    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            2
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_scale_create(matrix, 2);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            3, 6,
            9, 12
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_scale_create(matrix, 3);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4,
            5, 6
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            4, 8,
            12, 16,
            20, 24
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 3);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_scale_create(matrix, 4);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, subtraction_write) {
    {
        const DR_FLOAT_TYPE left_arr[] = {
            1
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 1, 1);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_subtraction_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 4,
            33, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            2, 6,
            7, 8
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            -1, -2,
            26, -4
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_subtraction_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            6, 11,
            -2, 0,
            4, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            -4, -12,
            7, 10,
            14, -20
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            10, 23,
            -9, -10,
            -10, 24
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 3);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 3);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_alloc(right.width, left.height);

        dr_matrix_subtraction_write(left, right, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, subtraction_create) {
    {
        const DR_FLOAT_TYPE left_arr[] = {
            1
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 1, 1);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_subtraction_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 4,
            33, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            2, 6,
            7, 8
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            -1, -2,
            26, -4
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 2);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_subtraction_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            6, 11,
            -2, 0,
            4, 4
        };

        const DR_FLOAT_TYPE right_arr[] = {
            -4, -12,
            7, 10,
            14, -20
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            10, 23,
            -9, -10,
            -10, 24
        };

        dr_matrix left            = dr_matrix_create_from_array(left_arr, 2, 3);
        dr_matrix right           = dr_matrix_create_from_array(right_arr, 2, 3);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_subtraction_create(left, right);

        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, transpose_write) {
    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            0
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_alloc(matrix.height, matrix.width);

        dr_matrix_transpose_write(matrix, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            1, 3,
            2, 4
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_alloc(matrix.height, matrix.width);

        dr_matrix_transpose_write(matrix, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2, -3,
            0, 1, -5
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            1, 0,
            2, 1,
            -3, -5
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 3, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_alloc(matrix.height, matrix.width);

        dr_matrix_transpose_write(matrix, result);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, transpose_create) {
    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            0
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            0
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 1, 1);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 1, 1);
        dr_matrix result          = dr_matrix_transpose_create(matrix);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2,
            3, 4
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            1, 3,
            2, 4
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 2, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 2);
        dr_matrix result          = dr_matrix_transpose_create(matrix);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE matrix_arr[] = {
            1, 2, -3,
            0, 1, -5
        };

        const DR_FLOAT_TYPE expected_result_arr[] = {
            1, 0,
            2, 1,
            -3, -5
        };

        dr_matrix matrix          = dr_matrix_create_from_array(matrix_arr, 3, 2);
        dr_matrix expected_result = dr_matrix_create_from_array(expected_result_arr, 2, 3);
        dr_matrix result          = dr_matrix_transpose_create(matrix);
        EXPECT_TRUE(dr_matrix_equals(result, expected_result));

        dr_matrix_free(&matrix);
        dr_matrix_free(&expected_result);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, copy_write) {
    {
        const DR_FLOAT_TYPE src_arr[] = {
            1
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 1, 1);
        dr_matrix result = dr_matrix_create_filled(1, 1, 0);
        dr_matrix_copy_write(src, result);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE src_arr[] = {
            1, 2,
            3, 4
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 2, 2);
        dr_matrix result = dr_matrix_create_filled(2, 2, 0);
        dr_matrix_copy_write(src, result);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE src_arr[] = {
            1, 2,
            3, 4,
            5, 6
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 2, 3);
        dr_matrix result = dr_matrix_create_filled(2, 3, 0);
        dr_matrix_copy_write(src, result);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, copy_create) {
    {
        const DR_FLOAT_TYPE src_arr[] = {
            1
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 1, 1);
        dr_matrix result = dr_matrix_copy_create(src);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE src_arr[] = {
            1, 2,
            3, 4
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 2, 2);
        dr_matrix result = dr_matrix_copy_create(src);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }

    {
        const DR_FLOAT_TYPE src_arr[] = {
            1, 2,
            3, 4,
            5, 6
        };

        dr_matrix src    = dr_matrix_create_from_array(src_arr, 2, 3);
        dr_matrix result = dr_matrix_copy_create(src);

        EXPECT_TRUE(dr_matrix_equals(result, src));

        dr_matrix_free(&src);
        dr_matrix_free(&result);
    }
}

UTEST(dr_matrix, equals_to_array) {
    {
        const DR_FLOAT_TYPE left_arr[] = {
            1
        };
        dr_matrix left_mat = dr_matrix_create_from_array(left_arr, 1, 1);
        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        EXPECT_TRUE(dr_matrix_equals_to_array(left_mat, right_arr, 1, 1));

        dr_matrix_free(&left_mat);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };
        dr_matrix left_mat = dr_matrix_create_from_array(left_arr, 2, 2);
        const DR_FLOAT_TYPE right_arr[] = {
            1
        };

        EXPECT_FALSE(dr_matrix_equals_to_array(left_mat, right_arr, 1, 1));

        dr_matrix_free(&left_mat);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1
        };
        dr_matrix left_mat = dr_matrix_create_from_array(left_arr, 1, 1);
        const DR_FLOAT_TYPE right_arr[] = {
            1, 2,
            3, 4
        };

        EXPECT_FALSE(dr_matrix_equals_to_array(left_mat, right_arr, 2, 2));

        dr_matrix_free(&left_mat);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            3, 4
        };
        dr_matrix left_mat = dr_matrix_create_from_array(left_arr, 2, 2);
        const DR_FLOAT_TYPE right_arr[] = {
            1, 2,
            3, 4
        };

        EXPECT_TRUE(dr_matrix_equals_to_array(left_mat, right_arr, 2, 2));

        dr_matrix_free(&left_mat);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2,
            2, 4
        };
        dr_matrix left_mat = dr_matrix_create_from_array(left_arr, 2, 2);
        const DR_FLOAT_TYPE right_arr[] = {
            1, 2,
            3, 4
        };

        EXPECT_FALSE(dr_matrix_equals_to_array(left_mat, right_arr, 2, 2));

        dr_matrix_free(&left_mat);
    }
}

UTEST(dr_matrix, equals) {
    {
        dr_matrix left  = dr_matrix_create_empty();
        dr_matrix right = dr_matrix_create_empty();

        EXPECT_TRUE(dr_matrix_equals(left, right));
    }

    {
        dr_matrix left  = dr_matrix_create_empty();
        dr_matrix right = dr_matrix_create_filled(1, 2, 0);

        EXPECT_FALSE(dr_matrix_equals(left, right));

        dr_matrix_free(&right);
    }

    {
        dr_matrix left  = dr_matrix_create_filled(3, 2, 0);
        dr_matrix right = dr_matrix_create_filled(2, 2, 0);

        EXPECT_FALSE(dr_matrix_equals(left, right));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        dr_matrix left  = dr_matrix_create_from_array(left_arr, 3, 3);
        dr_matrix right = dr_matrix_create_from_array(right_arr, 3, 3);

        EXPECT_TRUE(dr_matrix_equals(left, right));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
    }

    {
        const DR_FLOAT_TYPE left_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        const DR_FLOAT_TYPE right_arr[] = {
            1, 2, 3,
            4, 5, 6,
            7, 6, 9
        };

        dr_matrix left  = dr_matrix_create_from_array(left_arr, 3, 3);
        dr_matrix right = dr_matrix_create_from_array(right_arr, 3, 3);

        EXPECT_FALSE(dr_matrix_equals(left, right));

        dr_matrix_free(&left);
        dr_matrix_free(&right);
    }
}