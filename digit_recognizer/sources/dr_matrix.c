#include "dr_matrix.h"

bool dr_matrix_correct_sizes(const size_t width, const size_t height) {
    return (width > 0 && height > 0) || (width == 0 && height == 0);
}

void dr_matrix_assert_compat_elements_and_sizes(const dr_matrix matrix) {
    DR_ASSERT_MSG((matrix.elements && matrix.width > 0 && matrix.height > 0) ||
        (!matrix.elements && matrix.width == 0 && matrix.height == 0),
        "the matrix has sizes, but the pointer to the elements is null");
}

dr_matrix dr_matrix_alloc(const size_t width, const size_t height) {
    DR_ASSERT_MSG(dr_matrix_correct_sizes(width, height), "attempt to alloc a matrix with impossible sizes");
    dr_matrix matrix;
    if (width == 0 || height == 0) {
        matrix.elements = NULL;
    } else {
        matrix.elements = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * width * height);
        DR_ASSERT_MSG(matrix.elements, "alloc matrix error");
    }
    matrix.width  = width;
    matrix.height = height;
    return matrix;
}

void dr_matrix_unchecked_free(dr_matrix* matrix) {
    matrix->width  = 0;
    matrix->height = 0;
    if (!matrix->elements) {
        return;
    }
    DR_FREE(matrix->elements);
    matrix->elements = NULL;
}

void dr_matrix_free(dr_matrix* matrix) {
    DR_ASSERT_MSG(matrix, "attempt to free null matrix ptr");
    dr_matrix_assert_compat_elements_and_sizes(*matrix);
    dr_matrix_unchecked_free(matrix);
}

void dr_matrix_unchecked_fill(dr_matrix matrix, const DR_FLOAT_TYPE value) {
    const size_t matrix_size = dr_matrix_unchecked_size(matrix);
    for (size_t i = 0; i < matrix_size; ++i) {
        matrix.elements[i] = value;
    }
}

void dr_matrix_fill(dr_matrix matrix, const DR_FLOAT_TYPE value) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_fill(matrix, value);
}

void dr_matrix_unchecked_fill_random(dr_matrix matrix, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    const size_t matrix_size = dr_matrix_unchecked_size(matrix);
    for (size_t i = 0; i < matrix_size; ++i) {
        matrix.elements[i] = dr_random_float(min, max);
    }
}

void dr_matrix_fill_random(dr_matrix matrix, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_fill_random(matrix, min, max);
}

void dr_matrix_unchecked_copy_to_array(const dr_matrix matrix, DR_FLOAT_TYPE* array) {
    const size_t size = dr_matrix_unchecked_size(matrix);
    for (size_t i = 0; i < size; ++i) {
        array[i] = matrix.elements[i];
    }
}

void dr_matrix_copy_to_array(const dr_matrix matrix, DR_FLOAT_TYPE* array) {
    DR_ASSERT_MSG(array, "attempt to copy a matrix to a NULL array");
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_copy_to_array(matrix, array);
}

void dr_matrix_unchecked_copy_array(dr_matrix matrix, const DR_FLOAT_TYPE* array) {
    const size_t size = dr_matrix_unchecked_size(matrix);
    for (size_t i = 0; i < size; ++i) {
        matrix.elements[i] = array[i];
    }
}

void dr_matrix_copy_array(dr_matrix matrix, const DR_FLOAT_TYPE* array) {
    DR_ASSERT_MSG(array, "attempt to copy a NULL array to matrix");
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_copy_array(matrix, array);
}

void dr_matrix_unchecked_copy_write(const dr_matrix src_matrix, dr_matrix dst_matrix) {
    dr_matrix_unchecked_copy_array(dst_matrix, src_matrix.elements);
}

void dr_matrix_copy_write(const dr_matrix src_matrix, dr_matrix dst_matrix) {
    DR_ASSERT_MSG(src_matrix.width == dst_matrix.width && src_matrix.height == dst_matrix.height,
        "attempt to copy a matrix into a matrix of a different size");
    dr_matrix_assert_compat_elements_and_sizes(src_matrix);
    dr_matrix_assert_compat_elements_and_sizes(dst_matrix);
    dr_matrix_unchecked_copy_write(src_matrix, dst_matrix);
}

dr_matrix dr_matrix_unchecked_copy_create(const dr_matrix matrix) {
    dr_matrix result = dr_matrix_alloc(matrix.width, matrix.height);
    dr_matrix_unchecked_copy_write(matrix, result);
    return result;
}

dr_matrix dr_matrix_copy_create(const dr_matrix matrix) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_copy_create(matrix);
}

dr_matrix dr_matrix_create_empty() {
    dr_matrix matrix;
    matrix.elements = NULL;
    matrix.width    = 0;
    matrix.height   = 0;
    return matrix;
}

dr_matrix dr_matrix_unchecked_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value) {
    dr_matrix matrix = dr_matrix_alloc(width, height);
    dr_matrix_unchecked_fill(matrix, value);
    return matrix;
}

dr_matrix dr_matrix_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value) {
    if (width * height == 0) {
        return dr_matrix_create_empty();
    }
    return dr_matrix_unchecked_create_filled(width, height, value);
}

dr_matrix dr_matrix_create_from_array(const DR_FLOAT_TYPE* array, const size_t width, const size_t height) {
    if (width * height == 0) {
        return dr_matrix_create_empty();
    }
    DR_ASSERT_MSG(array, "attempt to create a matrix from a NULL array with positive sizes");
    dr_matrix result  = dr_matrix_alloc(width, height);
    dr_matrix_unchecked_copy_array(result, array);
    return result;
}

DR_FLOAT_TYPE dr_matrix_unchecked_get_element(const dr_matrix matrix, const size_t column, const size_t row) {
    return matrix.elements[row * matrix.width + column];
}

DR_FLOAT_TYPE dr_matrix_get_element(const dr_matrix matrix, const size_t column, const size_t row) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_get_element(matrix, column, row);
}

void dr_matrix_unchecked_set_element(
    dr_matrix matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value) {
    matrix.elements[row * matrix.width + column] = value;
}

void dr_matrix_set_element(dr_matrix matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_set_element(matrix, column, row, value);
}

size_t dr_matrix_unchecked_size(const dr_matrix matrix) {
    return matrix.width * matrix.height;
}

size_t dr_matrix_size(const dr_matrix matrix) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_size(matrix);
}

void dr_matrix_unchecked_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    for (size_t i = 0; i < result.width; ++i) {
        for (size_t j = 0; j < result.height; ++j) {
            DR_FLOAT_TYPE sum = 0;
            for (size_t k = 0; k < left.width; ++k) {
                const DR_FLOAT_TYPE left_val  = dr_matrix_unchecked_get_element(left, k, j);
                const DR_FLOAT_TYPE right_val = dr_matrix_unchecked_get_element(right, i, k);
                sum += left_val * right_val;
            }
            dr_matrix_unchecked_set_element(result, i, j, sum);
        }
    }
}

void dr_matrix_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    DR_ASSERT_MSG(result.elements,
        "attempt to write the result of matrix multiplication into a matrix with NULL elements");
    DR_ASSERT_MSG(result.width == right.width && result.height == left.height,
        "it is impossible to write the result of matrix multiplication: "
        "the width of the resulting matrix should be as follows: width - right.width, height - left.height");
    DR_ASSERT_MSG(left.width == right.height, "when multiplying the matrix, the number of columns of the left matrix "
        "should be equal to the number of rows of the right matrix");
    dr_matrix_assert_compat_elements_and_sizes(left);
    dr_matrix_assert_compat_elements_and_sizes(right);
    dr_matrix_unchecked_multiplication_write(left, right, result);
}

dr_matrix dr_matrix_unchecked_multiplication_create(const dr_matrix left, const dr_matrix right) {
    dr_matrix result = dr_matrix_alloc(right.width, left.height);
    dr_matrix_unchecked_multiplication_write(left, right, result);
    return result;
}

dr_matrix dr_matrix_multiplication_create(const dr_matrix left, const dr_matrix right) {
    DR_ASSERT_MSG(left.width == right.height, "when multiplying the matrix, the number of columns of the left matrix "
        "should be equal to the number of rows of the right matrix");
    dr_matrix_assert_compat_elements_and_sizes(left);
    dr_matrix_assert_compat_elements_and_sizes(right);
    return dr_matrix_unchecked_multiplication_create(left, right);
}

void dr_matrix_unchecked_rows_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    for (size_t row = 0; row < result.height; ++row) {
        DR_FLOAT_TYPE left_val = 1;
        for (size_t column = 0; column < left.width; ++column) {
            left_val *= dr_matrix_unchecked_get_element(left, column, row);
        }
        DR_FLOAT_TYPE right_val = 1;
        for (size_t column = 0; column < right.width; ++column) {
            right_val *= dr_matrix_unchecked_get_element(right, column, row);
        }
        result.elements[row] = left_val * right_val;
    }
}

void dr_matrix_rows_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    DR_ASSERT_MSG(left.height == right.height && right.height == result.height,
        "attempt to matrix rows multiplication with different number of rows");
    DR_ASSERT_MSG(result.width == 1,
        "when performing matrix rows multiplication, the width of the resulting matrix must be one.");
    dr_matrix_assert_compat_elements_and_sizes(left);
    dr_matrix_assert_compat_elements_and_sizes(right);
    dr_matrix_assert_compat_elements_and_sizes(result);
    dr_matrix_unchecked_rows_multiplication_write(left, right, result);
}

dr_matrix dr_matrix_unchecked_rows_multiplication_create(const dr_matrix left, const dr_matrix right) {
    dr_matrix result = dr_matrix_alloc(1, left.height);
    dr_matrix_unchecked_rows_multiplication_write(left, right, result);
    return result;
}

dr_matrix dr_matrix_rows_multiplication_create(const dr_matrix left, const dr_matrix right) {
    DR_ASSERT_MSG(left.height == right.height, "attempt to matrix rows multiplication with different number of rows");
    dr_matrix_assert_compat_elements_and_sizes(left);
    dr_matrix_assert_compat_elements_and_sizes(right);
    return dr_matrix_unchecked_rows_multiplication_create(left, right);
}

void dr_matrix_unchecked_scale_write(const dr_matrix matrix, const DR_FLOAT_TYPE value, dr_matrix result) {
    const size_t size = dr_matrix_unchecked_size(matrix);
    for (size_t i = 0; i < size; ++i) {
        result.elements[i] = matrix.elements[i] * value;
    }
}

void dr_matrix_scale_write(const dr_matrix matrix, const DR_FLOAT_TYPE value, dr_matrix result) {
    DR_ASSERT_MSG(result.width == matrix.width && result.height == matrix.height,
        "an attempt to write the result of matrix scale to a matrix with a different size");
    DR_ASSERT_MSG(matrix.elements && result.elements, "attempt to scale an invalid matrices");
    dr_matrix_unchecked_scale_write(matrix, value, result);
}

dr_matrix dr_matrix_unchecked_scale_create(const dr_matrix matrix, const DR_FLOAT_TYPE value) {
    dr_matrix result = dr_matrix_alloc(matrix.width, matrix.height);
    dr_matrix_unchecked_scale_write(matrix, value, result);
    return result;
}

dr_matrix dr_matrix_scale_create(const dr_matrix matrix, const DR_FLOAT_TYPE value) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_scale_create(matrix, value);
}

void dr_matrix_unchecked_subtraction_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    const size_t size = dr_matrix_unchecked_size(result);
    for (size_t i = 0; i < size; ++i) {
        result.elements[i] = left.elements[i] - right.elements[i];
    }
}

void dr_matrix_subtraction_write(const dr_matrix left, const dr_matrix right, dr_matrix result) {
    DR_ASSERT_MSG(left.width == right.width && left.height == right.height,
        "attempt to subtract matrices of different sizes");
    DR_ASSERT_MSG(result.width == left.width && result.height == left.height,
        "attempt to write the result of matrices to a matrix with a different sizes");
    if (dr_matrix_size(result) == 0) {
        return;
    }
    DR_ASSERT_MSG(left.elements && right.elements && result.elements,
        "attempt at subtraction with invalid matrices");
    dr_matrix_unchecked_subtraction_write(left, right, result);
}

dr_matrix dr_matrix_unchecked_subtraction_create(const dr_matrix left, const dr_matrix right) {
    dr_matrix result = dr_matrix_alloc(left.width, left.height);
    dr_matrix_unchecked_subtraction_write(left, right, result);
    return result;
}

dr_matrix dr_matrix_subtraction_create(const dr_matrix left, const dr_matrix right) {
    DR_ASSERT_MSG(left.width == right.width && left.height == right.height,
        "attempt to subtract matrices of different sizes");
    DR_ASSERT_MSG(left.elements && right.elements, "attempt at subtraction with invalid matrices");
    return dr_matrix_unchecked_subtraction_create(left, right);
}

void dr_matrix_unchecked_transpose_write(const dr_matrix matrix, dr_matrix result) {
    for (size_t row = 0; row < matrix.height; ++row) {
        for (size_t column = 0; column < matrix.width; ++column) {
            const DR_FLOAT_TYPE value = dr_matrix_unchecked_get_element(matrix, column, row);
            dr_matrix_unchecked_set_element(result, row, column, value);
        }
    }
}

void dr_matrix_transpose_write(const dr_matrix matrix, dr_matrix result) {
    DR_ASSERT_MSG(result.elements,
        "attempt to write the result of matrix transpose into a matrix with NULL elements");
    DR_ASSERT_MSG(result.width == matrix.height && result.height == matrix.width,
        "it is impossible to write the result of matrix transpose: "
        "the width of the resulting matrix should be as follows: width - matrix.height, height - matrix.width");
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_matrix_unchecked_transpose_write(matrix, result);
}

dr_matrix dr_matrix_unchecked_transpose_create(const dr_matrix matrix) {
    dr_matrix result = dr_matrix_alloc(matrix.height, matrix.width);
    dr_matrix_unchecked_transpose_write(matrix, result);
    return result;
}

dr_matrix dr_matrix_transpose_create(const dr_matrix matrix) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_transpose_create(matrix);
}

bool dr_matrix_unchecked_equals_to_array(
    const dr_matrix matrix, const DR_FLOAT_TYPE* array, const size_t width, const size_t height) {
    if (matrix.width != width || matrix.height != height) {
        return false;
    }
    const size_t size = width * height;
    for (size_t i = 0; i < size; ++i) {
        if (matrix.elements[i] != array[i]) {
            return false;
        }
    }
    return true;
}

bool dr_matrix_equals_to_array(
    const dr_matrix matrix, const DR_FLOAT_TYPE* array, const size_t width, const size_t height) {
    DR_ASSERT_MSG(array && width * height > 0, "comparison of a matrix with an array is possible only if "
        "the array is not NULL and its size is specified as positive.");
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    return dr_matrix_unchecked_equals_to_array(matrix, array, width, height);
}

bool dr_matrix_unchecked_equals(const dr_matrix left, const dr_matrix right) {
    return dr_matrix_unchecked_equals_to_array(left, right.elements, right.width, right.height);
}

bool dr_matrix_equals(const dr_matrix left, const dr_matrix right) {
    dr_matrix_assert_compat_elements_and_sizes(left);
    dr_matrix_assert_compat_elements_and_sizes(right);
    return dr_matrix_unchecked_equals(left, right);
}

void dr_matrix_print_space(
    const dr_matrix matrix, const size_t space_open, const size_t space_data, const size_t space_close) {
    dr_matrix_assert_compat_elements_and_sizes(matrix);
    dr_print_spaces(space_open);
    printf("%s\n", "[");
    for (size_t i = 0; i < matrix.height; ++i) {
        for (size_t j = 0; j < matrix.width; ++j) {
            dr_print_spaces(space_data);
            printf("%10.3f ", dr_matrix_get_element(matrix, j, i));
        }
        printf("\n");
    }
    dr_print_spaces(space_close);
    printf("%s\n", "]");
}

void dr_matrix_print_name_space(const dr_matrix matrix, const char* name,
    const size_t space_open, const size_t space_data, const size_t space_close) {
    dr_print_spaces(space_open);
    printf("%s: ", name);
    dr_matrix_print_space(matrix, 0, space_data, space_close);
}

void dr_matrix_print(const dr_matrix matrix) {
    dr_matrix_print_space(matrix, 0, 0, 0);
}

void dr_matrix_print_name(const dr_matrix matrix, const char* name) {
    printf("%s: ", name);
    dr_matrix_print(matrix);
}