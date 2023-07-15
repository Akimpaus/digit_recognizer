#ifndef DR_TESTING_H
#define DR_TESTING_H

#include <utest.h>
#include <neural_network/dr_matrix.h>

static bool dr_testing_matrix_filled(const dr_matrix matrix, const DR_FLOAT_TYPE val) {
    const size_t size = matrix.width * matrix.height;
    for (size_t i = 0; i < size; ++i) {
        if (matrix.elements[i] != val) {
            return false;
        }
    }
    return true;
}

static bool dr_testing_matrix_filled_random(const dr_matrix matrix, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max) {
    const size_t size = matrix.width * matrix.height;
    for (size_t i = 0; i < size; ++i) {
        const DR_FLOAT_TYPE val = matrix.elements[i];
        if (val < min || val > max) {
            return false;
        }
    }
    return true;
}

static bool dr_testing_matrix_array_equals(const size_t size, const DR_FLOAT_TYPE* left, const DR_FLOAT_TYPE* right) {
    if (left == right) {
        return true;
    }
    if ((left && !right) || (!left && right)) {
        return false;
    }
    for (size_t i = 0; i < size; ++i) {
        if (left[i] != right[i]) {
            return false;
        }
    }
    return true;
}

#endif // DR_TESTING_H