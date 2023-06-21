#ifndef DR_TESTING_H
#define DR_TESTING_H

#include <utest.h>
#include <dr_matrix.h>

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

#endif // DR_TESTING_H