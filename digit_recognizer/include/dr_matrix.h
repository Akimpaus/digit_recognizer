#ifndef DR_MATRIX_H
#define DR_MATRIX_H

#include <stdio.h>
#include <stdbool.h>
#include "dr_utils.h"

typedef struct {
    DR_FLOAT_TYPE* elements;
    size_t width;
    size_t height;
} dr_matrix;

bool dr_matrix_correct_sizes(const size_t width, const size_t height);

void dr_matrix_assert_compat_elements_and_sizes(const dr_matrix matrix);

dr_matrix dr_matrix_alloc(const size_t width, const size_t height);

void dr_matrix_unchecked_free(dr_matrix* matrix);

void dr_matrix_free(dr_matrix* matrix);

void dr_matrix_unchecked_fill(dr_matrix* matrix, const DR_FLOAT_TYPE value);

void dr_matrix_fill(dr_matrix* matrix, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_unchecked_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_create_empty();

dr_matrix dr_matrix_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_create_from_array(const DR_FLOAT_TYPE* array, const size_t width, const size_t height);

DR_FLOAT_TYPE dr_matrix_unchecked_get_element(const dr_matrix matrix, const size_t column, const size_t row);

DR_FLOAT_TYPE dr_matrix_get_element(const dr_matrix matrix, const size_t column, const size_t row);

void dr_matrix_unchecked_set_element(
    dr_matrix* matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value);

void dr_matrix_set_element(dr_matrix* matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value); 

size_t dr_matrix_unchecked_size(const dr_matrix matrix);

size_t dr_matrix_size(const dr_matrix matrix);

void dr_matrix_unchecked_multiplication(const dr_matrix left, const dr_matrix right, dr_matrix* result);

void dr_matrix_multiplication(const dr_matrix left, const dr_matrix right, dr_matrix* result);

bool dr_matrix_unchecked_equals(const dr_matrix left, const dr_matrix right);

bool dr_matrix_unchecked_equals_to_array(
    const dr_matrix matrix, const size_t width, const size_t height, const DR_FLOAT_TYPE* array);

bool dr_matrix_equals_to_array(
    const dr_matrix matrix, const size_t width, const size_t height, const DR_FLOAT_TYPE* array);

bool dr_matrix_equals(const dr_matrix left, const dr_matrix right);

void dr_matrix_print(const dr_matrix matrix);

void dr_matrix_print_name(const dr_matrix matrix, const char* name);

#endif // DR_MATRIX_H