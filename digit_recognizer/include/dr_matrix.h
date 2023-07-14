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

void dr_matrix_unchecked_fill(dr_matrix matrix, const DR_FLOAT_TYPE value);

void dr_matrix_fill(dr_matrix matrix, const DR_FLOAT_TYPE value);

void dr_matrix_unchecked_fill_random(dr_matrix matrix, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max);

void dr_matrix_fill_random(dr_matrix matrix, const DR_FLOAT_TYPE min, const DR_FLOAT_TYPE max);

void dr_matrix_unchecked_copy_to_array(const dr_matrix matrix, DR_FLOAT_TYPE* array);

void dr_matrix_copy_to_array(const dr_matrix matrix, DR_FLOAT_TYPE* array);

void dr_matrix_unchecked_copy_array(dr_matrix matrix, const DR_FLOAT_TYPE* array);

void dr_matrix_copy_array(dr_matrix matrix, const DR_FLOAT_TYPE* array);

void dr_matrix_unchecked_copy_write(const dr_matrix src_matrix, dr_matrix dst_matrix);

void dr_matrix_copy_write(const dr_matrix src_matrix, dr_matrix dst_matrix);

dr_matrix dr_matrix_unchecked_copy_create(const dr_matrix matrix);

dr_matrix dr_matrix_copy_create(const dr_matrix matrix);

dr_matrix dr_matrix_unchecked_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_create_empty();

dr_matrix dr_matrix_create_filled(const size_t width, const size_t height, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_create_from_array(const DR_FLOAT_TYPE* array, const size_t width, const size_t height);

DR_FLOAT_TYPE dr_matrix_unchecked_get_element(const dr_matrix matrix, const size_t column, const size_t row);

DR_FLOAT_TYPE dr_matrix_get_element(const dr_matrix matrix, const size_t column, const size_t row);

void dr_matrix_unchecked_set_element(
    dr_matrix matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value);

void dr_matrix_set_element(dr_matrix matrix, const size_t column, const size_t row, const DR_FLOAT_TYPE value); 

size_t dr_matrix_unchecked_size(const dr_matrix matrix);

size_t dr_matrix_size(const dr_matrix matrix);

void dr_matrix_unchecked_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

void dr_matrix_multiplication_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

dr_matrix dr_matrix_unchecked_multiplication_create(const dr_matrix left, const dr_matrix right);

dr_matrix dr_matrix_multiplication_create(const dr_matrix left, const dr_matrix right);

void dr_matrix_unchecked_dot_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

void dr_matrix_dot_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

dr_matrix dr_matrix_unchecked_dot_create(const dr_matrix left, const dr_matrix right);

dr_matrix dr_matrix_dot_create(const dr_matrix left, const dr_matrix right);

void dr_matrix_unchecked_scale_write(const dr_matrix matrix, const DR_FLOAT_TYPE value, dr_matrix result);

void dr_matrix_scale_write(const dr_matrix matrix, const DR_FLOAT_TYPE value, dr_matrix result);

dr_matrix dr_matrix_unchecked_scale_create(const dr_matrix matrix, const DR_FLOAT_TYPE value);

dr_matrix dr_matrix_scale_create(const dr_matrix matrix, const DR_FLOAT_TYPE value);

void dr_matrix_unchecked_subtraction_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

void dr_matrix_subtraction_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

dr_matrix dr_matrix_unchecked_subtraction_create(const dr_matrix left, const dr_matrix right);

dr_matrix dr_matrix_subtraction_create(const dr_matrix left, const dr_matrix right);

void dr_matrix_unchecked_addition_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

void dr_matrix_addition_write(const dr_matrix left, const dr_matrix right, dr_matrix result);

dr_matrix dr_matrix_unchecked_addition_create(const dr_matrix left, const dr_matrix right);

dr_matrix dr_matrix_addition_create(const dr_matrix left, const dr_matrix right);

void dr_matrix_unchecked_transpose_write(const dr_matrix matrix, dr_matrix result);

void dr_matrix_transpose_write(const dr_matrix matrix, dr_matrix result);

dr_matrix dr_matrix_unchecked_transpose_create(const dr_matrix matrix);

dr_matrix dr_matrix_transpose_create(const dr_matrix matrix);

bool dr_matrix_unchecked_equals(const dr_matrix left, const dr_matrix right);

bool dr_matrix_unchecked_equals_to_array(
    const dr_matrix matrix, const DR_FLOAT_TYPE* array, const size_t width, const size_t height);

bool dr_matrix_equals_to_array(
    const dr_matrix matrix, const DR_FLOAT_TYPE* array, const size_t width, const size_t height);

bool dr_matrix_equals(const dr_matrix left, const dr_matrix right);

void dr_matrix_print_space(
    const dr_matrix matrix, const size_t space_open, const size_t space_data, const size_t space_close);

void dr_matrix_print_name_space(const dr_matrix matrix, const char* name,
    const size_t space_open, const size_t space_data, const size_t space_close);

void dr_matrix_print(const dr_matrix matrix);

void dr_matrix_print_name(const dr_matrix matrix, const char* name);

#endif // DR_MATRIX_H