#include "dr_matrix.h"

int main() {
    dr_matrix matrix = dr_matrix_create_filled(10, 10, 10);
    dr_matrix_set_element(&matrix, 3, 4, -3);
    dr_matrix_print_name(matrix, "test");
    dr_matrix_free(&matrix);
    return 0;
}