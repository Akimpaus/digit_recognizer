#ifndef DR_UTILS_H
#define DR_UTILS_H

// TODO: add checks (where it's no) on DR_MALLOC. after it assert on not NULL.

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define DR_FLOAT_TYPE float

static inline void* dr_malloc(const size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "%s %zu %s\n", "Error when trying to allocate memory for", size, "bytes");
        exit(-1);
    }
    return ptr;
}

#ifndef DR_MALLOC
# define DR_MALLOC(size) dr_malloc(size)
#endif

#ifndef DR_FREE
# define DR_FREE(ptr) free(ptr)
#endif

#ifndef DR_ASSERT
# define DR_ASSERT(cond) assert(cond)
#endif

#ifndef DR_ASSERT_MSG
# define DR_ASSERT_MSG(cond, msg) assert((cond && msg))
#endif

#define DR_ARRAY_LENGTH(array) (sizeof(array) / sizeof(*array))

static inline size_t dr_size_t_len(size_t number) {
    size_t len = 0;
    do {
        ++len;
        number /= 10;
    } while(number > 0);
    return len;
}

static inline DR_FLOAT_TYPE dr_random_float(DR_FLOAT_TYPE min, DR_FLOAT_TYPE max) {
    if (min > max) {
        const DR_FLOAT_TYPE temp_min = min;
        min = max;
        max = temp_min;
    }
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

static inline void dr_print_spaces(const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        printf(" ");
    }
}

static inline DR_FLOAT_TYPE** dr_array_2d_float_alloc(const size_t width, const size_t height) {
    DR_FLOAT_TYPE** array = (DR_FLOAT_TYPE**)DR_MALLOC(sizeof(DR_FLOAT_TYPE*) * height);
    for (size_t i = 0; i < height; ++i) {
        array[i] = (DR_FLOAT_TYPE*)DR_MALLOC(sizeof(DR_FLOAT_TYPE) * width);
    }
    return array;
}

static inline void dr_array_2d_float_free(DR_FLOAT_TYPE** array, const size_t height) {
    for (size_t i = 0; i < height; ++i) {
        DR_FREE(array[i]);
    }
    DR_FREE(array);
}

static inline char* dr_str_alloc(const char* str) {
    char* new_str = DR_MALLOC(sizeof(char) * strlen(str));
    if (!new_str) {
        return NULL;
    }
    strcpy(new_str, str);
    return new_str;
}

#endif // !DR_UTILS_H