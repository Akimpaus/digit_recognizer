#ifndef DR_UTILS_H
#define DR_UTILS_H

#include <stdlib.h>
#include <assert.h>

#define DR_FLOAT_TYPE float

#if defined(__cplusplus)
# define DR_CLITERAL(type) type
#else
# define DR_CLITERAL(type) (type)
#endif

static void* dr_malloc(const size_t size) {
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

static size_t dr_size_t_len(size_t number) {
    size_t len = 0;
    do {
        ++len;
        number /= 10;
    } while(number > 0);
    return len;
}

static DR_FLOAT_TYPE dr_random_float(DR_FLOAT_TYPE min, DR_FLOAT_TYPE max) {
    if (min > max) {
        const DR_FLOAT_TYPE temp_min = min;
        min = max;
        max = temp_min;
    }
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

static void dr_print_spaces(const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        printf(" ");
    }
}

#endif // !DR_UTILS_H