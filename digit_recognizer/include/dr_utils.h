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

#ifndef DR_MALLOC
# define DR_MALLOC(sz) malloc(sz)
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

static size_t size_t_len(size_t number) {
    size_t len = 0;
    do {
        ++len;
        number /= 10;
    } while(number > 0);
    return len;
}

#endif // !DR_UTILS_H