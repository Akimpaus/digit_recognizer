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

#ifndef DR_CALLOC
# define DR_CALLOC(n, sz) calloc(n, sz)
#endif

#ifndef DR_REALLOC
# define DR_REALLOC(ptr, sz) realloc(ptr, sz)
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

#endif // !DR_UTILS