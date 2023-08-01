#ifndef DR_THREAD_H
#define DR_THREAD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
  typedef uint32_t dr_thread_id_t;
  typedef uint32_t dr_thread_function_result_t;
  typedef void*    dr_thread_handle_t;
#else
  #include <pthread.h>
  typedef pthread_t dr_thread_id_t;
  typedef void*     dr_thread_function_result_t;
  typedef int       dr_thread_handle_t;
#endif // _WIN32

#ifdef _WIN32
  #define DR_WINAPI __stdcall
#else
  #define DR_WINAPI
#endif // _WIN32

typedef dr_thread_function_result_t(DR_WINAPI *dr_thread_function_t)(void*);

dr_thread_handle_t dr_thread_create(dr_thread_id_t* thread_id, dr_thread_function_t thread_function);

bool dr_check_thread_handle(const dr_thread_handle_t thread_handle);

bool dr_thread_join(dr_thread_handle_t thread_handle, dr_thread_id_t thread_id);

bool dr_thread_close(dr_thread_handle_t thread_handle);

#endif // DR_THREAD_H