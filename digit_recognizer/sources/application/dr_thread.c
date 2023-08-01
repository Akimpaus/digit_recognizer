#include <application/dr_thread.h>

#ifdef _WIN32
  #include <Windows.h>
#endif // _WIN32

dr_thread_handle_t dr_thread_create(dr_thread_id_t* thread_id, dr_thread_function_t thread_function) {
#ifdef _WIN32
    return CreateThread(NULL, 0, thread_function, NULL, 0, thread_id);
#else
    return pthread_create(thread_id, NULL, thread_function, NULL);
#endif // _WIN32
}

bool dr_check_thread_handle(const dr_thread_handle_t thread_handle) {
#ifdef _WIN32
    return thread_handle != NULL;
#else
    return thread_handle == 0;
#endif // _WIN32
}

bool dr_thread_join(dr_thread_handle_t thread_handle, dr_thread_id_t thread_id) {
#ifdef _WIN32
    return WaitForSingleObject(thread_handle, INFINITE) != (uint32_t)0xFFFFFFFF;
#else
    return pthread_join(thread_id, NULL) == 0;
#endif // _WIN32
}

bool dr_thread_close(dr_thread_handle_t thread_handle) {
#ifdef _WIN32
    return CloseHandle(thread_handle);
#else
    return true;
#endif // _WIN32
}