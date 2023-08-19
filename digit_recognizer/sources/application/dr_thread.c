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

dr_mutex_t dr_mutex_create() {
#ifdef _WIN32
    return CreateMutex(NULL, FALSE, NULL);
#else
    const dr_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    return mutex;
#endif // _WIN32
}

bool dr_check_mutex(const dr_mutex_t mutex) {
#ifdef _WIN32
    return mutex != NULL;
#else
    return true;
#endif // _WIN32
}

bool dr_mutex_lock(dr_mutex_t* mutex) {
#ifdef _WIN32
    return WaitForSingleObject(*mutex, INFINITE) == WAIT_OBJECT_0;
#else
    return pthread_mutex_lock(mutex) == 0;
#endif // _WIN32
}

bool dr_mutex_unlock(dr_mutex_t* mutex) {
#ifdef _WIN32
    return ReleaseMutex(*mutex);
#else
    return pthread_mutex_unlock(mutex) == 0;
#endif // _WIN32
}

bool dr_mutex_close(dr_mutex_t mutex) {
#ifdef _WIN32
    return CloseHandle(mutex);
#else
    return true;
#endif // _WIN32
}
