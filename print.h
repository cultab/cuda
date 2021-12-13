#ifndef H_PRINT
#define H_PRINT

#include <stdint.h>
#include <unistd.h>

#ifdef __CUDA_ARCH__
__device__
#endif
void print_array(int *, size_t, const char *);

#ifdef __CUDA_ARCH__
__device__
#endif
void print_array(uint *, size_t, const char *);

/*
 * Debug print, only prints if DEBUG is defined
 */

#ifdef DEBUG
#define debug(...) \
    printf(__VA_ARGS__);
#else
#define debug(...)
#endif

#endif
