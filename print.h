#ifndef H_PRINT
#define H_PRINT

#include <unistd.h>
#include <stdint.h>
#include "types.h"

#ifdef __CUDA_ARCH__
__device__
#endif
void print_array(int *, size_t, const char *);

#ifdef __CUDA_ARCH__
__device__
#endif
void print_array(uint *, size_t, const char *);

void print_bits(int);
void print_array_bits(int *, size_t, const char *);
void print_array_bits(uint *, size_t, const char *);
void print_compare_array(uint *, unsigned int *, size_t);

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

