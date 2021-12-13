#ifndef H_PRINT_ARRAY
#define H_PRINT_ARRAY

#include <unistd.h>
#include <stdint.h>

#ifdef __CUDA_ARCH__
__device__
#else
__host__
#endif
void print_array(int *, size_t, const char *);

#ifdef __CUDA_ARCH__
__device__
#else
__host__
#endif
void print_array(uint *, size_t, const char *);

#endif
