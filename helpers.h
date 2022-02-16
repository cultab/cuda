#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <stdio.h>

/*
 * Macros
 */

inline void cudaPrintError(cudaError_t cudaerr, const char *file, int line)
{
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(cudaerr), file, line);
        exit(cudaerr);
    }
}

#define cudaErr(ans)                                                                                                   \
    do {                                                                                                               \
        cudaPrintError((ans), __FILE__, __LINE__);                                                                     \
    } while (0)

#define cudaLastErr()                                                                                                  \
    do {                                                                                                               \
        cudaError_t cudaerr = cudaDeviceSynchronize();                                                                 \
        cudaPrintError(cudaerr, __FILE__, __LINE__);                                                                   \
    } while (0)


/*
 * Stop clangd from complaining of missing symbols in .cu files
 */

#ifdef __CUDA_ARCH__
#define syncthreads() __syncthreads()
#else
#define syncthreads()
#endif

__global__ void zero_array(unsigned int *, size_t);

#endif
