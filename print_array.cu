#include <stdio.h>

/*
 * Print array, compiled for both host and device.
 */

#ifdef __CUDA_ARCH__
__device__
#else
__host__
#endif
void print_array(int *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}


#ifdef __CUDA_ARCH__
__device__
#else
__host__
#endif
void print_array(uint *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%u ", arr[i]);
    }
    printf("\n");
}
