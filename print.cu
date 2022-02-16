#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "print.h"
#include "types.h"

// define restrict because it's not standard c++

/*
 * Print array, compiled for both host and device.
 */

__device__
__host__
void print_array(int *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}


__device__
__host__
void print_array(unsigned int *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%u ", arr[i]);
    }
    printf("\n");
}

void print_compare_array(unsigned int *a, unsigned int *b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i])
            printf("%4u | %4u\n", a[i], b[i]);
    }
}

void print_array_bits(int *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("[%3ld]=", i);
        print_bits(arr[i]);
    }
    printf("\n");
}

void print_array_bits(unsigned int *arr, size_t size, const char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("[%3ld]=", i);
        print_bits(arr[i]);
    }
    printf("\n");
}
void print_bits(int num) {
    unsigned int size = sizeof(unsigned int);
    unsigned int maxPow = 1<<(size*8-1);
    for (size_t i = 0; i < size * 8; ++i) {
        // print last bit and shift left.
        printf("%u ",num&maxPow ? 1 : 0);
        num = num<<1;
        if ((i + 1) % 8 == 0) {
            printf("| ");
        }
    }
    printf("\n");
}
