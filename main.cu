#include <stdio.h>

#include "print.h"
#include "types.h"
#include "radix.h"

#define THREADS 128
// multiple of 96 for 1050ti says nvvp
#define BLOCKS (96 * 2)
// size of unsorted array
#define SIZE 250000000 // 2Gbs worth of elems
// #define SIZE 10
// #define PRINT

#define MAX_VALUE 1024

int main(void)
{

    // check for GPUs
    int gpus;
    cudaGetDeviceCount(&gpus);
    if (gpus < 1) {
        fprintf(stderr, "Need at least one GPU\n");
        exit(0);
    }

    // print device info (assuming exactly one device)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
    printf("\tDevice name: %s\n", prop.name);
    printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("\tPeak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*((float)prop.memoryBusWidth/8)/1.0e6);

    elem *unsorted = NULL;
    elem *sorted = NULL;
    elem *sorted2 = NULL;
    size_t size = SIZE;
    int threads = THREADS;
    int blocks = BLOCKS;

    printf("Allocating memory for unsorted.\n");
    unsorted = (elem *)malloc(size * sizeof(elem));
    if (unsorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    printf("Generating input array.\n");
    for (size_t i = 0; i < size; ++i) {
        // unsorted[i] = rand() % INT32_MAX;
        // unsorted[i] = rand() % (1 * KEY_MAX_VALUE) + KEY_MAX_VALUE;
        // unsorted[i] = rand() % 10;
        unsorted[i] = rand() % MAX_VALUE;
        // unsorted[i] = rand() % 10000;
        // unsorted[i] = rand() % KEY_MAX_VALUE;
        // unsorted[i] = size - i;
    }
    unsorted[3] = 0;

    #ifdef PRINT
    print_array(unsorted, size, "unsorted");
    #endif

    // sorted = radix_sort(unsorted, size, threads, blocks);
    sorted = counting_sort(unsorted, size, threads, blocks, MAX_VALUE);

    #ifdef PRINT
    print_array(sorted, size, "sorted");
    #endif

    // sorted2 = radix_sort(unsorted, size, threads, blocks);
    // #ifdef PRINT
    // print_array(sorted2, size, "sorted");
    // #endif
    //
    // print_compare_array((uint*)sorted, (uint*)sorted2, size);
}
