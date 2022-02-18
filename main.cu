#include <stdio.h>
#include <time.h>

#include "print.h"
#include "types.h"
#include "counting.h"
#include "bitonic.h"
#include "radix.h"

// #define PRINT

/*
 * Default values
 */

static int THREADS = 256;

// multiple of 96 for 1050ti says nvvp
static int BLOCKS = (96 * 16);

// size of unsorted array
// static int SIZE = 16777216; // 2^24
// static int SIZE = 250000000; // 2Gbs worth of elems
// static int SIZE = 268435456; // about the same
// static int SIZE = 65536; // 2^16
static int SIZE = 100;

// max value for couting sort
static int MAX_VALUE = 1024;

// sorting method
enum Method {
    RADIX = 0,
    COUNTING = 1,
    BITONIC = 2,
};

static int METHOD = 0;

#define HELPSTRING                                                                                                     \
    "Not enough arguments!\n"                                                                                          \
    "./sort METHOD <size> <threads> <blocks> [max value]\n"                                                            \
    "\n"                                                                                                               \
    "METHOD is one of three:\n"                                                                                        \
    "\t 0 --> Radix sort\n"                                                                                            \
    "\t 1 --> Counting sort\n"                                                                                         \
    "\t 2 --> Bitonic sort\n"                                                                                          \
    "\n"                                                                                                               \
    "Giving 0 instead of <size> <threads> or <blocks> uses their default values.\n"                                    \
    "\n"                                                                                                               \
    "Counting sort also requires the extra argument [max value].\n"

int main(int argc, char *argv[])
{
    int tmp;

    if (argc < 5) {
        fprintf(stderr, "%s", HELPSTRING);
        exit(-1);
    }

    // parse arguments
    tmp = atoi(argv[1]);
    if (tmp != 0)
        METHOD = tmp;
    tmp = atoi(argv[2]);
    if (tmp != 0)
        SIZE = tmp;
    tmp = atoi(argv[3]);
    if (tmp != 0)
        THREADS = tmp;
    tmp = atoi(argv[4]);
    if (tmp != 0)
        BLOCKS = tmp;

    // get extra argument if required
    if (METHOD == COUNTING) {
        if (argc < 6) {
            fprintf(stderr, "%s", HELPSTRING);
            exit(-1);
        } else {
            tmp = atoi(argv[5]);
            if (tmp != 0)
                MAX_VALUE = tmp;
        }
    }

    printf("Using %d threads in %d blocks\n", THREADS, BLOCKS);
    printf("Array of size %d\n", SIZE);

    // check for GPUs
    int gpus;
    cudaGetDeviceCount(&gpus);
    if (gpus < 1) {
        fprintf(stderr, "Need at least one GPU\n");
        exit(-1);
    }

    // print device info (assuming exactly one device)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*((float)prop.memoryBusWidth/8)/1.0e6);

    elem *unsorted = NULL;
    // elem *sorted2 = NULL;
    size_t size = SIZE;
    int threads = THREADS;
    int blocks = BLOCKS;

    printf("Allocating memory for unsorted..\n");
    unsorted = (elem *)malloc(size * sizeof(elem));
    if (unsorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    // get seed for rand_r
    srand(time(0));
    unsigned int seed = rand();

    printf("Generating input array..\n");
    for (size_t i = 0; i < size; ++i) {
        unsorted[i] = rand_r(&seed) % MAX_VALUE;
    }

    #ifdef PRINT
    if (size <= 100)
        print_array(unsorted, size, "unsorted");
    #endif

    Result result;

    switch (METHOD) {
    case RADIX:
        fprintf(stderr, "Using Radix sort!\n");
        result = radix_sort(unsorted, size, threads, blocks);
        break;
    case COUNTING:
        fprintf(stderr, "Using Counting sort!\n");
        result = counting_sort(unsorted, size, threads, blocks, MAX_VALUE);
        break;
    case BITONIC:
        fprintf(stderr, "Using Bitonic sort!\n");
        result = bitonic_sort(unsorted, size, threads, blocks);
        break;
    default:
        fprintf(stderr, "No such method: %d\n", METHOD);
        fprintf(stderr, "%s", HELPSTRING);
        exit(-1);
    }

    #ifdef PRINT
    if (size <= 100)
        print_array(result.sorted, size, "sorted");
    #endif

    printf("Finished sorting in %f ms\n", result.time);

    // sorted2 = radix_sort(unsorted, size, threads, blocks);
    // #ifdef PRINT
    // print_array(sorted2, size, "sorted");
    // #endif
    //
    // print_compare_array((unsigned int*)sorted, (unsigned int*)sorted2, size);
}
