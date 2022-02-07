#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "types.cuh"
#include "print.cuh"

#define THREADS 128
// multiple of 96 for 1050ti says nvvp
#define BLOCKS (96 * 2)

// #define SIZE 8
// #define SIZE 2048
#define SIZE 32768

#ifdef __CUDA_ARCH__
#define syncthreads() __syncthreads()
#else
#define syncthreads()
#endif

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


__global__ void bitonic_step(elem* d_arr, size_t size, int k, int j)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x + gridDim.x;

    int tmp;
    int ij;
    int v;
    // i is the index of our first element
    for (int i = tid; i < size; i += stride) {
        // ij is the index of the second element, the one we compare the first with
        ij = i ^ j;
        // v keeps track of the order we want to sort the i'th chunk of size k,
        // if it's 0 it's ascending else it's descending
        v = i & k;
        // printf("\t\ti = %2d ij = %2d v = %2d\n", i, ij, v); printf("\t\tshould we swap %d with %d?\n", d_arr[i], d_arr[ij]);

        // if the element we compare to is after our element ?
        if (ij > i) {
            // if the elements are not in the correct order that we want
            if ((v == 0 && d_arr[i] > d_arr[ij]) || (v != 0 && (d_arr[i] < d_arr[ij]))) {
                // swap the elements at index i and ij
                tmp = d_arr[i];
                d_arr[i] = d_arr[ij];
                d_arr[ij] = tmp;
                debug("\t\t\tyes! because v = %d\n", v == 0);
            }
        }
    }
}

int main(void)
{
    int threads = THREADS;
    int blocks = BLOCKS;
    size_t size = SIZE;

    elem *sorted = NULL;
    elem *unsorted = NULL;

    elem *d_sorted = NULL;
    elem *d_unsorted = NULL;

    // _true will also record the time to copy the data back and forth
    float time, time_true;
    cudaEvent_t start, stop, start_true, stop_true;

    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));
    cudaErr(cudaEventCreate(&start_true));
    cudaErr(cudaEventCreate(&stop_true));

    printf("Allocating memory for unsorted.\n");
    unsorted = (elem *)malloc(size * sizeof(elem));
    if (unsorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    printf("Allocating memory for sorted.\n");
    sorted = (elem *)malloc(size * sizeof(elem));
    if (sorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    printf("Allocating memory for d_unsorted.\n");
    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    printf("Allocating memory for d_sorted.\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    printf("Generating input array.\n");
    for (size_t i = 0; i < size; ++i) {
        // unsorted[i] = rand() % INT32_MAX;
        // unsorted[i] = rand() % (1 * KEY_MAX_VALUE) + KEY_MAX_VALUE;
        unsorted[i] = rand() % 10;
        // unsorted[i] = rand();
        // unsorted[i] = rand() % 10000;
        // unsorted[i] = rand() % KEY_MAX_VALUE;
        // unsorted[i] = size - i;

        // sorted[i] = -1337;
    }

    // print_array(unsorted, size, "unsorted");


    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    cudaErr(cudaEventRecord(start_true));
    cudaErr(cudaEventRecord(start));

    for (int k = 2; k <= size; k *= 2) { // k is doubled every iteration
        debug("k = %d\n", k);
        for (int j = k/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
            debug("\tj = %d\n", j);
            // bitonic_step<<<blocks, threads>>>(d_unsorted, size, k, j);
            bitonic_step<<<1, 1>>>(d_unsorted, size, k, j);
            cudaLastErr();
            cudaErr(cudaMemcpy(sorted, d_unsorted, size * sizeof(elem), cudaMemcpyDeviceToHost));
            print_array(sorted, size, "step");

        }
    }

    cudaErr(cudaEventRecord(stop));

    // cudaErr(cudaMemcpy(sorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToHost));
    cudaErr(cudaMemcpy(sorted, d_unsorted, size * sizeof(elem), cudaMemcpyDeviceToHost));

    cudaErr(cudaEventRecord(stop_true));

    cudaErr(cudaEventSynchronize(stop));
    cudaErr(cudaEventSynchronize(stop_true));

    cudaEventElapsedTime(&time, start, stop);
    cudaEventElapsedTime(&time_true, start_true, stop_true);

    printf("Kernel execution time: %f ms!\n"
           "Including memcpy:      %f ms!\n", time, time_true);


    // print_array(sorted, size, "sorted");


    return 0;
}
