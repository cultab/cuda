#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "types.h"
#include "print.h"
#include "helpers.h"

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
                // debug("\t\t\tyes! because v = %d\n", v == 0);
            }
        }
    }
}

struct Result bitonic_sort(elem *unsorted, size_t size, int threads, int blocks)
{

    elem *sorted = NULL;

    elem *d_sorted = NULL;
    elem *d_unsorted = NULL;

    // _true will also record the time to copy the data back and forth
    float time;
    cudaEvent_t start, stop;

    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    printf("Allocating memory for sorted..\n");
    sorted = (elem *)malloc(size * sizeof(elem));
    if (sorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    printf("Allocating memory for d_unsorted..\n");
    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    printf("Allocating memory for d_sorted..\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    cudaErr(cudaEventRecord(start));

    printf("Sorting..\n");

    // copy unsorted to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    for (int k = 2; k <= (int)size; k *= 2) { // k is doubled every iteration
        debug("k = %d\n", k);
        for (int j = k/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
            debug("\tj = %d\n", j);
            bitonic_step<<<blocks, threads>>>(d_unsorted, size, k, j);

            // #ifdef DEBUG
            // cudaErr(cudaMemcpy(sorted, d_unsorted, size * sizeof(elem), cudaMemcpyDeviceToHost));
            // print_array(sorted, size, "step");
            // #endif

        }
    }
    cudaLastErr();

    cudaErr(cudaMemcpy(sorted, d_unsorted, size * sizeof(elem), cudaMemcpyDeviceToHost));


    cudaErr(cudaEventRecord(stop));
    cudaErr(cudaEventSynchronize(stop));

    cudaErr(cudaEventElapsedTime(&time, start, stop));

    /* free device memory */
    printf("Freeing device memory..\n");
    cudaErr(cudaFree((void*)d_unsorted));
    cudaErr(cudaFree((void*)d_sorted));

    debug("DONE\n");

    Result res = { .sorted = sorted, .time = time };

    return res;
}
