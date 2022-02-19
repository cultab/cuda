#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "counting.h"

#include "helpers.h"
#include "prefix_sum.h"
#include "print.h"
#include "types.h"

__global__ void count(elem *array, size_t size, unsigned int *counts, elem max_value)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* block id and stride */
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    extern __shared__ unsigned int local_counts[];

    // zero out the block local shared memory
    for (size_t i = block_tid; i < max_value; i += block_stride) {
        local_counts[i] = 0;
        // debug("block(%d) zero'ed [%lu]\n", blockIdx.x, i);
    }
    // printf("tid(%d) is here\n", tid);
    syncthreads();

    // if (tid == 0) {
    //     print_array(counts, KEYS_COUNT, "local_counts -----------");
    // }
    // syncthreads();

    for (size_t i = tid; i < size; i += stride) {
        atomicAdd(&local_counts[array[i]], 1);
    }

    syncthreads();

    // copy per block results back to global memory
    for (size_t i = block_tid; i < max_value; i += block_stride) {
        // debug("block(%d): adding local_counts[%lu]=%u to counts[%lu]=%u\n",blockIdx.x, i, local_counts[i], i, counts[i]);
        atomicAdd(&(counts[i]), local_counts[i]);
    }
}

__global__ void counting_move(unsigned int *d_prefix_sums, elem *d_unsorted, elem *d_sorted, size_t size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    elem cur_elem;
    unsigned int offset = 0;

    // int offset = 0;
    //
    // for (size_t j = 0; j < size; ++j) {
    //     ulong masked_elem = (unsorted[j] & mask) >> (mask_size * shift);
    //
    //     offset = prefix_sums[masked_elem];
    //     // debug("! offset = prefix_sums[%lu] = %d, elem = %d, masked = %lu\n", masked_elem, offset, unsorted[j], masked_elem);
    //     prefix_sums[masked_elem] += 1;
    //     sorted[offset] = unsorted[j];
    // }
    for (int i = tid; i < size; i += stride) {
        cur_elem = d_unsorted[i];
        offset = atomicAdd(&d_prefix_sums[cur_elem], 1);
        d_sorted[offset] = cur_elem;
    }
}

struct Result counting_sort(elem* unsorted, size_t size, int threads, int blocks, elem max_value)
{
    float time;

    // needed for current prefix_sum implementation
    if (threads * blocks < max_value) {
        fprintf(stderr, "We need at least KEYS_COUNT(=%d) threads in total. We have %d..\n", max_value, threads * blocks);
        exit(-1);
    }

    cudaEvent_t start, stop;

    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    elem *sorted = NULL;
    unsigned int *counts = NULL;
    unsigned int *prefix_sums = NULL;

    elem *d_unsorted = NULL;
    unsigned int *d_counts = NULL;
    unsigned int *d_prefix_sums = NULL;
    elem *d_sorted = NULL;

    printf("Allocating memory for counts..\n");
    counts = (unsigned int *)malloc(max_value * sizeof(unsigned int));
    if (counts == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    printf("Allocating memory for prefix_sums..\n");
    prefix_sums = (unsigned int *)malloc(max_value * sizeof(unsigned int));
    if (prefix_sums == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    printf("Allocating memory for sorted..\n");
    sorted = (elem *)malloc(size * sizeof(elem));
    if (sorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/


    printf("Allocating memory for d_unsorted..\n");
    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    printf("Allocating memory for d_counts..\n");
    cudaErr(cudaMalloc((void **)&d_counts, max_value * sizeof(unsigned int)));
    printf("Allocating memory for d_sorted..\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    // record start time
    cudaEventRecord(start);

    // move array to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    // print_array(unsorted, size, "unsorted");
    printf("Sorting..\n");

    printf("In device zero_array()\n");
    zero_array<<<blocks, threads>>>(d_counts, max_value);
    cudaLastErr();

    // print_array(unsorted, size, "unsorted");
    // print_array_bits(unsorted, size, "unsorted bits");

    printf("In device count()\n");
    // count frequencies
    count<<<blocks, threads, max_value * sizeof(unsigned int)>>>(d_unsorted, size, d_counts, max_value);
    cudaLastErr();

    // copy counts back to host only to print them
    // cudaErr(cudaMemcpy(counts, d_counts, max_value * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // print_array(counts, max_value, "counts");

    printf("In prefix_sum()\n");
    // get prefix sums of counts
    d_prefix_sums = prefix_sum(d_counts, max_value, blocks, threads);

    // copy prefix sums back to host because we *might* need them
    // cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, max_value * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // print_array(prefix_sums, max_value, "prefix_sum");
    // print_compare_array(counts, prefix_sums, max_value);

    printf("In device move()\n");
    // host_move(prefix_sums, unsorted, sorted, size, 0xffffffff, 32, 0);
    counting_move<<<blocks, threads>>>(d_prefix_sums, d_unsorted, d_sorted, size);
    cudaLastErr();

    // copy result back
    cudaErr(cudaMemcpy(sorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToHost));

    // record stop time
    cudaErr(cudaEventRecord(stop));
    cudaErr(cudaEventSynchronize(stop));

    cudaErr(cudaEventElapsedTime(&time, start, stop));

    /* free device memory */
    printf("Freeing device memory..\n");
    cudaErr(cudaFree((void*)d_unsorted));
    cudaErr(cudaFree((void*)d_counts));
    cudaErr(cudaFree((void*)d_prefix_sums));
    cudaErr(cudaFree((void*)d_sorted));

    /* free host memory */
    printf("Freeing host memory..\n");
    free(counts);
    free(prefix_sums);

    debug("DONE\n");

    Result res = { .sorted = sorted, .time = time };

    return res;
}
