#include <stdio.h>
#include <unistd.h>

#include "radix.h"

#include "prefix_sum.h"
#include "helpers.h"
#include "types.h"
#include "print.h"

/*
 * Functions
 */

__global__ void count_masked(elem *array, size_t size, unsigned int *counts, unsigned int mask, size_t shift, size_t mask_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* block id and stride */
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    __shared__ unsigned int local_counts[KEYS_COUNT];

    // zero out the block local shared memory
    for (size_t i = block_tid; i < KEYS_COUNT; i += block_stride) {
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
        // debug("tid(%d): size=%ld stride=%d i=%ld\n", tid, size, stride, i);
        atomicAdd(&local_counts[(array[i] & mask) >> (mask_size * shift)], 1);
        // debug("tid(%d): local_counts[%u] = %d\n", tid, (array[i] & mask) >> (mask_size * shift), local_counts[(array[i] & mask) >> (mask_size * shift)]);
    }

    // __syncthreads(); if (tid == 0) print_array(local_counts, KEYS_COUNT, "local_counts");

    syncthreads();

    // copy per block results back to ____global____ memory
    for (size_t i = block_tid; i < KEYS_COUNT; i += block_stride) {
        // debug("block(%d): adding local_counts[%lu]=%u to counts[%lu]=%u\n",blockIdx.x, i, local_counts[i], i, counts[i]);
        atomicAdd(&(counts[i]), local_counts[i]);
    }
}

/*
 * Should be called with KEYS_COUNT many threads in total.
 * Assuming 256 KEYS_COUNT and at least 32 threads per block so we get a full warp,
 * 8 blocks with 32 threads each seems reasonable.
 *
 * Allows us to possibly not copy between host and device so much.
 *
 * PERF: Not viable
 */
__global__ void move(elem *d_unsorted, size_t size, unsigned int *d_prefix, elem *d_sorted, unsigned int mask, size_t shift, size_t mask_size) {
    // use the thread id as a mask since we launch KEYS_COUNT many threads{{{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;


    int offset = 0;


    for (size_t j = 0; j < size; ++j) {
        ulong masked_elem = (d_unsorted[j] & mask) >> (mask_size * shift);

        // if this thread should handle this element
        if (masked_elem == tid) {
            offset = d_prefix[masked_elem];
            d_prefix[masked_elem] += 1;
            // debug("tid(%d) moved unsorted[%4lu]=%4d(%4d) to output[%4d]\n", j, d_unsorted[j], masked_elem, offset);
            d_sorted[offset] = d_unsorted[j];
        }
    }
}/*}}}*/


__host__ void host_move(unsigned int *prefix_sums, elem *unsorted, elem *sorted, size_t size, unsigned int mask, unsigned long mask_size, unsigned long shift) {

    // move elements to sorted position {{{
    int offset = 0;

    for (size_t j = 0; j < size; ++j) {
        ulong masked_elem = (unsorted[j] & mask) >> (mask_size * shift);

        offset = prefix_sums[masked_elem];
        // debug("! offset = prefix_sums[%lu] = %d, elem = %d, masked = %lu\n", masked_elem, offset, unsorted[j], masked_elem);
        prefix_sums[masked_elem] += 1;
        sorted[offset] = unsorted[j];
    }
} //}}}}


struct Result radix_sort(elem* unsorted, size_t size, int threads, int blocks)
{
    float time;
    // int threads = THREADS;
    // int blocks = BLOCKS;
    // size_t size = SIZE;

    // needed for current prefix_sum implementation
    if (threads * blocks < KEYS_COUNT) {
        fprintf(stderr, "We need at least KEYS_COUNT(=%d) threads in total. We have %d..\n", KEYS_COUNT, threads * blocks);
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
    counts = (unsigned int *)malloc(KEYS_COUNT * sizeof(unsigned int));
    if (counts == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    printf("Allocating memory for prefix_sums..\n");
    prefix_sums = (unsigned int *)malloc(KEYS_COUNT * sizeof(unsigned int));
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
    cudaErr(cudaMalloc((void **)&d_counts, KEYS_COUNT * sizeof(unsigned int)));
    printf("Allocating memory for d_sorted..\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    // size of elem in bits
    size_t elem_bit_size = sizeof(elem) * 8;
    // number of iterations needed to sort based on all bits
    ulong iters = (ulong)((double)elem_bit_size / log2(KEYS_COUNT));
    // size of the mask used to extract a key from an elem
    size_t mask_size = elem_bit_size / iters;

    unsigned int mask = 0;
    unsigned int mask_shift = 0;

    debug("KEYS_COUNT=%d\n", KEYS_COUNT);
    debug("elem_size=%lu\n", elem_bit_size);
    debug("iters=%lu\n", iters);
    debug("mask_size=%lu\n", mask_size);

    // check if KEYS_COUNT is correctly set
    if (mask_size * iters != elem_bit_size) {
        fprintf(stderr, "Bad KEYS_COUNT=%d value, elem_size=%lu is not integer "
            "divisible into iters=%lu many parts.\n", KEYS_COUNT, elem_bit_size, iters);
        exit(-1);
    }

    // record start time
    cudaEventRecord(start);

    // move array to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    // for testing, to make sure the values change
    // cudaErr(cudaMemcpy(d_output, output, size * sizeof(elem), cudaMemcpyHostToDevice));

    // print_array(unsorted, size, "unsorted");
    printf("Sorting..\n");

    for (size_t shift=0; shift < iters; ++shift) {

        printf("In device zero_array()\n");
        zero_array<<<blocks, threads>>>(d_counts, KEYS_COUNT);
        cudaLastErr();

        printf("##########################\n# ITERATION %2lu OUT OF %2lu #\n##########################\n", shift+1, iters);
        // keep a copy of the mask
        unsigned int old_mask = mask;

        // create a mask of size mask_size shifted appropriately
        for (; mask_shift < mask_size * (shift + 1); mask_shift++) {
                mask |= (1 << mask_shift);
        }

        // TODO: comments
        // now use the old_mask to trim off bits that we already used
        if (shift > 0) {
            mask ^= old_mask;
        }

        printf("mask:\n");
        print_bits(mask);

        // print_array(unsorted, size, "unsorted");
        // print_array_bits(unsorted, size, "unsorted bits");

        printf("In device count_masked()\n");
        // count frequencies
        count_masked<<<blocks, threads>>>(d_unsorted, size, d_counts, mask, shift, mask_size);
        cudaLastErr();

        // copy counts back to host only to print them
        // cudaErr(cudaMemcpy(counts, d_counts, KEYS_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // print_array(counts, KEYS_COUNT, "counts");

        printf("In device prefix_sum()\n");
        // get prefix sums of counts
        d_prefix_sums = prefix_sum(d_counts, KEYS_COUNT, blocks, threads);

        // copy prefix sums back to host because we *might* need them
        cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, KEYS_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // print_array(prefix_sums, KEYS_COUNT, "prefix_sum");
        // print_compare_array(counts, prefix_sums, KEYS_COUNT);
        // exit(0);

        printf("In host move()\n");
        host_move(prefix_sums, unsorted, sorted, size, mask, mask_size, shift);

        cudaErr(cudaMemcpy(d_unsorted, sorted, size * sizeof(elem), cudaMemcpyHostToDevice));
        memcpy(unsorted, sorted, size * sizeof(elem));
    }

    // record stop time
    cudaErr(cudaEventRecord(stop));
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
