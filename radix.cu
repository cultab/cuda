#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <set>

#include "helpers.h"
#include "radix.h"
#include "types.h"
#include "print.h"

/*
 * Functions
 */

__global__ void count_radix(elem *array, size_t size, uint *counts, uint mask, size_t shift, size_t mask_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* block id and stride */
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    __shared__ uint local_counts[KEYS_COUNT];

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

    // copy per block results back to global memory
    for (size_t i = block_tid; i < KEYS_COUNT; i += block_stride) {
        // debug("block(%d): adding local_counts[%lu]=%u to counts[%lu]=%u\n",blockIdx.x, i, local_counts[i], i, counts[i]);
        atomicAdd(&(counts[i]), local_counts[i]);
    }
}

__global__ void count_counting(elem *array, size_t size, uint *counts, elem max_value)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* block id and stride */
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    extern __shared__ uint local_counts[];

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

/*
 * Implements All Partial Sums of an Array from:
 *
 *     Hillis, W. Daniel; Steele, Jr., Guy L. (December 1986).
 *     "Data parallel algorithms".
 *     Communications of the ACM. 29 (12): 1170–1183.
 *     doi:10.1145/7902.7903
 *
 */
 /*          = prefix_sum(      d_counts,  KEYS_COUNT,     blocks,     threads); */
__host__ uint *prefix_sum(uint *d_counts, size_t size, int blocks, int threads)
{
    uint *d_in;
    uint *d_out;
    uint *d_temp;

    // uint *check = NULL;
    // check = (uint *)malloc(size * sizeof(uint));

    cudaErr(cudaMalloc((void **)&d_out, size * sizeof(uint)));
    cudaErr(cudaMalloc((void **)&d_in, size * sizeof(uint)));

    /*
     * Initialize in and out array to counts
     * but shifted once to the right,
     * the first element of each array is memset to 0. (so that we can set it from host code)
     */
    cudaMemset(d_in, 0, 1);
    cudaMemset(d_out, 0, 1);
    cudaErr(cudaMemcpy(d_in + 1, d_counts, (size - 1) * sizeof(uint), cudaMemcpyDeviceToDevice));
    cudaErr(cudaMemcpy(d_out + 1, d_counts, (size - 1) * sizeof(uint), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaLastErr();
        // cudaErr(cudaMemcpy(check, d_out, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
        // print_array(check, size, "out array:");

        // copy result back to input
        cudaErr(cudaMemcpy(d_in, d_out, size * sizeof(uint), cudaMemcpyDeviceToDevice));
        // swap in and out
        d_temp = d_in;
        d_in = d_out;
        d_out = d_temp;
    }

    // free out
    cudaErr(cudaFree(d_out));

    // NOTE: return input array (yes it's backwards)
    return d_in;
}

// TODO: maybe support ACTUALLY using multiple blocks
__global__ void prefix_sum_kernel(uint *in, uint *out, uint j, size_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if (tid == 0) {
    //     printf("in[0] = %d\n", in[0]);
    // }

    syncthreads();

    // PERF: shift instead of pow(2, *)?
    // don't go out of bounds
    if (tid < size) {
        // debug("tid(%d): something\n", tid);
        // printf("tid(%d) did something\n", tid);
        if (tid >= __powf(2, j - 1)) {
            out[tid] += in[tid - (int)__powf(2, j - 1)];
            // debug("out[%d] += %d\n", tid + 1, in[tid - (int)__powf(2, j - 1)]);
        }
    }
}

__global__ void hello()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    debug("tid(%d): blockDim %d blockIdx %d threadIdx %d\n", tid, blockDim.x, blockIdx.x, threadIdx.x);
}

/* zero out a device array */
__global__ void zero_array(uint *d_array, size_t size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < size; i += stride) {
        // debug("write to %ld size = %ld\n", i, size);
        d_array[i] = 0;
    }
}

/*
 * Should be called with KEYS_COUNT many threads in total.
 * Assuming 256 KEYS_COUNT and at least 32 threads per block so we get a full warp,
 * 8 blocks with 32 threads each seems reasonable.
 *
 * Allows us to possibly not copy between host and device so much.
 *
 */
__global__ void move(elem *d_unsorted, size_t size, uint *d_prefix, elem *d_sorted, uint mask, size_t shift, size_t mask_size) {
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


void host_move(uint *prefix_sums, elem *unsorted, elem *sorted, size_t size, uint mask, ulong mask_size, ulong shift) {

    /* move elements to sorted position *//*{{{*/
    int offset = 0;

    for (size_t j = 0; j < size; ++j) {
        ulong masked_elem = (unsorted[j] & mask) >> (mask_size * shift);

        offset = prefix_sums[masked_elem];
        // debug("! offset = prefix_sums[%lu] = %d, elem = %d, masked = %lu\n", masked_elem, offset, unsorted[j], masked_elem);
        prefix_sums[masked_elem] += 1;
        sorted[offset] = unsorted[j];
    }

    // print_array(output, size, "sorted");
    // print_array_bits(output, size, "sorted bits");

    // if (shift == 0) {
    //     exit(0);
    // }}}}
}


elem *radix_sort(elem* unsorted, size_t size, int threads, int blocks)
{
    debug("Start!\n");
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
    uint *counts = NULL;
    uint *prefix_sums = NULL;

    elem *d_unsorted = NULL;
    uint *d_counts = NULL;
    uint *d_prefix_sums = NULL;
    elem *d_sorted = NULL;

    debug("Allocating memory for counts.\n");
    counts = (uint *)malloc(KEYS_COUNT * sizeof(uint));
    if (counts == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    debug("Allocating memory for prefix_sums.\n");
    prefix_sums = (uint *)malloc(KEYS_COUNT * sizeof(uint));
    if (prefix_sums == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    debug("Allocating memory for sorted.\n");
    sorted = (elem *)malloc(size * sizeof(elem));
    if (sorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/


    debug("Allocating memory for d_unsorted.\n");
    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    debug("Allocating memory for d_counts.\n");
    cudaErr(cudaMalloc((void **)&d_counts, KEYS_COUNT * sizeof(uint)));
    debug("Allocating memory for d_sorted.\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    // move array to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    // for testing, to make sure the values change
    // cudaErr(cudaMemcpy(d_output, output, size * sizeof(elem), cudaMemcpyHostToDevice));

    // size of elem in bits
    size_t elem_bit_size = sizeof(elem) * 8;
    // number of iterations needed to sort based on all bits
    ulong iters = (ulong)((double)elem_bit_size / log2(KEYS_COUNT));
    // size of the mask used to extract a key from an elem
    size_t mask_size = elem_bit_size / iters;

    uint mask = 0;
    uint mask_shift = 0;

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

    // print_array(unsorted, size, "unsorted");

    // record start time
    cudaEventRecord(start);

    for (size_t shift=0; shift < iters; ++shift) {

        debug("In device zero_array()\n");
        zero_array<<<blocks, threads>>>(d_counts, KEYS_COUNT);
        cudaLastErr();
        debug("Out of device zero_array()\n");

        debug("##########################\n# ITERATION %2lu OUT OF %2lu #\n##########################\n", shift+1, iters);
        // keep a copy of the mask
        uint old_mask = mask;

        // create a mask of size mask_size shifted appropriately
        for (; mask_shift < mask_size * (shift + 1); mask_shift++) {
                mask |= (1 << mask_shift);
        }

        // TODO: comments
        // now use the old_mask to trim off bits that we already used
        if (shift > 0) {
            mask ^= old_mask;
        }

        debug("mask:\n");
        print_bits(mask);

        // print_array(unsorted, size, "unsorted");
        // print_array_bits(unsorted, size, "unsorted bits");

        debug("In device count_atomic()\n");
        // count frequencies
        count_radix<<<blocks, threads>>>(d_unsorted, size, d_counts, mask, shift, mask_size);
        cudaLastErr();
        debug("Out of device zero_array()\n");

        // copy counts back to host only to print them
        cudaErr(cudaMemcpy(counts, d_counts, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
        print_array(counts, KEYS_COUNT, "counts");

        debug("In prefix_sum()\n");
        // get prefix sums of counts
        d_prefix_sums = prefix_sum(d_counts, KEYS_COUNT, blocks, threads);
        debug("Out of prefix_sum()\n");

        // copy prefix sums back to host because we *might* need them
        cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
        print_array(prefix_sums, KEYS_COUNT, "prefix_sum");
        // print_compare_array(counts, prefix_sums, KEYS_COUNT);
        // exit(0);

        debug("In host move()\n");
        host_move(prefix_sums, unsorted, sorted, size, mask, mask_size, shift);
        debug("Out host of move()\n");
        cudaErr(cudaMemcpy(d_unsorted, sorted, size * sizeof(elem), cudaMemcpyHostToDevice));
        memcpy(unsorted, sorted, size * sizeof(elem));

        // printf("In device move()\n");
        // move<<<8, 32>>>(d_unsorted, size, d_prefix_sums, d_sorted, mask, shift, mask_size);
        // cudaLastErr();
        // printf("Out of device move()\n");
        // cudaErr(cudaMemcpy(d_unsorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToDevice));

        // cudaErr(cudaMemcpy(unsorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToHost));

        // print_array(sorted, size, "sorted");
    }
    // cudaErr(cudaMemcpy(sorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToHost));

    // record stop time
    cudaErr(cudaEventRecord(stop));
    cudaErr(cudaEventRecord(stop));

    cudaEventElapsedTime(&time, start, stop);

    printf("Finished sorting in %.2f ms!\n", time);

    /* free device memory */
    debug("Free device memory.\n");
    cudaErr(cudaFree((void*)d_unsorted));
    cudaErr(cudaFree((void*)d_counts));
    cudaErr(cudaFree((void*)d_prefix_sums));
    cudaErr(cudaFree((void*)d_sorted));

    /* free host memory */
    debug("Free host memory.\n");
    // free(unsorted);
    // free(counts);
    // free(prefix_sums);
    // free(sorted);

    debug("DONE\n");

    return sorted;
}

__global__ void counting_move(uint *d_prefix_sums, elem *d_unsorted, elem *d_sorted, size_t size)
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

elem *counting_sort(elem* unsorted, size_t size, int threads, int blocks, elem max_value)
{
    debug("Start!\n");
    float time;

    // needed for current prefix_sum implementation
    if (threads * blocks < max_value) {
        fprintf(stderr, "We need at least KEYS_COUNT(=%d) threads in total. We have %d..\n", KEYS_COUNT, threads * blocks);
        exit(-1);
    }

    cudaEvent_t start, stop;

    cudaErr(cudaEventCreate(&start));
    cudaErr(cudaEventCreate(&stop));

    elem *sorted = NULL;
    uint *counts = NULL;
    uint *prefix_sums = NULL;

    elem *d_unsorted = NULL;
    uint *d_counts = NULL;
    uint *d_prefix_sums = NULL;
    elem *d_sorted = NULL;

    debug("Allocating memory for counts.\n");
    counts = (uint *)malloc(max_value * sizeof(uint));
    if (counts == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    debug("Allocating memory for prefix_sums.\n");
    prefix_sums = (uint *)malloc(max_value * sizeof(uint));
    if (prefix_sums == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    debug("Allocating memory for sorted.\n");
    sorted = (elem *)malloc(size * sizeof(elem));
    if (sorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/


    debug("Allocating memory for d_unsorted.\n");
    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    debug("Allocating memory for d_counts.\n");
    cudaErr(cudaMalloc((void **)&d_counts, max_value * sizeof(uint)));
    debug("Allocating memory for d_sorted.\n");
    cudaErr(cudaMalloc((void **)&d_sorted, size * sizeof(elem)));

    // move array to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    // for testing, to make sure the values change
    // cudaErr(cudaMemcpy(d_output, output, size * sizeof(elem), cudaMemcpyHostToDevice));

    // print_array(unsorted, size, "unsorted");

    // record start time
    cudaEventRecord(start);

    debug("In device zero_array()\n");
    zero_array<<<blocks, threads>>>(d_counts, max_value);
    cudaLastErr();
    debug("Out of device zero_array()\n");

    // print_array(unsorted, size, "unsorted");
    // print_array_bits(unsorted, size, "unsorted bits");

    debug("In device count_atomic()\n");
    // count frequencies
    count_counting<<<blocks, threads, max_value * sizeof(uint)>>>(d_unsorted, size, d_counts, max_value);
    cudaLastErr();
    debug("Out of device zero_array()\n");

    // copy counts back to host only to print them
    cudaErr(cudaMemcpy(counts, d_counts, max_value * sizeof(uint), cudaMemcpyDeviceToHost));
    print_array(counts, max_value, "counts");

    debug("In prefix_sum()\n");
    // get prefix sums of counts
    d_prefix_sums = prefix_sum(d_counts, max_value, blocks, threads);
    debug("Out of prefix_sum()\n");

    // copy prefix sums back to host because we *might* need them
    cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, max_value * sizeof(uint), cudaMemcpyDeviceToHost));
    print_array(prefix_sums, max_value, "prefix_sum");
    // print_compare_array(counts, prefix_sums, max_value);

    debug("In move()\n");
    // host_move(prefix_sums, unsorted, sorted, size, 0xffffffff, 32, 0);
    counting_move<<<blocks, threads>>>(d_prefix_sums, d_unsorted, d_sorted, size);
    cudaLastErr();
    debug("Out of move()\n");
    cudaErr(cudaMemcpy(sorted, d_sorted, size * sizeof(elem), cudaMemcpyDeviceToHost));

    // record stop time
    cudaErr(cudaEventRecord(stop));
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    printf("Finished sorting in %f ms!\n", time);

    /* free device memory */
    debug("Free device memory.\n");
    cudaErr(cudaFree((void*)d_unsorted));
    cudaErr(cudaFree((void*)d_counts));
    cudaErr(cudaFree((void*)d_prefix_sums));
    cudaErr(cudaFree((void*)d_sorted));

    /* free host memory */
    debug("Free host memory.\n");
    // free(unsorted);
    // free(counts);
    // free(prefix_sums);
    // free(sorted);

    debug("DONE\n");

    return sorted;
}
