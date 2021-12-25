/* #include <cuda.h> */
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "types.h"
#include "print.h"

/*
 * Constant Definitions and Cuda weirdness
 */

// number of different keys to count
#define KEYS_COUNT 256

// we need at least KEYS_COUNT threads in total
#define THREADS 128
#define BLOCKS 20
#define SIZE 130

// HACK: parenthesis are VERY IMPORTANT
#define KEY_MAX_VALUE (KEYS_COUNT - 1)

#ifdef __CUDA_ARCH__
#define syncthreads() __syncthreads()
#else
#define syncthreads()
#endif

/*
 * Forward Declarations
 */

__global__ void count_atomic(elem *, size_t, uint *, uint, size_t);
__host__ uint *prefix_sum(uint *, size_t, int, int);
__global__ void prefix_sum_kernel(uint *, uint *, uint, size_t);
__global__ void move(uint *, elem *, uint, uint);

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
 * Functions
 */

__global__ void count_atomic(elem *array, size_t size, uint *counts, uint mask, size_t shift, size_t mask_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ uint local_counts[KEYS_COUNT];

    if (tid < KEYS_COUNT) {
        local_counts[threadIdx.x] = 0;
    }
    /* printf("tid(%d) is here\n", tid); */
    syncthreads();

    for (size_t i = tid; i < size; i += stride) {
        /* debug("tid(%d): size=%ld stride=%d i=%ld\n", tid, size, stride, i); */
        atomicAdd(&local_counts[(array[i] & mask) >> (mask_size * shift)], 1);
        /* debug("tid(%d): local_counts[%u] = %d\n", tid, (array[i] & mask) >> (mask_size * shift), local_counts[(array[i] & mask) >> (mask_size * shift)]); */
    }

    /* __syncthreads(); if (tid == 0) print_array(local_counts, KEYS_COUNT, "local_counts"); */

    syncthreads();

    if (tid < KEYS_COUNT) {
        /* debug("adding local_counts[%d]=%d to counts[%d]=%d\n", threadIdx.x, local_counts[threadIdx.x], threadIdx.x, counts[threadIdx.x]); */
        atomicAdd(&(counts[threadIdx.x]), local_counts[threadIdx.x]);
    } else {
        /* debug("%d did nothing\n", tid); */
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

    /* uint *check = NULL; */
    /* check = (uint *)malloc(size * sizeof(uint)); */

    cudaErr(cudaMalloc((void **)&d_out, size * sizeof(uint)));
    cudaErr(cudaMalloc((void **)&d_in, size * sizeof(uint)));

    // initialize in and out array to counts
    cudaErr(cudaMemcpy(d_in, d_counts, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToDevice));
    cudaErr(cudaMemcpy(d_out, d_counts, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaLastErr();
        /* cudaErr(cudaMemcpy(check, d_out, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost)); */
        /* print_array(check, size, "out array:"); */

        // copy result back to input
        cudaErr(cudaMemcpy(d_in, d_out, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToDevice));
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

    // PERF: shift instead of pow(2, *)?
    // don't go out of bounds
    if (tid < size) {
        /* debug("tid(%d): something\n", tid); */
        /* printf("tid(%d) did something\n", tid); */
        if (tid >= __powf(2, j - 1)) {
            out[tid] += in[tid - (int)__powf(2, j - 1)];
            /* debug("out[%d] += %d\n", tid, in[tid - (int)__powf(2, j - 1)]); */
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
        /* debug("write to %ld size = %ld\n", i, size); */
        d_array[i] = 0;
    }
}

int main(void)
{
    int threads = THREADS;
    int blocks = BLOCKS;
    size_t size = SIZE;

    if (threads * blocks < KEYS_COUNT) {
        printf("We need at least KEYS_COUNT(=%d) threads in total.\n", KEYS_COUNT);
        exit(-1);
    }

    elem *unsorted = NULL;
    elem *output = NULL;
    uint *counts = NULL;
    uint *prefix_sums = NULL;

    elem *d_unsorted = NULL;
    uint *d_counts = NULL;
    uint *d_prefix_sums = NULL;
    elem *d_output = NULL;

    unsorted = (elem *)malloc(size * sizeof(elem));
    if (unsorted == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    counts = (uint *)malloc(KEYS_COUNT * sizeof(uint));
    if (counts == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    prefix_sums = (uint *)malloc(KEYS_COUNT * sizeof(uint));
    if (prefix_sums == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/
    output = (elem *)malloc(size * sizeof(elem));
    if (output == NULL) {/*{{{*/
        printf("malloc failed at line: %d in file %s\n", __LINE__, __FILE__);
        exit(-1);
    }/*}}}*/

    cudaErr(cudaMalloc((void **)&d_unsorted, size * sizeof(elem)));
    cudaErr(cudaMalloc((void **)&d_counts, KEYS_COUNT * sizeof(uint)));
    cudaErr(cudaMalloc((void **)&d_output, size * sizeof(elem)));

    for (size_t i = 0; i < size; ++i) {
        /* unsorted[i] = rand() % (1 * KEY_MAX_VALUE) + KEY_MAX_VALUE; */
        /* unsorted[i] = rand() % 1000; */
        /* unsorted[i] = rand(); */
        /* unsorted[i] = rand() % KEY_MAX_VALUE; */
        unsorted[i] = i;

        output[i] = -1337;
    }
    /* unsorted[2] = 0; */
    /* unsorted[4] = 1; */
    /* unsorted[5] = 255; */

    /* for (int i = 0; i < KEYS_COUNT; ++i) { */
    /*     counts[i] = 0; */
    /* } */

    // move array to device
    cudaErr(cudaMemcpy(d_unsorted, unsorted, size * sizeof(elem), cudaMemcpyHostToDevice));

    // for testing, to make sure the values change
    cudaErr(cudaMemcpy(d_output, output, size * sizeof(elem), cudaMemcpyHostToDevice));

    size_t elem_size = sizeof(elem) * 8;
    long unsigned iters = (long unsigned)((double)elem_size / log2(KEYS_COUNT));
    size_t mask_size = elem_size / iters;
    unsigned int mask = 0;
    unsigned int mask_shift = 0;
    /* unsigned int KEYS_COUNT = pow(2, 8); */

    for (size_t shift=0; shift < iters; ++shift) {

        zero_array<<<blocks, threads>>>(d_counts, KEYS_COUNT);
        cudaLastErr();

        debug("##########################\n# ITERATION %2lu OUT OF %2lu #\n##########################\n", shift+1, iters);
        // keep a copy of the mask
        uint old_mask = mask;

        // create a mask of size mask_s
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

        print_array(unsorted, size, "unsorted");
        /* print_array_bits(unsorted, size, "unsorted bits"); */

        // count frequencies
        count_atomic<<<blocks, threads>>>(d_unsorted, size, d_counts, mask, shift, mask_size);
        cudaLastErr();

        // copy counts back to host only to print them
        cudaErr(cudaMemcpy(counts, d_counts, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
        print_array(counts, KEYS_COUNT, "counts");

        // get prefix sums of counts
        d_prefix_sums = prefix_sum(d_counts, KEYS_COUNT, blocks, threads);

        // copy prefix sums back to host because we need them
        cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, KEYS_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
        print_array(prefix_sums, KEYS_COUNT, "prefix_sum");
        /* print_compare_array(counts, prefix_sums, KEYS_COUNT); */

        /* move elements to sorted position */
        int offset = 0;
        prefix_sums[KEYS_COUNT - 1] = 0;
        for (size_t j = 0; j < size; ++j) {
            unsigned long masked_elem = (unsorted[j] & mask) >> (mask_size * shift);
            
            /* printf("elem %d\nmasked ", unsorted[j]); */
            /* print_bits(masked_elem); */

            if (masked_elem != 0) {
                offset = prefix_sums[masked_elem - 1];
                /* debug("! offset = prefix_sums[%lu] = %d, elem = %d, masked = %lu\n", masked_elem - 1, offset, unsorted[j], masked_elem); */
                prefix_sums[masked_elem - 1] += 1;
                /* debug("n moved unsorted[%4lu]=%4d to output[%4d]\n", j, unsorted[j], offset); */
            } else {
                offset = prefix_sums[KEYS_COUNT - 1];
                /* debug("0 offset = prefix_sums[%d] = %d, elem = %d, masked = %lu\n", KEYS_COUNT - 1, offset, unsorted[j], masked_elem); */
                prefix_sums[KEYS_COUNT - 1] += 1;
                /* debug("0 moved unsorted[%4lu]=%4d to output[%4d]\n", j, unsorted[j], offset); */
            }

            /* if (offset > size) { */
            /*     debug("OFFSET = %d mskelem = %lu\n", offset, masked_elem); */
            /*     exit(-1); */
            /* } */

            output[offset] = unsorted[j];
        }

        print_array(output, size, "sorted");
        /* print_array_bits(output, size, "sorted bits"); */

        cudaErr(cudaMemcpy(d_unsorted, output, size * sizeof(elem), cudaMemcpyHostToDevice));
        cudaErr(cudaMemcpy(unsorted, output, size * sizeof(elem), cudaMemcpyHostToHost));
    }

    /* free device memory */
    puts("FREE DEVICE");
    cudaErr(cudaFree((void*)d_unsorted));
    cudaErr(cudaFree((void*)d_counts));
    cudaErr(cudaFree((void*)d_prefix_sums));
    cudaErr(cudaFree((void*)d_output));

    /* free host memory */
    puts("FREE HOST");
    free(unsorted);
    free(counts);
    free(prefix_sums);
    free(output);

    puts("DONE");
}

/* trash code {{{ */

/* nope */
/* __global__ void move(elem *unsorted, size_t size, uint *prefix_sums, elem *output, uint mask, uint shift) */
/* { */
/*     int tid = blockDim.x * blockIdx.x + threadIdx.x; */
/*     int stride = blockDim.x * gridDim.x; */
/*     int offset = 0; */
/*  */
    /*     __shared__ uint local_offsets[KEYS_COUNT]; */
/*  */
    /* // offset is prefix sum of previous number, */
    /* // if there is no previous thread, use the last pos in the array, */
    /* // initializing it to zero */
    /* if (tid == 0) { */
    /*     prefix_sums[KEYS_COUNT - 1] = prefix_sums[1]; */
    /* } */
/*  */
/*     syncthreads(); */
/*  */
/*     // i is int, should it be size_t? */
/*     for (int i = size - tid - 1; i >= 0; i -= stride) { */
/*         if ((unsorted[i] & mask) >> (8 * shift) != 0) { */
/*             offset = atomicSub(&prefix_sums[(unsorted[i] & mask) >> (8 * shift)], 1); */
/*             debug("tid(%d) move unsorted[%d]=%d to output[%d]=%d\n", tid, i, unsorted[i], offset - 1, output[offset - 1]); */
/*             output[offset - 1] = unsorted[i]; */
/*         } */
/*     } */
/*  */
/*     syncthreads(); */
/*  */
/*  */
    /* __syncthreads(); */
    /* if (tid == 0) print_array(local_counts, KEYS_COUNT, "local_counts"); */
/* } */

/*         prefix_sums[KEYS_COUNT - 1] = 0; */
/*  */
/*         for (int j = (int)size - 1; j >= 0; --j) { */
/*             unsigned long masked_elem = (unsorted[j] & mask) >> (mask_size * shift); */
/*              */
/*             printf("elem %d\nmasked ", unsorted[j]); */
/*             print_bits(masked_elem); */
/*  */
/*             if (masked_elem != 0) { */
/*                 offset = prefix_sums[masked_elem - 1]; */
                /* debug("! offset = prefix_sums[%lu] = %d, elem = %d, masked = %lu\n", masked_elem - 1, offset, unsorted[j], masked_elem); */
/*                 prefix_sums[masked_elem - 1] += 1; */
/*             } else { */
/*                 offset = prefix_sums[KEYS_COUNT - 1]; */
                /* debug("0 offset = prefix_sums[%d] = %d, elem = %d, masked = %lu\n", KEYS_COUNT - 1, offset, unsorted[j], masked_elem); */
/*                 prefix_sums[KEYS_COUNT - 1] += 1; */
/*             } */
/*  */
            /* if (offset > size) { */
            /*     debug("OFFSET = %d mskelem = %lu\n", offset, masked_elem); */
            /*     exit(-1); */
            /* } */
/*  */
/*             debug("moved unsorted[%4d]=%4d to output[%4d]\n", j, unsorted[j], offset); */
/*             output[offset] = unsorted[j]; */
/*         } */

        /* prefix_sums[KEYS_COUNT - 1] = prefix_sums[0] - 1; */



// }}}

/* commented  out code {{{*/
/* __global__ void move(int *array, int size, int *prefix, int *output, unsigned int mask) { */
/*         int tid = threadIdx.x; */
/*         int offset = 0; */
/*  */
/*         if (tid != 0) { */
/*                 offset = prefix[tid - 1]; */
/*         } else { */
/*                 offset = 0; */
/*                 // print_arr_in_gpu(prefix, size, "dev_prefix"); */
/*                 // printf("-------------- prefix[tid-1] = prefix[%d] = %d\n", tid-1, prefix[tid-1]); */
/*                 // printf("-------------- prefix[tid-2] = prefix[%d] = %d\n", tid-2, prefix[tid-2]); */
/*         } */
/*  */
/*         printf("tid(%d): offset=%d\n", tid, offset); */
/*  */
/*         for (int i=size-1; i>=0; --i) { */
/*                 // if this thread cares for the current number */
/*                 if ((array[i]) == tid) { */
/*                         output[offset++] = array[i]; */
/*                         printf("moving %d from array[%d] to output[%d]\n", array[i], i, offset - 1); */
/*                         // printf("output[%d] = array[%d] = %d\n", offset - 1, i, array[i]); */
/*                 } */
/*         } */
/* } */
/*}}}*/
