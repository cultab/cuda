#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "types.h"
#include "print_array.h"

/*
 * Type and Constant Definitions
 */

// number of different keys to count
#define MAGIC_NUM 256

/*
 * Forward Declarations
 */

__global__ void countAtomic(elem *, int, uint *);

__host__ uint *prefix_sum(uint *, size_t, int, int);
__global__ void prefix_sum_kernel(uint *, uint *, uint, size_t);

__global__ void move(uint *, elem *);

/*
 * Macros
 */

inline void cudaPrintError(cudaError_t cudaerr, char *file, int line)
{
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__);
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

__global__ void test()
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d here\n", tid);
}
__global__ void countAtomic(elem *array, int size, uint *counts)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ uint local_counts[MAGIC_NUM];

    if (tid < MAGIC_NUM) {
        local_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    // HACK: make it by order
    static uint mask = 0b11111111;

    for (size_t i = tid; i < size; i += stride) {
        /* printf("tid(%d): size=%d stride=%d i=%lu\n", tid, size, stride, i); */
        /* printf("%lu %p\n", i, array); */
        atomicAdd(&local_counts[array[i] & mask], 1);
        printf("tid(%d): local_counts[%u] = %d\n", tid, array[i] & mask, local_counts[array[i] & mask]);
    }

    /* __syncthreads(); */
    /* if (tid == 0) print_array(local_counts, MAGIC_NUM, "local_counts"); */

    __syncthreads();

    if (tid < MAGIC_NUM) {
        /* printf("adding local_counts[%d]=%d to counts[%d]=%d\n", threadIdx.x, local_counts[threadIdx.x], threadIdx.x, */
        /* counts[threadIdx.x]); */
        atomicAdd(&(counts[threadIdx.x]), local_counts[threadIdx.x]);
    } else {
        /* printf("%d did nothing\n", tid); */
    }
}

__host__ uint *prefix_sum(uint *counts, size_t size, int blocks, int threads)
{
    uint *d_in;
    uint *d_out;
    uint *d_temp;

    /* uint *check = NULL; */
    /* check = (uint *)malloc(size * sizeof(uint)); */

    cudaErr(cudaMalloc((void **)&d_out, size * sizeof(uint)));
    cudaErr(cudaMalloc((void **)&d_in, size * sizeof(uint)));

    // initialize in and out array to counts
    cudaErr(cudaMemcpy(d_in, counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));
    cudaErr(cudaMemcpy(d_out, counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaLastErr();

        /* cudaErr(cudaMemcpy(check, d_out, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost)); */
        /* print_array(check, size, "out array:"); */
        // copy result back to input
        cudaErr(cudaMemcpy(d_in, d_out, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));
        // swap in and out
        d_temp = d_in;
        d_in = d_out;
        d_out = d_temp;
    }

    // free out
    cudaErr(cudaFree(d_out));

    // return input array (yes)
    return d_in;
}

// TODO: maybe support ACTUALLY using multiple blocks
__global__ void prefix_sum_kernel(uint *in, uint *out, uint j, size_t size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // don't go out of bounds
    if (tid < size) {
        if (tid >= pow(2, j - 1)) {
            /* printf("adding %d to %d\n",in_idx, out_idx); */
            out[tid] += in[tid - (int)pow(2, j - 1)];
        }
    }
}

__global__ void move(elem *unsorted, size_t size, uint *prefix_sums, elem *output)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    /*     __shared__ uint local_offsets[MAGIC_NUM]; */
    /*  */
    /*     if (tid < MAGIC_NUM) { */
    /*         local_offsets[threadIdx.x] = 0; */
    /*     } */
    /*     __syncthreads(); */

    // HACK: make it by order
    static uint mask = 0b11111111;

    // offset is prefix sum of previous number,
    // if there is no previous thread, use the last pos in the array,
    // initializing it to zero
    if (tid == 0) {
        prefix_sums[MAGIC_NUM - 1] = 0;
    }

    // i is int, should it be size_t?
    for (int i = size - tid - 1; i >= 0; i -= stride) {
        if (unsorted[i] == 0) {
            offset = atomicAdd(&prefix_sums[MAGIC_NUM - 1], 1);
        } else {
            offset = atomicAdd(&prefix_sums[unsorted[i] - 1], 1);
        }
        printf("tid(%d) move unsorted[%d]=%d to output[%d]=%d\n", tid, i, unsorted[i], offset, output[offset]);
        output[offset] = unsorted[i];
    }

    /* __syncthreads(); */
    /* if (tid == 0) print_array(local_counts, MAGIC_NUM, "local_counts"); */

    /* __syncthreads(); */
}

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

int main(void)
{
    int threads = 256;
    int blocks = 2;

    int size = 10;
    elem *array = NULL;
    elem *output = NULL;
    uint *counts = NULL;
    uint *prefix_sums = NULL;

    elem *d_array = NULL;
    uint *d_counts = NULL;
    uint *d_prefix_sums = NULL;
    elem *d_output = NULL;

    array = (elem *)malloc(size * sizeof(elem));
    counts = (uint *)malloc(MAGIC_NUM * sizeof(uint));
    prefix_sums = (uint *)malloc(MAGIC_NUM * sizeof(uint));
    output = (elem *)malloc(size * sizeof(elem));

    cudaErr(cudaMalloc((void **)&d_array, size * sizeof(elem)));
    cudaErr(cudaMalloc((void **)&d_counts, MAGIC_NUM * sizeof(uint)));
    cudaErr(cudaMalloc((void **)&d_output, size * sizeof(elem)));

    for (size_t i = 0; i < size; ++i) {
        // HACK: only because we only iterate once
        array[i] = rand() % MAGIC_NUM;
        output[i] = -1337;
    }
    /* for (int i = 0; i < MAGIC_NUM; ++i) { */
    /*     counts[i] = 0; */
    /* } */

    cudaErr(cudaMemcpy(d_array, array, size * sizeof(elem), cudaMemcpyHostToDevice));

    // for testing, to make sure the values change
    cudaErr(cudaMemcpy(d_output, output, size * sizeof(elem), cudaMemcpyHostToDevice));

    print_array(array, size, "unsorted");
    /* test<<<blocks, threads>>>(); */

    countAtomic<<<blocks, threads>>>(d_array, size, d_counts);

    cudaErr(cudaMemcpy(counts, d_counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost));

    print_array(counts, MAGIC_NUM, "counts");

    d_prefix_sums = prefix_sum(d_counts, MAGIC_NUM, 4, MAGIC_NUM / 4);

    cudaErr(cudaMemcpy(prefix_sums, d_prefix_sums, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost));

    print_array(prefix_sums, MAGIC_NUM, "prefix_sum");

    move<<<blocks, threads>>>(d_array, size, d_prefix_sums, d_output);

    cudaErr(cudaMemcpy(output, d_output, size * sizeof(elem), cudaMemcpyDeviceToHost));

    print_array(output, size, "sorted?");
}
