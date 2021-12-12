#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define MAGIC_NUM 256

#define cudaErrChk(ans)                                                                                                \
    do {                                                                                                               \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define cudaError()                                                                                                    \
    do {                                                                                                               \
        cudaError_t cudaerr = cudaDeviceSynchronize();                                                                 \
        if (cudaerr != cudaSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(cudaerr), __FILE__,      \
                    __LINE__);                                                                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// create a new type to easily change it later if need be
typedef int32_t elem;

void print_array(elem *, size_t, char *);
void print_array(uint *, size_t, char *);
__device__ void print_array_GPU(elem *, size_t, char *);
__device__ void print_array_GPU(uint *, size_t, char *);
__global__ void countAtomic(elem *, int, uint *);
__host__ uint *prefix_sum(uint *, size_t, int, int);
__global__ void prefix_sum_kernel(uint *, uint *, uint, size_t);

/* print_array {{{*/
void print_array(int *arr, size_t size, char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_array(uint *arr, size_t size, char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%u ", arr[i]);
    }
    printf("\n");
}

__device__ void print_array_GPU(int *arr, size_t size, char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

__device__ void print_array_GPU(uint *arr, size_t size, char *name)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < size; ++i) {
        printf("%u ", arr[i]);
    }
    printf("\n");
} /*}}}*/

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
    /* if (tid == 0) print_array_GPU(local_counts, MAGIC_NUM, "local_counts"); */

    __syncthreads();

    if (tid < MAGIC_NUM) {
        printf("adding local_counts[%d]=%d to counts[%d]=%d\n", threadIdx.x, local_counts[threadIdx.x], threadIdx.x,
               counts[threadIdx.x]);
        atomicAdd(&(counts[threadIdx.x]), local_counts[threadIdx.x]);
    } else {
        printf("%d did nothing\n", tid);
    }
}

__host__ uint *prefix_sum(uint *counts, size_t size, int blocks, int threads)
{
    uint *d_in;
    uint *d_out;
    uint *d_temp;
    uint *check = NULL;

    check = (uint *)malloc(size * sizeof(uint));

    cudaErrChk(cudaMalloc((void **)&d_out, size * sizeof(uint)));
    cudaErrChk(cudaMalloc((void **)&d_in, size * sizeof(uint)));

    // initialize in and out array to counts
    cudaErrChk(cudaMemcpy(d_in, counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));
    cudaErrChk(cudaMemcpy(d_out, counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaError();

        /* cudaErrChk(cudaMemcpy(check, d_out, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost)); */
        /* print_array(check, size, "out array:"); */
        // copy result back to input
        cudaErrChk(cudaMemcpy(d_in, d_out, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToDevice));
        // swap in and out
        d_temp = d_in;
        d_in = d_out;
        d_out = d_temp;
    }

    // free out
    cudaErrChk(cudaFree(d_out));

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

int main(void)
{
    int threads = 256;
    int blocks = 2;

    int size = 10;
    elem *array = NULL;
    uint *counts = NULL;
    uint *prefix_sums = NULL;

    elem *d_array = NULL;
    uint *d_counts = NULL;
    uint *d_prefix_sums = NULL;

    array = (elem *)malloc(size * sizeof(elem));
    counts = (uint *)malloc(MAGIC_NUM * sizeof(uint));
    prefix_sums = (uint *)malloc(MAGIC_NUM * sizeof(uint));

    cudaErrChk(cudaMalloc((void **)&d_array, size * sizeof(elem)));
    cudaErrChk(cudaMalloc((void **)&d_counts, MAGIC_NUM * sizeof(uint)));

    for (size_t i = 0; i < size; ++i) {
        // HACK: only because we only iterate once
        array[i] = rand() % MAGIC_NUM;
    }
    for (int i = 0; i < MAGIC_NUM; ++i) {
        counts[i] = -1;
    }

    cudaErrChk(cudaMemcpy(d_array, array, size * sizeof(elem), cudaMemcpyHostToDevice));

    print_array(array, size, "unsorted");
    /* test<<<blocks, threads>>>(); */
    countAtomic<<<blocks, threads>>>(d_array, size, d_counts);

    cudaErrChk(cudaMemcpy(counts, d_counts, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost));

    print_array(counts, MAGIC_NUM, "counts");

    d_prefix_sums = prefix_sum(d_counts, MAGIC_NUM, 4, MAGIC_NUM/4);

    cudaErrChk(cudaMemcpy(prefix_sums, d_prefix_sums, MAGIC_NUM * sizeof(uint), cudaMemcpyDeviceToHost));

    print_array(prefix_sums, MAGIC_NUM, "prefix_sum");
}
