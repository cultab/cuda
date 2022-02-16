#include "prefix_sum.h"

#include "types.h"
#include "print.h"
#include "helpers.h"

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
__host__ unsigned int *prefix_sum(unsigned int *d_counts, size_t size, int blocks, int threads)
{
    unsigned int *d_in;
    unsigned int *d_out;
    unsigned int *d_temp;

    // unsigned int *check = NULL;
    // check = (unsigned int *)malloc(size * sizeof(unsigned int));

    cudaErr(cudaMalloc((void **)&d_out, size * sizeof(unsigned int)));
    cudaErr(cudaMalloc((void **)&d_in, size * sizeof(unsigned int)));

    /*
     * Initialize in and out array to counts
     * but shifted once to the right,
     * the first element of each array is memset to 0. (so that we can set it from host code)
     */
    cudaMemset(d_in, 0, 1);
    cudaMemset(d_out, 0, 1);
    cudaErr(cudaMemcpy(d_in + 1, d_counts, (size - 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErr(cudaMemcpy(d_out + 1, d_counts, (size - 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaLastErr();
        // cudaErr(cudaMemcpy(check, d_out, KEYS_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // print_array(check, size, "out array:");

        // copy result back to input
        cudaErr(cudaMemcpy(d_in, d_out, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
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
__global__ void prefix_sum_kernel(unsigned int *in, unsigned int *out, unsigned int j, size_t size)
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

