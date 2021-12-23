#include <stdio.h>
#include <math.h>

#define DIGITS 52

__global__ void count(int*, int, int*, unsigned int);
__global__ void move(int*, int, int*, int*, unsigned int);
void print_arr(int *, int);
__device__ void print_arr_in_gpu(int *, int , char*);
__host__ void print_arr_in(int *, int , char*);
int *prefix_sum(int *, int);

__global__ void count(int *array, int size, int *counts, unsigned int mask)
{
        int tid = threadIdx.x;

        for (int i=0; i<size; ++i) {
                // printf("tid(%d): %d & %x = %d ?=? %d\n", tid, array[i], mask, (array[i] & mask), tid);
                // printf("tid(%d): %d ?=? %d\n", tid, array[i], tid);
                if ((array[i]) == tid) {
                        counts[tid] += 1;
                        // printf("counts[%d]++\n", tid);
                }
        }
}

__global__ void move(int *array, int size, int *prefix, int *output, unsigned int mask) {
        int tid = threadIdx.x;
        int offset = 0;

        if (tid != 0) {
                offset = prefix[tid - 1];
        } else {
                offset = 0;
                // print_arr_in_gpu(prefix, size, "dev_prefix");
                // printf("-------------- prefix[tid-1] = prefix[%d] = %d\n", tid-1, prefix[tid-1]);
                // printf("-------------- prefix[tid-2] = prefix[%d] = %d\n", tid-2, prefix[tid-2]);
        }

        printf("tid(%d): offset=%d\n", tid, offset);

        for (int i=size-1; i>=0; --i) {
                // if this thread cares for the current number
                if ((array[i]) == tid) {
                        output[offset++] = array[i];
                        printf("moving %d from array[%d] to output[%d]\n", array[i], i, offset - 1);
                        // printf("output[%d] = array[%d] = %d\n", offset - 1, i, array[i]);
                }
        }
}

void print_arr(int *arr, int size) {

        for (int i=0; i<size; ++i) {
                printf("%d ", arr[i]);
        }
        printf("\n");
}

__device__ void print_arr_in_gpu(int *arr, int size, char* name) {

        for (int i=0; i<size; ++i) {
                printf("%s[%d]=%d\n", name, i, arr[i]);
        }
}

__host__ void print_arr_in(int *arr, int size, char* name) {

        for (int i=0; i<size; ++i) {
                printf("%s[%d]=%d\n", name, i, arr[i]);
        }
}

// TODO: parallel
int *prefix_sum(int *counts, int size)
{
        int *prefix = (int*)malloc(sizeof(int) * size);

        prefix[0] = counts[0];

        for (int i=1; i<size; ++i) {
                prefix[i] = prefix[i-1] + counts[i];
        }
                
        return prefix;
}



int main(void)
{
        int size = 100;
        int counts[DIGITS] = {0};
        int *out;
        int *prefix;

        int *dev_array;
        int *dev_counts;
        int *dev_prefix;
        int *dev_out;

        int *array;

        array = (int*)malloc(sizeof(int) * size);

        for (int i=0; i<size; i++) {
                array[i] = rand() % DIGITS;
                //array[i] = 124;
        }

        unsigned int mask = 0;
        for (int i = 0; i < log2((double)DIGITS); i++) {
                mask |= (1 << i);
                // printf("mask: %x\n", mask);
        }

        printf("%x | %x | %x\n", DIGITS-1, 0b0111, mask);
        //exit(0);

        print_arr(array, size);

        cudaMalloc((void**)&dev_array, sizeof(int) * size);
        cudaMalloc((void**)&dev_counts, sizeof(int) * DIGITS);
        cudaMalloc((void**)&dev_prefix, sizeof(int) * size);
        cudaMalloc((void**)&dev_out, sizeof(int) * size);

        cudaMemcpy(dev_array, array, sizeof(int) * size, cudaMemcpyHostToDevice);

        count<<<1,DIGITS-1>>>(dev_array, size, dev_counts, mask);

        cudaMemcpy(counts, dev_counts, sizeof(int) * DIGITS, cudaMemcpyDeviceToHost);
        printf("counts\n");
        print_arr(counts, size);

        prefix = prefix_sum(counts, size);

        printf("prefix\n");
        print_arr_in(prefix, size, "prefix");

        cudaMemcpy(dev_prefix, prefix, sizeof(int) * size, cudaMemcpyHostToDevice);

        move<<<1,DIGITS-1>>>(dev_array, size, dev_prefix, dev_out, mask);

        out = (int*)malloc(sizeof(int) * size);

        cudaMemcpy(out, dev_out, sizeof(int) * size, cudaMemcpyDeviceToHost);

        printf("result:\n");
        print_arr(out, size);
}











