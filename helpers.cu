/* zero out a device array */
__global__ void zero_array(unsigned int *d_array, size_t size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < size; i += stride) {
        // debug("write to %ld size = %ld\n", i, size);
        d_array[i] = 0;
    }
}

