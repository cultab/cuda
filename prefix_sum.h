#ifndef PREFIX_SUM_H
#define PREFIX_SUM_H

#include <stdlib.h>

#include "types.h"
#include "helpers.h"

__host__ unsigned int *prefix_sum(unsigned int *, size_t, int, int);
__global__ void prefix_sum_kernel(unsigned int *, unsigned int *, unsigned int, size_t);

#endif
