#ifndef BITONIC_H
#define BITONIC_H

#include <stdlib.h>

#include "types.h"

__global__ void bitonic_step(elem*, size_t, int, int);
struct Result bitonic_sort(elem *, size_t, int, int);

#endif

