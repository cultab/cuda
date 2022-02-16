#ifndef COUNTING_H
#define COUNTING_H

#include <stdlib.h>
#include "types.h"

__global__ void count(elem *, size_t, unsigned int *, elem);
__global__ void counting_move(unsigned int *, elem *, elem *, size_t);
struct Result counting_sort(elem*, size_t, int, int, elem);

#endif
