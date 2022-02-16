#ifndef RADIX_H
#define RADIX_H
#include <unistd.h>

#include "types.h"

// number of different keys to count
#define KEYS_COUNT 256

/*
#define KEYS_COUNT 2    // 2 ^ 1
#define KEYS_COUNT 4    // 2 ^ 2
#define KEYS_COUNT 16   // 2 ^ 4 
#define KEYS_COUNT 256  // 2 ^ 8, just right ?
*/
/*
#define KEYS_COUNT 6536 // 2 ^ 16, way too big
*/

//  NOTE: parenthesis are VERY IMPORTANT
#define KEY_MAX_VALUE (KEYS_COUNT - 1)

/*
 * Forward Declarations
 */

__global__ void count_masked(elem *, size_t, unsigned int *, unsigned int, size_t);
__global__ void move(elem *, size_t, unsigned int *, elem, unsigned int, unsigned int, size_t, size_t);
__host__ void host_move(unsigned int *, elem *, elem *, size_t, unsigned int, unsigned long, unsigned long);
struct Result radix_sort(elem*, size_t, int, int);

#endif
