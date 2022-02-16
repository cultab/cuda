#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>

// create a new type to easily change it later if need be
typedef int32_t elem;
// typedef unsigned int unsigned int;
//

// stores the result array
// and the time for it to be computed
struct Result {
    elem* sorted;
    float time;
};

#endif
