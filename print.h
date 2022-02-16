#ifndef PRINT_H
#define PRINT_H

#include <unistd.h>
#include <stdint.h>
#include "types.h"

__device__
__host__
void print_array(int *, size_t, const char *);

__device__
__host__
void print_array(unsigned int *, size_t, const char *);

void print_bits(int);
void print_array_bits(int *, size_t, const char *);
void print_array_bits(unsigned int *, size_t, const char *);
void print_compare_array(unsigned int *, unsigned int *, size_t);
void print_bits(int num);

/*
 * Debug print, only prints if DEBUG is defined
 */

#ifdef DEBUG
#define debug(...) \
    printf(__VA_ARGS__);
#else
#define debug(...)
#endif

#endif

