// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
/// Implementation of a bump allocator. Bump allocators do not support freeing the memory.
#pragma once

#include <stdlib.h>
#include <stddef.h>

/// Bump allocator struct. Initialise with ADSP_BUMP_ALLOCATOR_INITIALISER and
/// use bump_allocator_malloc() to claim memory.
typedef struct {
    char* buf;
    int n_bytes_left;
} adsp_bump_allocator_t;

/// Initialise a bump allocator with this. Expects an array (not a pointer).
/// lifetime of the array must be at least as long as the lifetime of the
/// allocator.
#define ADSP_BUMP_ALLOCATOR_INITIALISER(array) {(sizeof(array) > 0) ? (void*)(array) : NULL, (sizeof(array) / sizeof(*(array)))}

/// Determine buf size required to ensure it can be DWORD_ALIGNED
#define ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(N) ( (N) + 7)

/// Allocate a DWORD_ALIGNED buffer
#define ADSP_BUMP_ALLOCATOR_DWORD_ALLIGNED_MALLOC(allocator, N) \
    (void*)(((((uint64_t)adsp_bump_allocator_malloc(allocator, ADSP_BUMP_ALLOCATOR_DWORD_N_BYTES(N))) + 7) >> 3) << 3)


/// Allocate some memory from the allocator, traps if there is not enough memory.
void* adsp_bump_allocator_malloc(adsp_bump_allocator_t* allocator, size_t n_bytes);
