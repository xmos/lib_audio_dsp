// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stages/bump_allocator.h>
#include <xcore/assert.h>


void* adsp_bump_allocator_malloc(adsp_bump_allocator_t* allocator, size_t n_bytes) {
    xassert(NULL != allocator);
    if(n_bytes & 0x03) {
        // A not word alligned size requested, this will break future
        // allocations so ban it.
        __builtin_trap();
    }
    if(n_bytes > allocator->n_bytes_left || NULL == allocator->buf) {
        // There is not enough space left in the allocator
        __builtin_trap();
    }
    if(0 == n_bytes) {
        return NULL;
    }

    void* ret = allocator->buf;

    allocator->buf += n_bytes;
    allocator->n_bytes_left -= n_bytes;

    return ret;
}
