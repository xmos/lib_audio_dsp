#include <stages/bump_allocator.h>
#include <xcore/assert.h>


void* adsp_bump_allocator_malloc(adsp_bump_allocator_t* allocator, size_t n_bytes) {
    xassert(NULL != allocator);
    if(n_bytes > allocator->n_bytes_left || NULL == allocator->buf) {
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
