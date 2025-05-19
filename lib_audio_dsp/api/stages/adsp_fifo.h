#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "swlock.h"

#define ADSP_FIFO_SIZE 1024

typedef enum {
    _ADSP_FIFO_READ,
    _ADSP_FIFO_WRITE,
    _ADSP_FIFO_READ_DONE,
    _ADSP_FIFO_WRITE_DONE,
} adsp_fifo_state_t;

/**
 * Efficient thread-safe FIFO for audio DSP applications.
 * Fixed buffer size of 1024 bytes.
 */
typedef struct {
    uint8_t buffer[ADSP_FIFO_SIZE];  // Fixed buffer of 1024 bytes
    int32_t head;               // Index of the next byte to read
    volatile adsp_fifo_state_t state; // State of the FIFO
} adsp_fifo_t;

/**
 * Initialize a FIFO.
 *
 * @param fifo Pointer to FIFO structure to initialize
 */
static inline void adsp_fifo_init(adsp_fifo_t* fifo) {
    fifo->head = 0;
    fifo->state = _ADSP_FIFO_READ_DONE;
}

static inline void adsp_fifo_write_start(adsp_fifo_t* fifo) {
    while (fifo->state != _ADSP_FIFO_READ_DONE) {
        // Wait for the FIFO to be ready for writing
    }
    fifo->state = _ADSP_FIFO_WRITE;
    fifo->head = 0;
}

/**
 * Write data to the FIFO. Blocks until all data can be written.
 * No divisions or modulo operations are used for efficiency.
 *
 * @param fifo Pointer to the FIFO
 * @param data Pointer to data to write
 * @param size_bytes Number of bytes to write
 */
static inline void adsp_fifo_write(adsp_fifo_t* fifo, const void* data, size_t size_bytes) {
    memcpy(&fifo->buffer[fifo->head], data, size_bytes);
    fifo->head += size_bytes;
}

static inline void adsp_fifo_write_done(adsp_fifo_t* fifo) {
    fifo->state = _ADSP_FIFO_WRITE_DONE;
}

static inline void adsp_fifo_read_start(adsp_fifo_t* fifo) {
    while (fifo->state != _ADSP_FIFO_WRITE_DONE) {
        // Wait for the FIFO to be ready for reading
    }
    fifo->state = _ADSP_FIFO_READ;
    fifo->head = 0;
}

/**
 * Read data from the FIFO. Blocks until all requested data is available.
 * No divisions or modulo operations are used for efficiency.
 *
 * @param fifo Pointer to the FIFO
 * @param data Pointer to buffer to store read data
 * @param size_bytes Number of bytes to read
 */
static inline void adsp_fifo_read(adsp_fifo_t* fifo, void* data, size_t size_bytes) {
    memcpy(data, &fifo->buffer[fifo->head], size_bytes);
    fifo->head += size_bytes;
}

static inline void adsp_fifo_read_done(adsp_fifo_t* fifo) {
    fifo->state = _ADSP_FIFO_READ_DONE;
}
