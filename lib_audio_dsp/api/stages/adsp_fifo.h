// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Very simple FIFO designed to provide channel-like API with asychronous comms.
//
// This utility is used in the generated pipeline and should not be considered safe
// for use in application code.
//
// This is single-producer, single-consumer FIFO. The assumption is that the producer
// will have 1 or more things to send to the receiver in a transation. The producer
// can therefore call write multiple times before finishing the transaction. The consumer
// will be blocked until the producer has completed a transaction and will then be able
// to read from the FIFO.
//
// This FIFO also uses a channel to notify the consumer that there is data available.
// This is to allow the consumer to have a select case which is activated by the FIFO.
//
// To do this select on the `rx_end` member of adsp_fifo_t.
#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "swlock.h"
#include "xcore/chanend.h"
#include "xcore/channel.h"

typedef enum {
    _ADSP_FIFO_READ,
    _ADSP_FIFO_WRITE,
    _ADSP_FIFO_READ_DONE,
    _ADSP_FIFO_WRITE_DONE,
} adsp_fifo_state_t;

typedef struct {
    uint8_t *buffer;
    int32_t head;               // Index of the next byte to read
    chanend_t tx_end;
    chanend_t rx_end;
    volatile adsp_fifo_state_t state; // State of the FIFO
} adsp_fifo_t;

/**
 * Initialize a FIFO.
 *
 * Provide a buffer which will be used to hold the data. The buffer must be big
 * enough to hold the sum of all data written in a write transaction.
 *
 * @warning its on you to make sure the buffer is big enough
 *
 * @param fifo Pointer to FIFO structure to initialize
 */
static inline void adsp_fifo_init(adsp_fifo_t* fifo, void* buffer) {
    fifo->head = 0;
    fifo->state = _ADSP_FIFO_READ_DONE;
    channel_t c = chan_alloc();
    fifo->tx_end = c.end_a;
    fifo->rx_end = c.end_b;
    fifo->buffer = (uint8_t*)buffer;
}

/**
 * Start a write, blocks until fifo state is ready for a write.
 */
static inline void adsp_fifo_write_start(adsp_fifo_t* fifo) {
    while (fifo->state != _ADSP_FIFO_READ_DONE) {
        // Wait for the FIFO to be ready for writing
    }
    fifo->state = _ADSP_FIFO_WRITE;
    fifo->head = 0;
}

/**
 * Write data to the FIFO.
 *
 * Always call adsp_fifo_write_start() before this function to ensure
 * the FIFO is in the correct state for writing.
 *
 * Always call adsp_fifo_write_done() after this function to notify
 * The reading thread that the FIFO is ready for reading again.
 */
static inline void adsp_fifo_write(adsp_fifo_t* fifo, const void* data, size_t size_bytes) {
    memcpy(&fifo->buffer[fifo->head], data, size_bytes);
    fifo->head += size_bytes;
}

/**
 * Update FIFO state to indicate the write is finished.
 *
 * This also sends a word on the internal channel to notify
 * a waiting thread that the FIFO is ready for a read.
 */
static inline void adsp_fifo_write_done(adsp_fifo_t* fifo) {
    fifo->state = _ADSP_FIFO_WRITE_DONE;
    // send notification
    chanend_out_word(fifo->tx_end, 0);
}

/**
 * Wait until write is done and update state to show read is in progress.
 *
 * Reads from the internal channel to clear the notification.
 */
static inline void adsp_fifo_read_start(adsp_fifo_t* fifo) {
    chanend_in_word(fifo->rx_end);
    while (fifo->state != _ADSP_FIFO_WRITE_DONE) {
        // Wait for the FIFO to be ready for reading
    }
    // clear notification
    fifo->state = _ADSP_FIFO_READ;
    fifo->head = 0;
}

/**
 * Read data from the FIFO.
 *
 * Always call adsp_fifo_read_start() before this function to ensure
 * the FIFO is in the correct state for reading.
 *
 * Always call adsp_fifo_read_done() after this function to notify
 * writing thread that the FIFIO is ready for writing again.
 *
 * @param fifo Pointer to the FIFO
 * @param data Pointer to buffer to store read data
 * @param size_bytes Number of bytes to read
 */
static inline void adsp_fifo_read(adsp_fifo_t* fifo, void* data, size_t size_bytes) {
    memcpy(data, &fifo->buffer[fifo->head], size_bytes);
    fifo->head += size_bytes;
}

/**
 * Update FIFO state to indicate the read is finished.
 */
static inline void adsp_fifo_read_done(adsp_fifo_t* fifo) {
    fifo->state = _ADSP_FIFO_READ_DONE;
}
