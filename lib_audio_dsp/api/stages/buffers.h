/// M to N buffers for use in generated code for cross thread comms.
///
/// TODO: Give each buffer a chanend and wait on an `in` rather than
///       polling full/empty

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  void* data;
  size_t nbytes;
  volatile uint_fast8_t full;
} buffer_1to1_t;

#define BUFFER_1TO1_FULL(DATA, BYTES, _IGNORED) {\
  .data = (DATA), \
  .nbytes = (BYTES), \
  .full = true \
  }

static void buffer_1to1_write(buffer_1to1_t* buffer, void* data) {
  while(buffer->full);
  memcpy(buffer->data, data, buffer->nbytes);
  buffer->full = true;
}

static void buffer_1to1_read(buffer_1to1_t* buffer, void* data) {
  while(!buffer->full);
  memcpy(data, buffer->data, buffer->nbytes);
  buffer->full = false;
}

typedef struct {
  void* data;
  size_t in_bytes;
  size_t out_bytes;
  size_t read_idx;
  volatile uint_fast8_t empty;
} buffer_bigtosmall_t;

#define BUFFER_BIGTOSMALL_FULL(DATA, IN_BYTES, OUT_BYTES) {\
  .data=(DATA),\
  .in_bytes=(IN_BYTES),\
  .out_bytes=(OUT_BYTES),\
  .read_idx=0,\
  .empty=false\
  }

static void buffer_bigtosmall_write(buffer_bigtosmall_t* buffer, void* data) {
  while(!buffer->empty);
  memcpy(buffer->data, data, buffer->in_bytes);
  buffer->empty = false;
}

static void buffer_bigtosmall_read(buffer_bigtosmall_t* buffer, void* data) {
  while(buffer->empty);
  memcpy(data, &buffer->data[buffer->read_idx], buffer->out_bytes);
  buffer->read_idx += buffer->out_bytes;
  if(buffer->read_idx >= buffer->in_bytes) {
    buffer->empty = true;
    buffer->read_idx = 0;
  }
}

typedef struct {
  void* data;
  size_t in_bytes;
  size_t out_bytes;
  size_t write_idx;
  volatile uint_fast8_t full;
} buffer_smalltobig_t;


#define BUFFER_SMALLTOBIG_FULL(DATA, IN_BYTES, OUT_BYTES) {\
  .data=(DATA),\
  .in_bytes=(IN_BYTES),\
  .out_bytes=(OUT_BYTES),\
  .write_idx=0,\
  .full=true\
  }

static void buffer_smalltobig_write(buffer_smalltobig_t* buffer, void* data) {
  while(buffer->full);
  memcpy(&buffer->data[buffer->write_idx], data, buffer->in_bytes);
  buffer->write_idx += buffer->in_bytes;
  if(buffer->write_idx >= buffer->out_bytes) {
    buffer->full = true;
    buffer->write_idx = 0;
  }
  
}

static void buffer_smalltobig_read(buffer_smalltobig_t* buffer, void* data) {
  while(!buffer->full);
  memcpy(data, buffer->data, buffer->out_bytes);
  buffer->full = false;
}
