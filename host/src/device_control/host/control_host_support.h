// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef __control_host_support_h__
#define __control_host_support_h__

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include "device_control_shared.h"

#if USE_XSCOPE

struct control_xscope_packet {
  control_resid_t resid;
  control_cmd_t cmd;
  uint8_t payload_len;
  uint8_t pad;
};

// hard limit of 256 bytes for xSCOPE uploads
#define XSCOPE_UPLOAD_MAX_BYTES (XSCOPE_UPLOAD_MAX_WORDS * 4)
#define XSCOPE_UPLOAD_MAX_WORDS 64
// subtract the header size from the total upload size
#define XSCOPE_DATA_MAX_BYTES (XSCOPE_UPLOAD_MAX_BYTES - 4)

#define XSCOPE_CONTROL_PROBE "Control Probe"

static inline size_t
control_xscope_create_upload_buffer(uint32_t buffer[XSCOPE_UPLOAD_MAX_WORDS],
                                    control_cmd_t cmd, control_resid_t resid,
                                    const uint8_t *payload, unsigned payload_len)
{
  const size_t header_size = sizeof(struct control_xscope_packet);
  struct control_xscope_packet *p;

  p = (struct control_xscope_packet*)buffer;

  p->resid = resid;
  p->cmd = cmd;

  assert(payload_len <= XSCOPE_DATA_MAX_BYTES && "payload length can't be represented as a uint8_t");
  p->payload_len = (uint8_t)payload_len;

  if (!IS_CONTROL_CMD_READ(cmd) && payload != NULL) {
    if (payload_len + header_size <= XSCOPE_UPLOAD_MAX_WORDS * sizeof(uint32_t)) {
      memcpy((uint8_t*)buffer + header_size, payload, payload_len);
    }
    return header_size + payload_len;
  }
  else {
    return header_size;
  }
}
#endif

#if USE_USB
#define USB_TRANSACTION_MAX_BYTES 256
static inline void
control_usb_fill_header(uint16_t *windex, uint16_t *wvalue, uint16_t *wlength,
                        control_resid_t resid, control_cmd_t cmd, unsigned payload_len)
{
  *windex = resid;
  *wvalue = cmd;

  assert(payload_len < (1<<16) && "payload length can't be represented as a uint16_t");
  *wlength = (uint16_t)payload_len;
}
#endif

#endif // __control_host_support_h__
