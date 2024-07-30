#if USE_XSCOPE

#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include "xscope_endpoint.h"
#include "device_control_host.h"
#include "control_host_support.h"
#include "util.h"

//#define DBG(x) x
#define DBG(x)
#define PRINT_ERROR(...)   fprintf(stderr, "Error  : " __VA_ARGS__)

#define UNUSED_PARAMETER(x) (void)(x)

static volatile unsigned int probe_id = 0xffffffff;
static volatile size_t record_count = 0;
static unsigned char *last_response = NULL;
static unsigned last_response_length = 0;
static unsigned num_commands = 0;

void register_callback(unsigned int id, unsigned int type,
  unsigned int r, unsigned int g, unsigned int b,
  unsigned char *name, unsigned char *unit,
  unsigned int data_type, unsigned char *data_name)
{
  UNUSED_PARAMETER(type);
  UNUSED_PARAMETER(r);
  UNUSED_PARAMETER(g);
  UNUSED_PARAMETER(b);
  UNUSED_PARAMETER(unit);
  UNUSED_PARAMETER(data_type);
  UNUSED_PARAMETER(data_name);

  if (strcmp((char*)name, XSCOPE_CONTROL_PROBE) == 0) {
    probe_id = id;
    DBG(printf("registered probe %d\n", id));
  }
}

void xscope_print(unsigned long long timestamp,
                  unsigned int length,
                  unsigned char *data) {
  UNUSED_PARAMETER(timestamp);

  if (length) {
    for (unsigned i = 0; i < length; i++) {
      printf("%c", *(&data[i]));
    }
  }
}

void record_callback(unsigned int id, unsigned long long timestamp,
  unsigned int length, unsigned long long dataval, unsigned char *databytes)
{
  UNUSED_PARAMETER(timestamp);
  UNUSED_PARAMETER(dataval);

  if (id == probe_id) {
    if (last_response != NULL) {
      free(last_response);
    }
    last_response = (unsigned char*)malloc(length);
    last_response_length = length;
    memcpy(last_response, databytes, length);

    record_count++;
  }
}

control_ret_t control_init_xscope(const char *host_str, const char *port_str)
{
  //int ret = xscope_ep_request_registered();
  //printf("xscope_ep_request_registered() returned %d\n", ret);

  if (xscope_ep_set_print_cb(xscope_print) != XSCOPE_EP_SUCCESS) {
    PRINT_ERROR("xscope_ep_set_print_cb failed\n");
    return CONTROL_ERROR;
  }

  if (xscope_ep_set_register_cb(register_callback) != XSCOPE_EP_SUCCESS) {
    PRINT_ERROR("xscope_ep_set_register_cb failed\n");
    return CONTROL_ERROR;
  }

  if (xscope_ep_set_record_cb(record_callback) != XSCOPE_EP_SUCCESS) {
    PRINT_ERROR("xscope_ep_set_record_cb failed\n");
    return CONTROL_ERROR;
  }

  if (xscope_ep_connect(host_str, port_str) != XSCOPE_EP_SUCCESS) {
    return CONTROL_ERROR;
  }

  //ret = xscope_ep_request_registered();
  //printf("xscope_ep_request_registered() returned %d\n", ret);

  DBG(printf("connected to server at port %s\n", port_str));

  // wait for xSCOPE probe registration
  while (probe_id == -1) {
    pause_short();
  }

  return CONTROL_SUCCESS;
}

/*
 * xSCOPE has an internally hardcoded limit of 256 bytes. Where it passes
 * the xSCOPE endpoint API upload command to xGDB server, it truncates
 * payload to 256 bytes.
 *
 * Let's have host code check payload size here. No additional checks on
 * device side. Device will need a 256-byte buffer to receive from xSCOPE
 * service.
 *
 * No checking of read data which goes in other direction, device to host.
 * This is xSCOPE probe bytes API, which has no limit.
 */
static bool upload_len_exceeds_xscope_limit(size_t len)
{
  if (len > XSCOPE_UPLOAD_MAX_BYTES) {
    PRINT_ERROR("Upload of %zd bytes requested\n", len);
    PRINT_ERROR("Maximum upload size is %d\n", XSCOPE_UPLOAD_MAX_BYTES);
    return true;
  }
  else {
    return false;
  }
}

control_ret_t
control_write_command(control_resid_t resid, control_cmd_t cmd,
                      const uint8_t payload[], size_t payload_len)
{
  unsigned b[XSCOPE_UPLOAD_MAX_WORDS];

  size_t len = control_xscope_create_upload_buffer(b,
    CONTROL_CMD_SET_WRITE(cmd), resid, payload, payload_len);

  if (upload_len_exceeds_xscope_limit(len))
    return CONTROL_DATA_LENGTH_ERROR;

  DBG(printf("%u: send write command: ", num_commands));
  DBG(print_bytes((unsigned char*)b, len));

  record_count = 0;

  if (xscope_ep_request_upload(len, (unsigned char*)b) != XSCOPE_EP_SUCCESS) {
    PRINT_ERROR("xscope_ep_request_upload failed\n");
    return CONTROL_ERROR;
  }
  // wait for response on xSCOPE probe
  while (record_count == 0) {
    pause_short();
  }

  DBG(printf("response: "));
  DBG(print_bytes(last_response, last_response_length));

  num_commands++;
  return last_response[0];
}

control_ret_t
control_read_command(control_resid_t resid, control_cmd_t cmd,
                     uint8_t payload[], size_t payload_len)
{
  unsigned b[XSCOPE_UPLOAD_MAX_WORDS];

  size_t len = control_xscope_create_upload_buffer(b,
    CONTROL_CMD_SET_READ(cmd), resid, NULL, payload_len);

  DBG(printf("%d: send read command, len %d: ", num_commands, len));
  DBG(print_bytes((unsigned char*)b, len));

  record_count = 0;

  if (xscope_ep_request_upload(len, (unsigned char*)b) != XSCOPE_EP_SUCCESS) {
    PRINT_ERROR("xscope_ep_request_upload failed\n");
    return CONTROL_ERROR;
  }

  // wait for response on xSCOPE probe
  while (record_count == 0) {
    pause_short();
  }

  DBG(printf("response: length %d: ", last_response_length));
  DBG(print_bytes(last_response, last_response_length));

  // ignore returned payload length, use one supplied in request
  memcpy(payload, last_response, payload_len);

  num_commands++;
  return CONTROL_SUCCESS;
}

control_ret_t control_cleanup_xscope(void)
{
  xscope_ep_disconnect();
  return CONTROL_SUCCESS;
}

#endif // USE_XSCOPE
