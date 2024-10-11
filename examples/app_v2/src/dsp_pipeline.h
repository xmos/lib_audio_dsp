
#pragma once

#include <xcore/assert.h>

#define ADSP_SOURCE_FN_GROUP __attribute__((fptrgroup("ADSP_GENERATED_SOURCE")))
#define ADSP_SINK_FN_GROUP __attribute__((fptrgroup("ADSP_GENERATED_SINK")))

typedef void (*adsp_source_fn_t)(int, void*);
typedef void (*adsp_sink_fn_t)(int, void*);

typedef struct {
  ADSP_SOURCE_FN_GROUP adsp_source_fn_t source_fn;
  ADSP_SINK_FN_GROUP adsp_sink_fn_t sink_fn;
  void* stages;
  int nstages;
} adsp_generated_t;

static void adsp_source(adsp_generated_t* dsp, int channel, void* data) {
  xassert(dsp);
  xassert(data);
  dsp->source_fn(channel, data);
}

static void adsp_sink(adsp_generated_t* dsp, int channel, void* data) {
  xassert(dsp);
  xassert(data);
  dsp->sink_fn(channel, data);
}
