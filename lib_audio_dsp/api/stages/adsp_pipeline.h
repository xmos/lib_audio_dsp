#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <xcore/select.h>

#include <stages/adsp_module.h>

typedef struct
{
    uint32_t channel_idx;
    uint32_t data_idx;
    uint32_t frame_size;
} adsp_mux_elem_t;

typedef struct
{
    adsp_mux_elem_t *chan_cfg;
    size_t n_chan;
} adsp_mux_t;

// All fields of this structure are private. Please do not access directly.
typedef struct
{
    channel_t *p_in;
    size_t n_in;
    channel_t *p_out;
    size_t n_out;
    channel_t *p_link;
    size_t n_link;
    module_instance_t **modules;
    size_t n_modules;
    adsp_mux_t input_mux;
    adsp_mux_t output_mux;
} adsp_pipeline_t;

static inline void adsp_pipeline_source(adsp_pipeline_t *adsp, int32_t **data)
{
    for (size_t chan_id = 0; chan_id < adsp->input_mux.n_chan; chan_id++)
    {
        adsp_mux_elem_t cfg = adsp->input_mux.chan_cfg[chan_id];
        chan_in_buf_word(adsp->p_in[cfg.channel_idx].end_a,
                         (uint32_t *)data[cfg.data_idx],
                         cfg.frame_size);
    }
}

static inline void adsp_pipeline_sink(adsp_pipeline_t *adsp, int32_t **data)
{
    for (size_t chan_id = 0; chan_id < adsp->output_mux.n_chan; chan_id++)
    {
        adsp_mux_elem_t cfg = adsp->output_mux.chan_cfg[chan_id];
        chan_in_buf_word(adsp->p_out[cfg.channel_idx].end_b,
                         (uint32_t *)data[cfg.data_idx],
                         cfg.frame_size);
    }
}

static inline bool check_chanend(chanend_t c) {
    SELECT_RES(CASE_THEN(c, has_data), DEFAULT_THEN(no_data)) {
        has_data: return true;
        no_data: return false;
    }
}

static inline bool adsp_pipeline_sink_nowait(adsp_pipeline_t *adsp,
                                             int32_t **data)
{
    bool ready = true;
    for (size_t chan = 0; chan < adsp->n_out; chan++)
    {
        ready &= check_chanend(adsp->p_out[chan].end_b);
    }
    if (ready)
    {
        adsp_pipeline_sink(adsp, data);
    }
    return ready;
}
