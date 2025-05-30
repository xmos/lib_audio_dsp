// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/// @file
///
/// Generated pipeline interface. Use the source and sink functions defined here
/// to send samples to the generated DSP and receive processed samples back.


#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <xcore/select.h>

#include "adsp_module.h"
#include "stages/adsp_fifo.h"

/// @cond
///
/// Private stuff

/// Mapping of input index to channel index for the source and sink configuration.
typedef struct
{
    /// @privatesection
    uint32_t channel_idx;
    uint32_t data_idx;
    uint32_t frame_size;
} adsp_mux_elem_t;

/// Source and sink configuration.
typedef struct
{
    /// @privatesection
    adsp_mux_elem_t *chan_cfg;
    size_t n_chan;
} adsp_mux_t;


/// Check if a chanend has an event pending.
static inline bool check_chanend(chanend_t c) {
    SELECT_RES(CASE_THEN(c, has_data), DEFAULT_THEN(no_data)) {
        has_data: return true;
        no_data: return false;
    }
}

/// @endcond

/// The DSP pipeline.
///
/// The generated pipeline will contain an init function that returns a pointer
/// to one of these. It can be used to send data in and out of the pipeline, and
/// also execute control commands.
typedef struct
{
    /// @privatesection
    adsp_fifo_t *p_in;
    size_t n_in;
    channel_t *p_out;
    size_t n_out;
    channel_t *p_link;
    size_t n_link;
    /// @publicsection
    module_instance_t *modules;  ///< Array of DSP stage states, must be used when calling one of the control functions.
    size_t n_modules;  ///< Number of modules in the @ref adsp_pipeline_t::modules array.
    /// @privatesection
    adsp_mux_t input_mux;
    adsp_mux_t output_mux;
} adsp_pipeline_t;


/// Pass samples into the DSP pipeline.
///
/// These samples are sent by value to the other thread, therefore the data buffer can be reused
/// immediately after this function returns. This function copies the data to a shared buffer where
/// the DSP pipeline will read them. If called again before the DSP pipeline has emptied the buffer
/// then this will block. If called when the buffer is empty then it will not block.
///
/// If this is called on the same thread as `adsp_pipeline_sink` then it will be most efficient to call
/// sink first followed by source. This allows the channel transactions internal to the pipeline to be
/// completed while the app is copying the input data to the buffer.
///
/// @param adsp The initialised pipeline.
/// @param data An array of arrays of samples. The length of the array shall be the number
///             of pipeline input channels. Each array contained within shall be contain a frame
///             of samples large enough to pass to the stage that it is connected to.
static inline void adsp_pipeline_source(adsp_pipeline_t *adsp, int32_t **data)
{
    for(int i = 0; i < adsp->n_in; i++)
    {
        adsp_fifo_write_start(&adsp->p_in[i]);
    }
    for (size_t chan_id = 0; chan_id < adsp->input_mux.n_chan; chan_id++)
    {
        adsp_mux_elem_t cfg = adsp->input_mux.chan_cfg[chan_id];
        adsp_fifo_write(&adsp->p_in[cfg.channel_idx],
                         (uint32_t *)data[cfg.data_idx],
                         cfg.frame_size*4);
    }
    for(int i = 0; i < adsp->n_in; i++)
    {
        adsp_fifo_write_done(&adsp->p_in[i]);
    }
}

/// Receive samples from the DSP pipeline.
///
/// Sink is implemented via a blocking channel transaction with the DSP thread and therefore will
/// block until the output threads are available to send the processed data. It should also be noted
/// that if `adsp_pipeline_sink` is called late then the DSP pipeline will block waiting for the
/// application to read the processed data. The application author must take care to ensure that data
/// is being read in a timely fashion.
///
/// @param adsp The initialised pipeline.
/// @param data An array of arrays that will be filled with processed samples from the pipeline.
///             The length of the array shall be the number
///             of pipeline input channels. Each array contained within shall be contain a frame
///             of samples large enough to pass to the stage that it is connected to.
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


/// Non-blocking receive from the pipeline. It is risky to use this API in an isochronous
/// application as the sink thread can lose synchronisation with the source thread which can
/// cause the source thread to block.
///
/// @param adsp The initialised pipeline.
/// @param data See adsp_pipeline_sink for details of same named param.
/// @retval true The data buffer has been filled with new values from the pipeline.
/// @retval false The pipeline has not produced any more data. The data buffer was untouched.
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
