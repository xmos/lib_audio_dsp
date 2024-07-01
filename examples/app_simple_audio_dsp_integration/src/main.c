// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/* System headers */
/* Standard library*/
#include <math.h>            // for M_TWOPI
#include <stdbool.h>         // for bool
#include <stdint.h>          // for int32_t and related typedefs

/* XMOS headers */
#include <platform.h>
#include <xs1.h>             // for resource macros such as XS1_PORT_*

/* lib_xcore */
/* This is an incomplete list of headers exposed by lib_xcore; see the XTC
    tools documentation for more details */
#include <xcore/assert.h>    // for xassert()
#include <xcore/chanend.h>   // for chanend_t
#include <xcore/channel.h>   // for channel_t and chan_(in|out)_word()
#include <xcore/hwtimer.h>   // for get_reference_time() and hwtimer functions
#include <xcore/parallel.h>  // for PAR_JOBS, DECLARE_JOB, and related macros
#include <xcore/select.h>    // for SELECT_RES and related macros

/* Other XMOS libraries */
#include <xmath/xmath.h>     // lib_xcore_math, an XS3 optimised maths library

/* Application libraries */
#include <adsp_generated_auto.h>

#define SINE_GENERATOR_FREQUENCY 1000  // Hertz
#define SAMPLE_RATE 16000              // Hertz
#define OUTPUT_BUFFER_LENGTH 1024      // samples
#define NUM_CHANNELS 1                 // dimensionless

/**
 * \enum channel_comms_t
 * \brief Defines some generic messages sent over channels to ease readability
 * \var channel_comms_t::CHAN_MSG_HANDSHAKE
 *   Use to denote a channel sending a "handshake" message, such as a request
 *   for data
 */
typedef enum channel_comms_t
{
  CHAN_MSG_HANDSHAKE
} channel_comms_t;

DECLARE_JOB(signal_producer, (chanend_t));
void signal_producer(chanend_t output_chanend)
{
    uint32_t const period_ticks = XS1_TIMER_HZ / SINE_GENERATOR_FREQUENCY;

    uint32_t ticks_since_period_start = 0;
    uint32_t ticks_since_last_call = 0;
    /* Initialise to a representative value - only matters on first call. */
    uint32_t last_process_ticks = 600;

    /* After initialisation, block until we receive our first request. Then,
     * send 0 immediately, and start timing - setting our sine's phase to be 0
     * when the first request is received. */
    chan_in_word(output_chanend);
    chan_out_word(output_chanend, 0);
    uint32_t then = get_reference_time();

    SELECT_RES(
        CASE_THEN(output_chanend, generate_sample)
    )
    {
    generate_sample:
    {
        /* We have received a sample request. Generate a sample and send. */

        /* Clear the event. */
        chan_in_word(output_chanend);

        uint32_t const now = get_reference_time();

        ticks_since_last_call = (now - then);
        /* We add last_process_ticks here to estimate how long this process will
         * take. If we do not, our output sine wave has too low a frequency! */
        ticks_since_period_start += ticks_since_last_call + last_process_ticks;
        ticks_since_period_start %= period_ticks;

        float angle = (ticks_since_period_start * M_TWOPI) / period_ticks;
        float result = f32_sin(angle);

        /* Output of f32_sin is not -1.0 <= sin(x) <= 1.0 due to float precision
         * Scaling by 1-__FLT_EPSILON__ here ensures -1.0 <= sin(x) <= 1.0. */
        int32_t int_result = (result * (1.0 - __FLT_EPSILON__)) * INT32_MAX;

        then = get_reference_time();
        last_process_ticks = (then - now);
        /* The out operation here is not expected to block, as the downstream
         * task should be blocking on us, having requested a sample. */
        chan_out_word(output_chanend, int_result);
    }
        SELECT_CONTINUE_NO_RESET;
  }
}

DECLARE_JOB(dsp_wrapper, (chanend_t, chanend_t, adsp_pipeline_t *));
void dsp_wrapper(chanend_t in, chanend_t out, adsp_pipeline_t * m_dsp)
{
    bool sample_requested = false;
    SELECT_RES(
        CASE_THEN(out, sample_requested),
        CASE_THEN(in, sample_received)
    )
    {
    sample_requested:
    {
        /* The downstream task has requested a sample. Send this request to the
         * sample generator. */

        /* Clear the event. */
        chan_in_word(out);

        /* If the previous request for a sample has not been serviced, we have
         * broken timing. */
        xassert_not(sample_requested);

        /* Send request for sample to upstream task. */
        chan_out_word(in, CHAN_MSG_HANDSHAKE);
        sample_requested = true;
    }
        SELECT_CONTINUE_NO_RESET;
    sample_received:
    {
        /* We have received a sample from the upstream task. Process it and send
         * the result downstream. */
        int32_t input_word = chan_in_word(in);

        /* If we are here but have not requested a sample, something has gone
         * very wrong. */
        xassert(sample_requested);

        /* Package the incoming sample. 
         * This example currently assumes one input channel. Additional members
         * of this array need to be supplied in order to add more channels. 
         * This example also currently assumes a frame size of 1. Increasing the
         * frame size would require each element of this array becoming an array
         * of length FRAME_SIZE. */
        int32_t * input_samples[NUM_CHANNELS] = {&input_word};

        /* Send the incoming sample to the DSP pipeline. */
        adsp_pipeline_source(m_dsp, input_samples);

        /* Get the processed output from the DSP pipeline. */
        int32_t * output_samples[NUM_CHANNELS] = {0};
        adsp_pipeline_sink(m_dsp, output_samples);

        /* We assume single-channel with frame size of 1. */
        int32_t output_word = output_samples[0][0];
        chan_out_word(out, output_word);

        sample_requested = false;
    }
        SELECT_CONTINUE_NO_RESET;
    }
}

DECLARE_JOB(signal_consumer, (chanend_t));
void signal_consumer(chanend_t input_chanend)
{
    /* Set up a dummy output buffer and state variable. */
    int32_t output_buffer[OUTPUT_BUFFER_LENGTH] = {0};
    uint32_t buffer_counter = 0;
    bool sample_pending = false;

    /* Get a new hardware timer and check we've got it. */
    timer_t event_timer = hwtimer_alloc();
    xassert(event_timer);

    /* Set up the first event on the hardware timer */
    uint32_t const interval_ticks = XS1_TIMER_HZ / SAMPLE_RATE;
    uint32_t now = hwtimer_get_time(event_timer);
    uint32_t trigger_time = now + interval_ticks;
    hwtimer_set_trigger_time(event_timer, trigger_time);

    /* Block on an event on either the event timer or the input chanend.
     * This is also implicitly a while(1) loop. */
    SELECT_RES( 
        CASE_THEN(event_timer, request_sample),
        CASE_THEN(input_chanend, sample_received)
    )
    {
    request_sample:
    {
        /* The event timer has gone off, which it will do every SAMPLE_RATE Hz. 
         * Send a handshake to the input chanend to request a new sample. */
        hwtimer_get_time(event_timer);

        /* The previous sample request should already have been serviced.
         * If it has not been, then we have broken timing. */
        xassert_not(sample_pending);

        /* Send a handshake - this should trigger the upstream tasks to send us
         * a sample. */
        chan_out_word(input_chanend, CHAN_MSG_HANDSHAKE);
        sample_pending = true;

        /* Set up the next trigger time for this event. */
        trigger_time += interval_ticks;
        hwtimer_set_trigger_time(event_timer, trigger_time);
    }
        SELECT_CONTINUE_NO_RESET;
    sample_received:
    {
        /* If we have received a sample when we have not requested one, 
         * something has gone very wrong. */
        xassert(sample_pending);

        /* Place the incoming sample in the output buffer, 
         * wrapping around when the buffer is full. */
        int32_t incoming_sample = chan_in_word(input_chanend);
        output_buffer[buffer_counter] = incoming_sample;
        buffer_counter = (buffer_counter + 1) % OUTPUT_BUFFER_LENGTH;
        sample_pending = false;
    }
        SELECT_CONTINUE_NO_RESET;
    }
}

void main()
{
    channel_t signal_dsp = chan_alloc();
    channel_t dsp_output = chan_alloc();
    adsp_pipeline_t * m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(
        PJOB(signal_producer, (signal_dsp.end_a)),
        PJOB(dsp_wrapper, (signal_dsp.end_b, dsp_output.end_a, m_dsp)),
        PJOB(signal_consumer, (dsp_output.end_b))
    );
}