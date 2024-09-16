// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/* System headers */
/* Standard library*/
#include <stdint.h>          // for int32_t and related typedefs
#include <string.h>          // for memset

/* XMOS headers */
#include <platform.h>
#include <xs1.h>             

/* lib_xcore */
/* This is an incomplete list of headers exposed by lib_xcore; see the XTC
    tools documentation for more details */
#include <xcore/assert.h>    // for xassert()
#include <xcore/hwtimer.h>   // for hwtimer_t and related functions
#include <xcore/parallel.h>  // for PAR_JOBS, DECLARE_JOB, and related macros

/* Application libraries */
#include <stages/adsp_pipeline.h>
#include <frequency_domain_pipeline.h>
#include "whitenoise_1024samples.h"

#define SAMPLE_RATE 48000              // Hertz
#define OUTPUT_BUFFER_LENGTH 256*4        // samples
#define NUM_CHANNELS 1                 // dimensionless
#define FRAME_LENGTH 256                 // samples

/* Set up a dummy output buffer and state variable. */
int32_t volatile output_buffer[OUTPUT_BUFFER_LENGTH];
uint8_t volatile buffer_loop_flag = 0;
uint32_t buffer_counter = 0;

DECLARE_JOB(signal_producer, (adsp_pipeline_t *));
void signal_producer(adsp_pipeline_t * m_dsp)
{
    /* Get a new hardware timer and check we've got it. */
    hwtimer_t event_timer = hwtimer_alloc();
    xassert(event_timer);
    printf("got timer\n");

    /* Set up the first event on the hardware timer */
    uint32_t const sample_period_ticks = XS1_TIMER_HZ / SAMPLE_RATE;
    uint32_t trigger_time = hwtimer_get_time(event_timer) + sample_period_ticks;
    hwtimer_set_trigger_time(event_timer, trigger_time);

    /* Set up the input buffer. We're using precalculated white noise
     * and will just loop through this over and over. */
    int32_t white_noise[] = {WHITENOISE_1024};
    uint32_t const n_samps = sizeof(white_noise) / sizeof(white_noise[0]);
    uint32_t sample_no = 0;

    while(1)
    {
        /* Block until the trigger time*/
        hwtimer_get_time(event_timer);
        
        /* Send the generated sample to the DSP pipeline. 
         * This example currently assumes one input channel. Additional members
         * of this array need to be supplied in order to add more channels. 
         * This example also currently assumes a frame size of 1. Increasing the
         * frame size would require each element of this array becoming an array
         * of length FRAME_SIZE. */
        int32_t input_samples[FRAME_LENGTH] = {0};
        memcpy(input_samples, &white_noise[sample_no], FRAME_LENGTH);

        // printf("input %ld %ld %ld %ld %ld\n", input_samples[0], input_samples[1], input_samples[2], input_samples[3], input_samples[4]);

        int32_t * input_channels[NUM_CHANNELS] = {input_samples};
        sample_no = (sample_no + FRAME_LENGTH) % n_samps;
        adsp_pipeline_source(m_dsp, input_channels);

        /* Set up the next sample period */
        trigger_time += sample_period_ticks;
        hwtimer_set_trigger_time(event_timer, trigger_time);
    }
}

DECLARE_JOB(signal_consumer, (adsp_pipeline_t *));
void signal_consumer(adsp_pipeline_t * m_dsp)
{
    int32_t output_word[FRAME_LENGTH] = {0};
    memset(output_buffer, 0, sizeof(int32_t) * OUTPUT_BUFFER_LENGTH);
    int32_t * output_samples[NUM_CHANNELS] = {output_word};
    while(1)
    {
        /* Get the processed output from the DSP pipeline. 
         * This operation blocks on data being available. 
         * We assume single-channel with frame size of 1. */
        adsp_pipeline_sink(m_dsp, output_samples);

        printf("%ld\n", output_word);
        /* Place the output word in the output buffer, 
         * wrapping around when the buffer is full. */
        output_buffer[buffer_counter] = output_word;
        // printf("outpt %ld %ld %ld %ld %ld\n", output_word[buffer_counter+0], output_word[buffer_counter+1], output_word[buffer_counter+2], output_word[buffer_counter+3], output_word[buffer_counter+4]);
        buffer_counter = (buffer_counter + 1) % OUTPUT_BUFFER_LENGTH;


        /* Set up an easy GDB watchpoint */
        if (buffer_counter == 0)
        {
            buffer_loop_flag = !buffer_loop_flag;
        }
    }
}

int main()
{
    adsp_pipeline_t * m_dsp = adsp_auto_pipeline_init();

    PAR_JOBS(
        PJOB(signal_producer, (m_dsp)),
        PJOB(signal_consumer, (m_dsp)),
        PJOB(adsp_auto_pipeline_main, (m_dsp))
    );
}
