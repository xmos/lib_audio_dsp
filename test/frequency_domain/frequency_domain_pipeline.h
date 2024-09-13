#pragma once
#include <stages/adsp_pipeline.h>
#include "xmath/xmath.h"



// typedef struct{
//     int n_inputs;
//     int n_outputs;
//     int frame_size;
//     int32_t buffer_len;
//     int32_t* buffer_data;
// } buffer_state_t;

// typedef struct {
//     // nothing configureable!
// } buffer_config_t;

typedef struct {
    int n_inputs;
    int n_outputs;
    int frame_size;
    int32_t nfft;
    int32_t* data;
    int32_t exp;
    bfp_s32_t signal;
    bfp_complex_s32_t spectrum;
} fft_state_t;



// void buffer_init(module_instance_t* instance,
//                  adsp_bump_allocator_t* allocator,
//                  uint8_t id,
//                  int n_inputs,
//                  int n_outputs,
//                  int frame_size);


//=======================================================

adsp_pipeline_t * adsp_auto_pipeline_init();

DECLARE_JOB(adsp_auto_pipeline_main, (adsp_pipeline_t*));
void adsp_auto_pipeline_main(adsp_pipeline_t* adsp);
void adsp_auto_print_thread_max_ticks(void);
