#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <xcore/parallel.h>

#ifdef __adsp_generated_h_exists__
#include <adsp_generated.h>
#else
// declare type which is never defined.
// now function definitions below are valid
// but this type can't be instatiated until 
// autogeneration has been done
struct audio_dsp_impl;
#endif

/// adsp pipeline state. The details of this struct 
/// depend on the pipeline which should be generated 
/// based on app requirements
typedef struct audio_dsp_impl audio_dsp_t;

void adsp_pipeline_init(audio_dsp_t* adsp);

DECLARE_JOB(adsp_pipeline_main, (audio_dsp_t*));
void adsp_pipeline_main(audio_dsp_t* adsp);

void adsp_pipeline_source(audio_dsp_t* adsp, int32_t** data);
bool adsp_pipeline_sink_nowait(audio_dsp_t* adsp, int32_t** data);
void adsp_pipeline_sink(audio_dsp_t* adsp, int32_t** data);

/// concrete type to give more flexibility to generated code.
typedef struct {
    module_instance_t** modules;
    size_t num_modules;
} adsp_module_array_t;


static inline adsp_module_array_t adsp_pipeline_get_modules(audio_dsp_t* adsp) {
    return (adsp_module_array_t){
        .modules = adsp->modules,
        .num_modules = adsp->num_modules
    };
}

