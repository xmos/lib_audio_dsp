#pragma once
#include <stages/adsp_pipeline.h>
#include <xcore/parallel.h>

adsp_pipeline_t * adsp_auto_pipeline_init();

DECLARE_JOB(adsp_auto_pipeline_main, (adsp_pipeline_t*));
void adsp_auto_pipeline_main(adsp_pipeline_t* adsp);
void adsp_auto_print_thread_max_ticks(void);
