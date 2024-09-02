// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.



// example that is used in the run time guide

// start example
#include <xcore/parallel.h>
#include "cmds.h"
#include "adsp_generated_auto.h"
#include "adsp_instance_id_auto.h"
#include "dsp/signal_chain.h"
#include "stages/adsp_control.h"
#include "stages/adsp_pipeline.h"

void control_thread(adsp_controller_t* control) {
  // convert desired value to parameter type
  float desired_vol_db = -6;
  int32_t desired_vol_raw = adsp_db_to_gain(desired_vol_db);

  adsp_stage_control_cmd_t command = {
    .instance_id = volume_stage_index,
    .cmd_id = CMD_VOLUME_CONTROL_TARGET_GAIN,
    .payload_len = sizeof(desired_vol_raw),
    .payload = &desired_vol_raw
  };

  // try write until success
  while(ADSP_CONTROL_SUCCESS != adsp_write_module_config(control, &command));

  // DONE!
}

void audio_source_sink(adsp_pipeline_t* p) {
  // sends and receives audio to the pipeline
}

void dsp_main(void) {
  adsp_pipeline_t* dsp = adsp_auto_pipeline_init();

  // created a controller instance for each thread.
  adsp_controller_t control;
  adsp_controller_init(&control, dsp);

  PAR_FUNCS(
    PFUNC(audio_source_sink, dsp),
    PFUNC(control_thread, &control),
    PFUNC(adsp_auto_pipeline_main, dsp)
  );
}
// end example

// start read
int32_t read_volume_gain(adsp_controller_t* control) {
  int32_t gain_raw;

  adsp_stage_control_cmd_t command = {
    .instance_id = volume_stage_index,
    .cmd_id = CMD_VOLUME_CONTROL_GAIN,
    .payload_len = sizeof(gain_raw),
    .payload = &gain_raw
  };

  // try write until success
  while(ADSP_CONTROL_SUCCESS != adsp_read_module_config(control, &command));

  return gain_raw;
}
// end read

// main just to ensure the code compiles. not expected to run
int main() {
  (void)read_volume_gain(NULL);
  dsp_main();
}
