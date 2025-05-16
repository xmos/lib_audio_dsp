
/// This test intended to emulate the scenario which the generated pipeline
/// is used in the XUA application. Source and sink are called on the same
/// thread with strict latency/jitter requirements (must fit in the I2S callback).
///
/// This application will fail if timing is not met.


#include "adsp_generated_auto.h"
#include "stages/adsp_pipeline.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <xcore/parallel.h>
#include <xcore/hwtimer.h>

#ifndef FS
#error "FS not defined"
#endif

#ifndef FRAME_SIZE
#error "FRAME_SIZE not defined"
#endif

#ifndef N_CHANS
#error "N_CHANS not defined"
#endif

#ifndef N_THREADS
#error "N_THREADS not defined"
#endif

#define LOOP_COUNT 10

#define I2S_PERIOD_TICKS ((uint32_t)(100e6 / FS))


void test(adsp_pipeline_t* dsp) {
	int32_t source_buffer[N_CHANS][FRAME_SIZE];
	int32_t* source[N_CHANS];
	int32_t sink_buffer[N_CHANS][FRAME_SIZE];
	int32_t* sink[N_CHANS];

	for (int i = 0; i < N_CHANS; i++) {
		source[i] = &source_buffer[i][0];
		sink[i] = &sink_buffer[i][0];
	}

	int frame_count = 0;
	uint32_t last_time = get_reference_time();
	bool failed = false;
	for(int i = 0; i < LOOP_COUNT; i++) {

		// check we got back to here fast enough
		uint32_t elapsed = get_reference_time() - last_time;
		if((elapsed) > I2S_PERIOD_TICKS) {
			printf("ERROR Too slow: %lu > %lu\n", elapsed, I2S_PERIOD_TICKS);
			failed = true;
			break;
		}

		// wait for the next I2S period
		while(get_reference_time() - last_time < I2S_PERIOD_TICKS) {
			// do nothing
		}

		last_time = get_reference_time();

		if(0 == frame_count) {
			frame_count = FRAME_SIZE;

			adsp_pipeline_source(dsp, source);
			adsp_pipeline_sink(dsp, sink);
		}
		frame_count -= 1;
	}
	printf("Done\n");
	exit(failed);
}

void do_dsp() {
	adsp_pipeline_t* dsp = adsp_auto_pipeline_init();
	PAR_FUNCS(
		PFUNC(test, dsp),
		PFUNC(adsp_auto_pipeline_main, dsp)
	);
}

int main(){
	printf("Starting DSP test\n");

	do_dsp();
}
