
#include <stages/buffers.h>
#include <xcore/parallel.h>
#include <stdio.h>
#include "dsp_pipeline.h"

#define MAX_(A, B) ((A) > (B) ? (A) : (B))

//// TODO use real stages

typedef struct {
	int temp;
} reverb_stage_t;

static void reverb_process(void* input, void* output, reverb_stage_t* state) {
	int32_t** i = input;
	int32_t** o = output;
	printf("reverb\t\t%ld %ld %ld %ld\n", i[0][0], i[0][1], i[0][2], i[0][3]);
	for(int j = 0; j < 4; ++j) {
		o[0][j] = i[0][j];
	}
}

typedef struct {
	int temp;
} biquad_stage_t;

static void biquad_process(void* input, void* output, biquad_stage_t* state) {
	int32_t** i = input;
	int32_t** o = output;
	printf("biquad %ld\n", i[0][0]);
	o[0][0] = i[0][0];
}

typedef struct {
	int temp;
} mixer_stage_t;

static void mixer_process(void* input, void* output, mixer_stage_t* state) {
	int32_t** i = input;
	int32_t** o = output;
	printf("mixer %ld %ld\n", i[0][0], i[1][0]);
	o[0][0]=i[0][0];
}

// Static Vars

static char arr_buffer_input[MAX_(4, 4)];
static buffer_1to1_t buffer_input = BUFFER_1TO1_FULL(arr_buffer_input, 4, 4);
static char arr_buffer_b[MAX_(4, 4)];
static buffer_1to1_t buffer_b = BUFFER_1TO1_FULL(arr_buffer_b, 4, 4);
static char arr_buffer_a[MAX_(4, 16)];
static buffer_smalltobig_t buffer_a = BUFFER_SMALLTOBIG_FULL(arr_buffer_a, 4, 16);
static char arr_buffer_output[MAX_(16, 4)];
static buffer_bigtosmall_t buffer_output = BUFFER_BIGTOSMALL_FULL(arr_buffer_output, 16, 4);

static mixer_stage_t stage_mix;
static biquad_stage_t stage_filter;
static reverb_stage_t stage_reverb;


// thread 1


// other threads
__attribute__((noinline))
static void thread0(void* _unused) {
	(void)_unused;
	int32_t edge_input_0[1];
	int32_t edge_filter_0[1];
	int32_t edge_mix_0[1];
	int32_t edge_b_0[1];

	for(;;) {
	
		// control
		
		// buffer read
		buffer_1to1_read(&buffer_input, edge_input_0);
		buffer_1to1_read(&buffer_b, edge_b_0);
		
		// process
		{
			void* inputs[] = {edge_input_0};
			void* outputs[] = {edge_filter_0};
			biquad_process(inputs, outputs, &stage_filter);
		}
		{
			void* inputs[] = {edge_filter_0, edge_b_0};
			void* outputs[] = {edge_mix_0};
			mixer_process(inputs, outputs, &stage_mix);
		}
		
		// buffer write
		buffer_1to1_write(&buffer_b, edge_mix_0);
		buffer_smalltobig_write(&buffer_a, edge_mix_0);
	}
}
__attribute__((noinline))
static void thread1(void* _unused) {
	(void)_unused;
	int32_t edge_a_0[4];
	int32_t edge_reverb_0[4];

	for(;;) {
	
		// control
		
		// buffer read
		buffer_smalltobig_read(&buffer_a, edge_a_0);
		
		// process
		{
			void* inputs[] = {edge_a_0};
			void* outputs[] = {edge_reverb_0};
			reverb_process(inputs, outputs, &stage_reverb);
		}
		
		// buffer write
		buffer_bigtosmall_write(&buffer_output, edge_reverb_0);
	}
}

ADSP_SOURCE_FN_GROUP
static void adsp_generated_source(int channel, void* data) {
	switch(channel) {
		case 0: {
			buffer_1to1_write(&buffer_input, data);
		} break;
	}
}

ADSP_SINK_FN_GROUP
static void adsp_generated_sink(int channel, void* data) {
	switch(channel) {
		case 0: {
			buffer_bigtosmall_read(&buffer_output, data);
		} break;
	}
}


// init
adsp_generated_t* adsp_generated_init() {
	static void* stages[] = {
		&stage_mix,
		&stage_filter,
		&stage_reverb,
	};
	static adsp_generated_t dsp = {
		.source_fn=adsp_generated_source,
		.sink_fn=adsp_generated_sink,
		.stages=stages,
		.nstages=(sizeof(stages)/sizeof(*stages))
	};
	return &dsp;
}

// main
void adsp_generated_main() {

	PAR_FUNCS(
		PFUNC(thread0, NULL),
		PFUNC(thread1, NULL)
	);
}
