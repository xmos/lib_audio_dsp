<%
	import itertools
	
	n_buffers = len(ir.config_struct.buffers)
	n_stages = len(ir.config_struct.stages)


	# Determine whether the par_funcs eol should have a comma
	par_funcs_eol = ["," for _ in threads]
	par_funcs_eol[-1] = ""
%>
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

%for line in buffer_defs:
${line}
%endfor

%for name, stage in ir.config_struct.stages.items():
static ${stage.type}_stage_t stage_${name};
%endfor


// thread 1


// other threads
%for thread_index, thread in enumerate(threads):
__attribute__((noinline))
static void thread${thread_index}(void* _unused) {
	(void)_unused;
	%for edge_definition in thread.edge_definitions:
	${edge_definition}
	%endfor

	for(;;) {
	
		// control
		
		// buffer read
		%for line in itertools.chain.from_iterable(thread.buffer_reads):
		${line}
		%endfor
		
		// process
		%for stage_process in thread.process:
		{
			%for line in stage_process:
			${line}
			%endfor
		}
		%endfor
		
		// buffer write
		%for line in itertools.chain.from_iterable(thread.buffer_writes):
		${line}
		%endfor
	}
}
%endfor

ADSP_SOURCE_FN_GROUP
static void adsp_generated_source(int channel, void* data) {
	switch(channel) {
		%for case, lines in source_cases:
		case ${case}: {
			%for line in lines:
			${line}
			%endfor
		} break;
		%endfor
	}
}

ADSP_SINK_FN_GROUP
static void adsp_generated_sink(int channel, void* data) {
	switch(channel) {
		%for case, lines in sink_cases:
		case ${case}: {
			%for line in lines:
			${line}
			%endfor
		} break;
		%endfor
	}
}


// init
adsp_generated_t* adsp_generated_init() {
	static void* stages[] = {
		%for name, stage in ir.config_struct.stages.items():
		&stage_${name},
		%endfor
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
		%for thread_idx, par_eol in zip(range(len(threads)), par_funcs_eol):
		PFUNC(thread${thread_idx}, NULL)${par_eol}
		%endfor
	);
}
