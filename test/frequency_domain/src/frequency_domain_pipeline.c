
#include <stages/adsp_pipeline.h>
#include <stages/adsp_control.h>
#include <xcore/select.h>
#include <xcore/channel.h>
#include <xcore/assert.h>
#include <xcore/hwtimer.h>
#include <xcore/thread.h>
#include <print.h>
#include "cmds.h" // Autogenerated
#include "cmd_offsets.h" // Autogenerated
#include <stages/bump_allocator.h>
#include <dsp/signal_chain.h>

static adsp_controller_t* m_control;
#include <stages/pipeline.h>
#include <stages/dsp_thread.h>

#include <stages/buffer.h>
#include <stages/fft.h>
#include <stages/ifft.h>
#include <stages/wola_rect.h>


adsp_pipeline_t * adsp_auto_pipeline_init() {

	// Copied from app_simple_audio_dsp_integration
	static adsp_pipeline_t adsp_auto;
	static adsp_controller_t adsp_auto_controller;
	m_control = &adsp_auto_controller;
	static channel_t adsp_auto_in_chans[1];
	static channel_t adsp_auto_out_chans[1];
	static channel_t adsp_auto_link_chans[0];
	static module_instance_t adsp_auto_modules[6];
	static adsp_mux_elem_t adsp_auto_in_mux_cfgs[] = {
		{ .channel_idx = 0, .data_idx = 0, .frame_size = 256},
	};
	static adsp_mux_elem_t adsp_auto_out_mux_cfgs[] = {
		{ .channel_idx = 0, .data_idx = 0, .frame_size = 256},
	};
	adsp_auto.input_mux.n_chan = 1;
	adsp_auto.input_mux.chan_cfg = (adsp_mux_elem_t *) adsp_auto_in_mux_cfgs;
	adsp_auto.output_mux.n_chan = 1;
	adsp_auto.output_mux.chan_cfg = (adsp_mux_elem_t *) adsp_auto_out_mux_cfgs;
	adsp_auto_in_chans[0] = chan_alloc();
	adsp_auto_out_chans[0] = chan_alloc();
	adsp_auto.p_in = (channel_t *) adsp_auto_in_chans;
	adsp_auto.n_in = 1;
	adsp_auto.p_out = (channel_t *) adsp_auto_out_chans;
	adsp_auto.n_out = 1;
	adsp_auto.p_link = (channel_t *) adsp_auto_link_chans;
	adsp_auto.n_link = 0;
	adsp_auto.modules = adsp_auto_modules;
	adsp_auto.n_modules = 6;
	static pipeline_config_t config0 = { .checksum = {252, 84, 60, 61, 184, 125, 172, 129, 216, 0, 154, 1, 157, 112, 69, 119} };

            static pipeline_state_t state0;
            static uint8_t memory0[PIPELINE_STAGE_REQUIRED_MEMORY];
            static adsp_bump_allocator_t allocator0 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory0);

            adsp_auto.modules[0].state = (void*)&state0;

            // Control stuff
            adsp_auto.modules[0].control.id = 0;
            adsp_auto.modules[0].control.config_rw_state = config_none_pending;
            
                adsp_auto.modules[0].control.config = (void*)&config0;
                adsp_auto.modules[0].control.module_type = e_dsp_stage_pipeline;
                adsp_auto.modules[0].control.num_control_commands = NUM_CMDS_PIPELINE;
                pipeline_init(&adsp_auto.modules[0], &allocator0, 0, 0, 0, 1);
	static dsp_thread_config_t config1 = {  };

            static dsp_thread_state_t state1;
            static uint8_t memory1[DSP_THREAD_STAGE_REQUIRED_MEMORY];
            static adsp_bump_allocator_t allocator1 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory1);

            adsp_auto.modules[1].state = (void*)&state1;

            // Control stuff
            adsp_auto.modules[1].control.id = 1;
            adsp_auto.modules[1].control.config_rw_state = config_none_pending;
            
                adsp_auto.modules[1].control.config = (void*)&config1;
                adsp_auto.modules[1].control.module_type = e_dsp_stage_dsp_thread;
                adsp_auto.modules[1].control.num_control_commands = NUM_CMDS_DSP_THREAD;
                dsp_thread_init(&adsp_auto.modules[1], &allocator1, 1, 0, 0, 1);

//#################################################################################################
	// Exciting new stuff from here

    // static buffer_config_t config2 = {};
    static buffer_state_t state2;
	static buffer_constants_t constants2 = {.buffer_len = 512};
	static uint8_t memory2[BUFFER_STAGE_REQUIRED_MEMORY(1, 512)];
	static adsp_bump_allocator_t allocator2 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory2);
	adsp_auto.modules[2].state = (void*)&state2;
	adsp_auto.modules[2].constants = (void*)&constants2;

	buffer_init(&adsp_auto.modules[2], &allocator2, 2, 1, 1, 256);


    // static fft_config_t config3 = {};
    static fft_state_t state3 = {};
	static fft_constants_t constants3 = {.nfft = 512, .exp=27};
	static uint8_t memory3[FFT_STAGE_REQUIRED_MEMORY(1)];
	static adsp_bump_allocator_t allocator3 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory3);
	adsp_auto.modules[3].state = (void*)&state3;
	adsp_auto.modules[3].constants = (void*)&constants3;

	fft_init(&adsp_auto.modules[3], &allocator3, 2, 1, 1, 256);

    // static ifft_config_t config5 = {};
    static ifft_state_t state5 = {};
	static ifft_constants_t constants5 = {.nfft = 512, .exp=27};
	static uint8_t memory5[IFFT_STAGE_REQUIRED_MEMORY(1)];
	static adsp_bump_allocator_t allocator5 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory5);
	adsp_auto.modules[5].state = (void*)&state5;
	adsp_auto.modules[5].constants = (void*)&constants5;

	ifft_init(&adsp_auto.modules[5], &allocator5, 2, 1, 1, 256);


    // static wola_rect_config_t config6 = {};
    static wola_rect_state_t state6 = {};
	static wola_rect_constants_t constants6 = {.win_start=256};
	static uint8_t memory6[WOLA_RECT_STAGE_REQUIRED_MEMORY];
	static adsp_bump_allocator_t allocator6 = ADSP_BUMP_ALLOCATOR_INITIALISER(memory6);
	adsp_auto.modules[6].state = (void*)&state6;
	adsp_auto.modules[6].constants = (void*)&constants6;

	wola_rect_init(&adsp_auto.modules[6], &allocator6, 2, 1, 1, 256);

	// adsp_controller_init(&adsp_auto_controller, &adsp_auto);
	printf("initialised\n");
	return &adsp_auto;
}


void dsp_auto_thread0(chanend_t* c_source, chanend_t* c_dest, module_instance_t** modules) {
	local_thread_mode_set_bits(thread_mode_high_priority);

	int32_t edge0[256] = {0}; // in
	int32_t* edge2[1] = {NULL}; // buffered
	bfp_complex_s32_t* edge3[1] = {NULL}; // fft'd
	int32_t* edge5[1] = {NULL}; // ifft'd
	int32_t edge6[256]; // wola'd -> last FD related stage so memcopy
    // int32_t edge1[1] = {0}; //don't ask
	// bfp_complex_s32_t edge4[1] = {0}; // mult'd
    // bfp_complex_s32_t edge7[1] = {0}; // saved filter

    int32_t* stage_2_input[] = {edge0};  // buffer
	int32_t** stage_2_output[] = {edge2};
	int32_t** stage_3_input[] = {edge2}; // fft
	bfp_complex_s32_t** stage_3_output[] = {edge3};
	bfp_complex_s32_t** stage_5_input[] = {edge3}; // ifft
	int32_t** stage_5_output[] = {edge5};
	int32_t** stage_6_input[] = {edge5}; // wola
	int32_t* stage_6_output[] = {edge6};

	// int32_t* stage_3_input[] = {edge3, edge7}; //bfp mult
	// int32_t* stage_3_output[] = {edge4};
	while(1) {
	int read_count = 1;
	SELECT_RES(
		CASE_THEN(c_source[0], case_0),
		DEFAULT_THEN(do_control)
	) {
		case_0: {
			chan_in_buf_word(c_source[0], (uint32_t*)edge0, 256); for(int idx = 0; idx < 256; ++idx) edge0[idx] = adsp_from_q31(edge0[idx]);
			if(!--read_count) break;
			else continue;
		}
		do_control: {
		// no control
		continue; }
	}
	printf("input %ld %ld %ld %ld %ld\n", edge0[0], edge0[1], edge0[2], edge0[3], edge0[4]);

	printf("buffering\n");
	buffer_process(
		stage_2_input,
		stage_2_output,
		modules[2]->state);
	printf("s2 out addr: %p\n", stage_2_output[0][0]);
	printf("s3 in addr: %p\n", stage_3_input[0][0]);
	printf("ffting\n");
	fft_process(
		stage_3_input,
		stage_3_output,
		modules[3]->state);
    printf("s3 output data addr: %p\n", stage_3_output[0][0]->data);
    printf("s5 in output data addr: %p\n", stage_5_input[0][0]->data);

	// bfp_mult_process(
	// 	stage_3_input,
	// 	stage_3_output,
	// 	modules[3]->state);
	printf("iffting\n");
	ifft_process(
		stage_5_input,
		stage_5_output,
		modules[5]->state);
	printf("wolaing\n");
	printf("edge6 0: %p\n", &edge6[0]);
	wola_rect_process(
		stage_6_input,
		stage_6_output,
		modules[6]->state);

	printf("output %ld %ld %ld %ld %ld\n", edge6[0], edge6[1], edge6[2], edge6[3], edge6[4]);

	for(int idx = 0; idx < 256; ++idx) edge6[idx] = adsp_to_q31(edge6[idx]); chan_out_buf_word(c_dest[0], (uint32_t*)edge6, 256);
	printf("end of looop\n");
	}
}

void adsp_auto_pipeline_main(adsp_pipeline_t* adsp) {
	module_instance_t* thread_0_modules[] = {
		&adsp->modules[0],
		&adsp->modules[1],
		&adsp->modules[2],
		&adsp->modules[3],
		&adsp->modules[4],
		&adsp->modules[5],
		&adsp->modules[6],

	};
	chanend_t thread_0_inputs[] = {
		adsp->p_in[0].end_b};
	chanend_t thread_0_outputs[] = {
		adsp->p_out[0].end_a};
	PAR_JOBS(
		PJOB(dsp_auto_thread0, (thread_0_inputs, thread_0_outputs, thread_0_modules))
	);
}