#include "frequency_domain_pipeline.h"

void buffer_process(int32_t **input, int32_t **output, void *app_data_state)
{
    buffer_state_t *state = app_data_state;

    int32_t* d = state->buffer_data;
    int32_t overlap = state->buffer_len - state->frame_size;

    // roll buffer
    memcpy(d, d[state->frame_size], overlap*sizeof(int32_t));

    // add new samples
    memcpy(d + state->frame_size, samples_in, state->frame_size*sizeof(int32_t));

    output[0] = d;

}

void fft_process(int32_t **input, int32_t **output, void *app_data_state)
{
    fft_state_t *state = app_data_state;
    // put signal int32_t array into bfp
    bfp_s32_init(state->signal, input[0], state->exp, state->nfft, 1);
    // do the FFT
    bfp_complex_s32_t * c = bfp_fft_forward_mono(state->signal);

    output[0] = c;

}

void bfp_mult_process(int32_t **input, int32_t **output, void *app_data_state)
{
    // if everything is compatible, this should just work?
    bfp_complex_s32_mul(output[0], input[0], input[1] );

    output[0] = c;

}

void ifft_process(int32_t **input, int32_t **output, void *app_data_state)
{
    ifft_state_t *state = app_data_state;
    bfp_s32_t *time_domain_result = bfp_fft_inverse_mono(input[0]);

    //denormalise and escape BFP domain
    bfp_s32_use_exponent(time_domain_result, state->exp);
    output[0] = time_domain_result->data;
}


void wola_rect_process(int32_t **input, int32_t **output, void *app_data_state)
{
    wola_state_t *state = state;
    // just point to the start of the valid data, length should just work out
    output[0] = &(input[0][state->win_start]);
}



void init_dsp(){
    int32_t __attribute__((aligned (8))) magic_shared_memory[512] = {0};

    static buffer_config_t config1 = {};
    static buffer_state_t state1 = {.buffer_len = 512, .buffer_step = 256, .buffer_data = &magic_shared_memory};

    static fft_config_t config1 = {};
    static fft_state_t state1 = {.nfft = 512, .data = &magic_shared_memory, .exp = 27};

}


void dsp_auto_thread0(chanend_t* c_source, chanend_t* c_dest, module_instance_t** modules) {

	int32_t edge0[256] = {0}; // in
    int32_t edge1[1] = {0}; //don't ask
	int32_t edge2[512] = {0}; // buffered
	bfp_complex_s32_t edge3[1] = {0}; // fft'd
	bfp_complex_s32_t edge4[1] = {0}; // mult'd
	int32_t edge5[512] = {0}; // ifft'd
	int32_t edge6[256] = {0}; // wola'd
    bfp_complex_s32_t edge7[1] = {0}; // saved filter


    int32_t* stage_1_input[] = {edge1};  // buffer
	int32_t* stage_1_output[] = {edge2};
	int32_t* stage_2_input[] = {edge2}; // fft
	int32_t* stage_2_output[] = {edge3};
	int32_t* stage_3_input[] = {edge3, edge7}; //bfp mult
	int32_t* stage_3_output[] = {edge4};
	int32_t* stage_4_input[] = {edge4}; // ifft
	int32_t* stage_4_output[] = {edge5};
	int32_t* stage_5_input[] = {edge4}; // wola
	int32_t* stage_5_output[] = {edge5};

	buffer_process(
		stage_1_input,
		stage_1_output,
		modules[1]->state);
	fft_process(
		stage_2_input,
		stage_2_output,
		modules[2]->state);
	bfp_mult_process(
		stage_3_input,
		stage_3_output,
		modules[3]->state);
	ifft_process(
		stage_4_input,
		stage_4_output,
		modules[4]->state);
	wola_rect_process(
		stage_5_input,
		stage_5_output,
		modules[5]->state);

    chan_out_buf_word(c_dest[0], (void*)edge6, 256);
}

void adsp_auto_pipeline_main(adsp_pipeline_t* adsp) {
	module_instance_t* thread_0_modules[] = {
		&adsp->modules[0],
		&adsp->modules[1],
		&adsp->modules[2],
		&adsp->modules[3],
		&adsp->modules[4],
		&adsp->modules[5],

	};
	chanend_t thread_0_inputs[] = {
		adsp->p_in[0].end_b};
	chanend_t thread_0_outputs[] = {
		adsp->p_link[0].end_a};

	PAR_JOBS(
		PJOB(dsp_auto_thread0, (thread_0_inputs, thread_0_outputs, thread_0_modules)),
	);
}