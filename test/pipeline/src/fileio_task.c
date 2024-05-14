// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <xcore/assert.h>
#include <xcore/channel.h>
#include <xcore/chanend.h>
#include "fileio.h"
#include "wav_utils.h"
#include "app_dsp.h"
#include "adsp_control.h"
#include "adsp_instance_id_auto.h"
#include "cmds.h"
#include "cmd_offsets.h"
#include "fileio_task.h"


#define FILEREAD_CHUNK_SIZE (1024)

/// @brief Read a chunk of data from the input file.
/// Takes care of different bit-depths and reads the data in left justified 32bit format.
/// @param input_file Input file handle
/// @param input Buffer to read into
/// @param input_header_struct wav header structure
static void read_input_frame(file_t *input_file, int32_t *input, wav_header *input_header_struct)
{
    unsigned bytes_per_frame = wav_get_num_bytes_per_frame(input_header_struct);

    if(input_header_struct->bit_depth == 32)
    {
        // Read directly to input
        file_read (input_file, (uint8_t*)&input[0], bytes_per_frame * FILEREAD_CHUNK_SIZE); // Read in FILEREAD_CHUNK_SIZE chunks otherwise it's really slow
    }
    else if(input_header_struct->bit_depth == 16)
    {
        int16_t input16[ FILEREAD_CHUNK_SIZE * MAX_CHANNELS];
        file_read (input_file, (uint8_t*)&input16[0], bytes_per_frame * FILEREAD_CHUNK_SIZE);
        for(int i=0; i<FILEREAD_CHUNK_SIZE*input_header_struct->num_channels; i++)
        {
            input[i] = input16[i] << 16;
        }
    }
    else if(input_header_struct->bit_depth == 24)
    {
        uint8_t input8[FILEREAD_CHUNK_SIZE * MAX_CHANNELS * 3];
        uint8_t *inptr = input8;
        file_read (input_file, (uint8_t*)&input8[0], bytes_per_frame * FILEREAD_CHUNK_SIZE);
        for(int i=0; i<FILEREAD_CHUNK_SIZE; i++)
        {
            for(int ch=0; ch<input_header_struct->num_channels; ch++)
            {
                int32_t temp = 0;
                for(int b=0; b<3; b++)
                {
                    uint32_t val = (uint32_t)(*inptr++);
                    temp = temp | ((uint32_t)val << ((b+1)*8));
                }
                input[i*input_header_struct->num_channels + ch] = temp;
            }
        }
    }
    else
    {
        printf("ERROR: Unsupported bit depth %d\n", input_header_struct->bit_depth);
        assert(0);
    }
}

static void prepare_input(int32_t** dsp_input, int32_t* data, int n_ch, int frame_size) {
    for(int f = 0; f < frame_size; ++f) {
        for(int ch=0; ch < n_ch; ++ch) {
            dsp_input[ch][f] = *data;
            ++data;
        }
    }
}


static void prepare_output(int32_t* data, int32_t** dsp_input, int n_ch, int frame_size) {
    for(int f = 0; f < frame_size; ++f) {
        for(int ch=0; ch < n_ch; ++ch) {
            *data = dsp_input[ch][f];
            ++data;
        }
    }
}


/// @brief Task responsible for sending data read from a wav file to the pipeline and writing the output received
/// from the pipeline to another file. File operations are done using xscope_fileio functions.
/// @param c_control Unused for now. Will be used for control in future.
//void fileio_task(adsp_pipeline_t * m_dsp, chanend_t c_control)
void fileio_task(chanend_t c_control)
{
    printf("In test app!\n");
    test_config_t test_config = {0};
    parse_args("args.txt", &test_config);

    printf("After parse_args: Input file %s, Output file %s, num channels %d\n", test_config.input_filename, test_config.output_filename, test_config.num_output_channels);

    assert(test_config.input_filename != NULL);
    assert(test_config.output_filename != NULL);

    chan_out_word(c_control, 0x12345678);

    file_t input_file, output_file;
    int ret = file_open(&input_file, test_config.input_filename, "rb");
    assert((!ret) && "Failed to open file");

    ret = file_open(&output_file, test_config.output_filename, "wb");
    assert((!ret) && "Failed to open file");

    wav_header input_header_struct, output_header_struct;
    unsigned input_header_size;
    if(get_wav_header_details(&input_file, &input_header_struct, &input_header_size) != 0){
        printf("error in get_wav_header_details()\n");
        _Exit(1);
    }

    if(test_config.num_output_channels == 0)
    {
        test_config.num_output_channels = input_header_struct.num_channels;
    }

    assert(input_header_struct.num_channels <= MAX_CHANNELS);
    assert(test_config.num_output_channels <= MAX_CHANNELS);

    unsigned frame_count = wav_get_num_frames(&input_header_struct);
    // Calculate number of frames in the wav file
    unsigned block_count = frame_count / (FILEREAD_CHUNK_SIZE);
    xassert(0 != block_count);

    wav_form_header(&output_header_struct,
            input_header_struct.audio_format,
            test_config.num_output_channels,
            input_header_struct.sample_rate,
            32, // Output always 32 bits
            block_count*FILEREAD_CHUNK_SIZE);

    file_write(&output_file, (uint8_t*)(&output_header_struct),  WAV_HEADER_BYTES);

    unsigned bytes_per_frame = wav_get_num_bytes_per_frame(&input_header_struct);

    printf("Num input channels = %d\n", input_header_struct.num_channels);
    printf("Num output channels = %d\n", test_config.num_output_channels);
    printf("Discard frames = %d\n", test_config.num_discard_frames);
    printf("bytes_per_frame = %d\n", bytes_per_frame);
    printf("Block count = %d\n", block_count);
    printf("Frame size = %d\n", app_dsp_frame_size());
    int32_t input[ FILEREAD_CHUNK_SIZE * MAX_CHANNELS] = {0}; // Array for storing interleaved input read from wav file
    int32_t output[FILEREAD_CHUNK_SIZE * MAX_CHANNELS] = {0};
    xassert(FILEREAD_CHUNK_SIZE % app_dsp_frame_size() ==0); // frame size must be a factor of read size

    int32_t** dsp_input = malloc(sizeof(int32_t*) * input_header_struct.num_channels);
    for(int i = 0; i < input_header_struct.num_channels; ++i) {
        dsp_input[i] = malloc(sizeof(int32_t) * app_dsp_frame_size());
    }
    int32_t** dsp_output = malloc(sizeof(int32_t*) * test_config.num_output_channels);
    for(int i = 0; i < test_config.num_output_channels; ++i) {
        dsp_output[i] = malloc(sizeof(int32_t) * app_dsp_frame_size());
    }

    int discard = test_config.num_discard_frames;
    for(int i=0; i<block_count; i++)
    {
        printf("block %d\n", i);
        read_input_frame(&input_file, input, &input_header_struct);

        int32_t* block_input = input;
        int32_t* block_output = output;

        // 1. discard initial outputs which will be 0
        for(int d = 0; d < discard; d++) {
            prepare_input(dsp_input, block_input, input_header_struct.num_channels, app_dsp_frame_size());
            app_dsp_source(dsp_input);
            block_input += input_header_struct.num_channels * app_dsp_frame_size();

            // discard these outputs
            app_dsp_sink(dsp_output);
        }

        // 2. source and sink pipeline
        int loop_n = FILEREAD_CHUNK_SIZE/app_dsp_frame_size();//- test_config.num_discard_frames;
        for(int j=0; j < loop_n; j++)
        {
            prepare_input(dsp_input, block_input, input_header_struct.num_channels, app_dsp_frame_size());
            app_dsp_source(dsp_input);
            app_dsp_sink(dsp_output);
            prepare_output(block_output, dsp_output, test_config.num_output_channels, app_dsp_frame_size());

            block_input += input_header_struct.num_channels * app_dsp_frame_size();
            block_output += test_config.num_output_channels * app_dsp_frame_size();
        }

        // 3. read only, should empty the pipeline
        for(int d = 0; d < discard; d++) {
            app_dsp_sink(dsp_output);
            prepare_output(block_output, dsp_output, test_config.num_output_channels, app_dsp_frame_size());
            block_output += test_config.num_output_channels * app_dsp_frame_size();
        }

        file_write(&output_file, (uint8_t*)&output[0], test_config.num_output_channels * FILEREAD_CHUNK_SIZE * sizeof(int32_t));
        discard = 0; // synchronisation complete
    }
    file_close(&input_file);
    file_close(&output_file);
    shutdown_session();
    printf("DONE\n");
    _Exit(0);
}
