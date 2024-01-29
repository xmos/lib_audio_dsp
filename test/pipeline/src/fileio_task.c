#include <string.h>
#include <stdbool.h>
#include <xcore/assert.h>
#include <xcore/channel.h>
#include <xcore/chanend.h>
#include "fileio.h"
#include "wav_utils.h"
#include "app_dsp.h"

#define FRAME_SIZE (1024)
#define MAX_CHANNELS (4)
void fileio_task(chanend_t c_control)
{
    const char *input_file_name = "input.wav";
    const char *output_file_name = "output.wav";

    file_t input_file, output_file;

    int ret = file_open(&input_file, input_file_name, "rb");
    assert((!ret) && "Failed to open file");
    
    printf("Opened input file\n");

    ret = file_open(&output_file, output_file_name, "wb");
    assert((!ret) && "Failed to open file");

    printf("Opened output file\n");

    wav_header input_header_struct, output_header_struct;
    unsigned input_header_size;
    if(get_wav_header_details(&input_file, &input_header_struct, &input_header_size) != 0){
        printf("error in get_wav_header_details()\n");
        _Exit(1);
    }

    unsigned frame_count = wav_get_num_frames(&input_header_struct);
    // Calculate number of frames in the wav file
    unsigned block_count = frame_count / FRAME_SIZE;

    wav_form_header(&output_header_struct,
            input_header_struct.audio_format,
            input_header_struct.num_channels,
            input_header_struct.sample_rate,
            input_header_struct.bit_depth,
            block_count*FRAME_SIZE);

    file_write(&output_file, (uint8_t*)(&output_header_struct),  WAV_HEADER_BYTES);

    unsigned bytes_per_frame = wav_get_num_bytes_per_frame(&input_header_struct);

    printf("Num Channels = %d\n",input_header_struct.num_channels);
    printf("bytes_per_frame = %d\n", bytes_per_frame);
    printf("Block count = %d\n", block_count);

    int32_t input[ FRAME_SIZE * MAX_CHANNELS] = {0}; // Array for storing interleaved input read from wav file
    int32_t output[FRAME_SIZE * MAX_CHANNELS] = {0};

    for(int i=0; i<block_count; i++)
    {
        printf("block %d\n", i);
        file_read (&input_file, (uint8_t*)&input[0], bytes_per_frame * FRAME_SIZE); // Read in FRAME_SIZE chunks otherwise it's really slow
        
        for(int i=0; i<FRAME_SIZE; i++)
        {
            app_dsp_source(&input[i * input_header_struct.num_channels]);
            app_dsp_sink(&output[i * input_header_struct.num_channels]);
        }

        file_write(&output_file, (uint8_t*)&output[0], input_header_struct.num_channels * FRAME_SIZE * sizeof(int32_t));
    }
    file_close(&input_file);
    file_close(&output_file);
    shutdown_session();
    printf("DONE\n");
    _Exit(0);
}
