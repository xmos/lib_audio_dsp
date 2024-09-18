#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <xcore/hwtimer.h>
#include <xcore/parallel.h>
#include <xcore/channel.h>
#include "dsp/td_block_fir.h"

#include "test_0.h"
#include "test_1.h"
#include "test_2.h"
#include "test_3.h"
#include "test_4.h"
#include "test_5.h"
#include "test_6.h"
#include "test_7.h"
#include "test_8.h"
#include "test_9.h"
#include "test_10.h"
#include "test_11.h"
#include "test_12.h"
#include "test_13.h"
#include "test_14.h"
#include "test_15.h"

#define WORKER_THREAD_COUNT 8

DECLARE_JOB(worker, (chanend_t, int32_t*, uint32_t, td_block_fir_filter_t*, int32_t*, uint32_t, td_block_fir_filter_t*));
void worker(chanend_t c, 
    int32_t * data0,
    uint32_t data0_elements, 
    td_block_fir_filter_t * f0, 
    int32_t * data1,
    uint32_t data1_elements, 
    td_block_fir_filter_t * f1)
{   
    td_block_fir_data_t d0, d1;
    td_block_fir_data_init(&d0, data0, data0_elements * sizeof(int32_t));
    td_block_fir_data_init(&d1, data1, data1_elements * sizeof(int32_t));
    memset(data0, 0, data0_elements *sizeof(int32_t));    
    memset(data1, 0, data1_elements*sizeof(int32_t));

    while(1){
        int32_t audio_channel_0[TD_BLOCK_FIR_LENGTH];
        int32_t audio_channel_1[TD_BLOCK_FIR_LENGTH];

        chan_in_buf_word(c, audio_channel_0, TD_BLOCK_FIR_LENGTH);
        chan_in_buf_word(c, audio_channel_1, TD_BLOCK_FIR_LENGTH);

        td_block_fir_add_data(&d0, audio_channel_0);
        td_block_fir_compute(audio_channel_0, &d0, f0);

        td_block_fir_add_data(&d1, audio_channel_1);
        td_block_fir_compute(audio_channel_1, &d1, f1);

        chan_out_buf_word(c, audio_channel_0, TD_BLOCK_FIR_LENGTH);
        chan_out_buf_word(c,audio_channel_1, TD_BLOCK_FIR_LENGTH);
    }
}


void worker_tile(chanend_t c[WORKER_THREAD_COUNT]){
    int32_t mem_0[test_0_DATA_BUFFER_ELEMENTS];
    int32_t mem_1[test_1_DATA_BUFFER_ELEMENTS];
    int32_t mem_2[test_2_DATA_BUFFER_ELEMENTS];
    int32_t mem_3[test_3_DATA_BUFFER_ELEMENTS];
    int32_t mem_4[test_4_DATA_BUFFER_ELEMENTS];
    int32_t mem_5[test_5_DATA_BUFFER_ELEMENTS];
    int32_t mem_6[test_6_DATA_BUFFER_ELEMENTS];
    int32_t mem_7[test_7_DATA_BUFFER_ELEMENTS];
    int32_t mem_8[test_8_DATA_BUFFER_ELEMENTS];
    int32_t mem_9[test_9_DATA_BUFFER_ELEMENTS];
    int32_t mem_10[test_10_DATA_BUFFER_ELEMENTS];
    int32_t mem_11[test_11_DATA_BUFFER_ELEMENTS];
    int32_t mem_12[test_12_DATA_BUFFER_ELEMENTS];
    int32_t mem_13[test_13_DATA_BUFFER_ELEMENTS];
    int32_t mem_14[test_14_DATA_BUFFER_ELEMENTS];
    int32_t mem_15[test_15_DATA_BUFFER_ELEMENTS];

    PAR_JOBS (
            PJOB(worker, (c[0], mem_0, test_0_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_0,
                                mem_1, test_1_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_1)),
            PJOB(worker, (c[1], mem_2, test_2_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_2,
                                mem_3, test_3_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_3)),
            PJOB(worker, (c[2], mem_4, test_4_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_4,
                                mem_5, test_5_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_5)),
            PJOB(worker, (c[3], mem_6, test_6_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_6,
                                mem_7, test_7_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_7)),
            PJOB(worker, (c[4], mem_8, test_8_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_8,
                                mem_9, test_9_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_9)),
            PJOB(worker, (c[5], mem_10, test_10_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_10,
                                mem_11, test_11_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_11)),
            PJOB(worker, (c[6], mem_12, test_12_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_12,
                                mem_13, test_13_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_13)),
            PJOB(worker, (c[7], mem_14, test_14_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_14,
                                mem_15, test_15_DATA_BUFFER_ELEMENTS, &td_block_fir_filter_test_15))
    );
}

void audio_gen(chanend_t c[WORKER_THREAD_COUNT]){

    hwtimer_t SysTimer = hwtimer_alloc();
    uint32_t from, to;

    const uint32_t loops = 128;
    from = hwtimer_get_time(SysTimer);

    int32_t buffer0[TD_BLOCK_FIR_LENGTH];
    int32_t buffer1[TD_BLOCK_FIR_LENGTH];
    for(int i=0;i<loops;i++){
        //send the unfiltered samples
        for(int worker_idx=0;worker_idx < WORKER_THREAD_COUNT;worker_idx++){
            chan_out_buf_word(c[worker_idx], buffer0, TD_BLOCK_FIR_LENGTH);
            chan_out_buf_word(c[worker_idx], buffer1, TD_BLOCK_FIR_LENGTH);
        }

        //recieve the filtered samples
        for(int worker_idx=0;worker_idx < WORKER_THREAD_COUNT;worker_idx++){
            chan_in_buf_word(c[worker_idx], buffer0, TD_BLOCK_FIR_LENGTH);
            chan_in_buf_word(c[worker_idx], buffer1, TD_BLOCK_FIR_LENGTH);
        }
    }
    to = hwtimer_get_time(SysTimer);
    uint32_t elapsed = to - from;
    printf("elapsed: %lu\n", elapsed/loops);
    exit(1);
}