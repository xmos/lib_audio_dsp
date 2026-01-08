// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <xcore/hwtimer.h>
#include "../autogen/dut.h"
#include "../autogen/dut_debug.h"

#include "ref_fir.h"

/*
This tests for equlivance between the FD implementation and the TD reference.
It has an allowed error of 32 for mean abs error and abs mean error. 
*/
int run_test(void){

    int32_t __attribute__((aligned (8))) data[dut_DATA_BUFFER_ELEMENTS];
    int32_t __attribute__((aligned (8))) new_data[dut_TD_BLOCK_LENGTH];
    int32_t __attribute__((aligned (8))) data_td[debug_dut_DATA_BUFFER_ELEMENTS];
    
    memset(new_data, 0, sizeof(new_data));
    memset(data_td, 0, sizeof(data_td));
    memset(data, 0, sizeof(data));
    fd_fir_data_t fd_fir_data_dut;
    fd_block_fir_data_init(&fd_fir_data_dut, data, 
        dut_FRAME_ADVANCE, 
        dut_TD_BLOCK_LENGTH, 
        dut_BLOCK_COUNT);

    int error_sum = 0;
    int abs_error_sum = 0;
    int count = 0;

    int32_t frame_overlap[dut_FRAME_OVERLAP];
    memset(frame_overlap, 0, sizeof(frame_overlap));
    for(int j=0;j<dut_BLOCK_COUNT + 2;j++)
    {
        for(int i=0;i<dut_FRAME_ADVANCE;i++)
            new_data[i] = rand()-rand();

        int32_t td_processed[dut_FRAME_ADVANCE + dut_FRAME_OVERLAP];

        memcpy(td_processed, frame_overlap, sizeof(frame_overlap));
        for(int i=0;i<dut_FRAME_ADVANCE;i++)
            td_processed[i+dut_FRAME_OVERLAP] = td_reference_fir(new_data[i], &td_block_debug_fir_filter_dut, data_td);
        memcpy(frame_overlap, td_processed + dut_FRAME_ADVANCE, sizeof(frame_overlap));

        int32_t __attribute__((aligned (8))) fd_processed[dut_TD_BLOCK_LENGTH] = {0};
        fd_block_fir_add_data(new_data, &fd_fir_data_dut);
        fd_block_fir_compute(
            fd_processed,
            &fd_fir_data_dut,
            &fd_fir_filter_dut);

        for(int i=0;i<dut_FRAME_ADVANCE + dut_FRAME_OVERLAP;i++){
            int error = td_processed[i] - fd_processed[i];
            // printf("%2d td:%12ld fd:%12ld error:%d\n", i, td_processed[i], fd_processed[i], error);
            error_sum += error;
            if(error < 0) error = -error;
            abs_error_sum += error;
            count++;
        }

    }
    float error_ave_abs =  (float)error_sum / count;
    if(error_ave_abs<0)error_ave_abs=-error_ave_abs;
    if (error_ave_abs > 32.0){
        printf("avg error:%f avg abs error:%f dut_TD_BLOCK_LENGTH:%d dut_BLOCK_COUNT:%d DATA_BUFFER_ELEMENTS:%d\n", (float)error_sum / count, (float)abs_error_sum / count, dut_TD_BLOCK_LENGTH, dut_BLOCK_COUNT, debug_dut_DATA_BUFFER_ELEMENTS);
        return 1;
    }
    if(((float)abs_error_sum / count) > 32.0){
        printf("avg error:%f avg abs error:%f dut_TD_BLOCK_LENGTH:%d dut_BLOCK_COUNT:%d DATA_BUFFER_ELEMENTS:%d\n", (float)error_sum / count, (float)abs_error_sum / count, dut_TD_BLOCK_LENGTH, dut_BLOCK_COUNT, debug_dut_DATA_BUFFER_ELEMENTS);
        return 1;
    }
    return 0;
}

int main() {
  return run_test();
}