// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "../autogen/dut.h"
#include "../autogen/dut_debug.h"
#include "ref_fir.h"

/*
This controls the number of extra block long the data is vs the length of the coefficient
array. It tests that with more data then you need the FIR is still correctly implemented.
*/
#define EXTRA_DATA_BLOCKS 2

/*
This tests for equlivance between the FD implementation and the TD reference.
It has an allowed error of 4 for mean abs error and abs mean error. 
*/
int run_test(void){
    //allocate a TD FIR for reference
    int32_t __attribute__((aligned (8))) data_debug[debug_dut_DATA_BUFFER_ELEMENTS];
    
    //allocate a TD BLOCK FIR for reference
    //one extra block for making the indexing easier and one as we are making TD_BLOCK_FIR_LENGTH outputs
    int32_t __attribute__((aligned (8))) block_data_td[dut_DATA_BUFFER_ELEMENTS+TD_BLOCK_FIR_LENGTH*EXTRA_DATA_BLOCKS];

    for( int b=0;b<EXTRA_DATA_BLOCKS;b++){

        int error_sum = 0;
        int abs_error_sum = 0;
        int count = 0;

        td_block_fir_data_t data;

        td_block_fir_data_init(&data, block_data_td, (dut_DATA_BUFFER_ELEMENTS + TD_BLOCK_FIR_LENGTH*b));

        memset(block_data_td, 0, sizeof(block_data_td));
        memset(data_debug, 0, sizeof(data_debug));

        int phases = td_block_fir_filter_dut.block_count;

        for(int j=0;j<phases+1;j++)
        {
            int32_t new_data[TD_BLOCK_FIR_LENGTH];
            for(int i=0;i<TD_BLOCK_FIR_LENGTH;i++)
                new_data[i] = (rand()-rand())>>1;

            int32_t td_processed[TD_BLOCK_FIR_LENGTH] = {0};
            int32_t fd_processed[TD_BLOCK_FIR_LENGTH] = {0};

            for(int i=0;i<TD_BLOCK_FIR_LENGTH;i++)
                td_processed[i] = td_reference_fir(new_data[i], &td_block_debug_fir_filter_dut, data_debug);

            td_block_fir_add_data(new_data, &data);

            td_block_fir_compute(
                fd_processed,
                &data,
                &td_block_fir_filter_dut);

            for(int i=0;i<TD_BLOCK_FIR_LENGTH;i++){
                int error = td_processed[i] - fd_processed[i];
                // printf("%ld %ld %.2f\n", td_processed[i], fd_processed[i], (float)td_processed[i] / (float)fd_processed[i]);
                error_sum += error;
                if(error < 0) error = -error;
                abs_error_sum += error;
                count++;
            }

        }
        float error_ave_abs =  (float)error_sum / count;
        if(error_ave_abs<0)error_ave_abs=-error_ave_abs;
        if (error_ave_abs > 4.0){
            printf("avg error:%f avg abs error:%f\n", (float)error_sum / count, (float)abs_error_sum / count);
            return 1;
        }
        if(((float)abs_error_sum / count) > 4.0){
            printf("avg error:%f avg abs error:%f\n", (float)error_sum / count, (float)abs_error_sum / count);
            return 1;
        }
    }
    return 0;
}

int main(void){
    return run_test();
}