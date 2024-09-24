#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <xcore/hwtimer.h>
#include "test_0.h"

void foo(){

    int32_t __attribute__((aligned (8))) new_data[test_0_TD_BLOCK_LENGTH];

    //allocate a TD FIR for the example
    int32_t data[test_0_DATA_BUFFER_ELEMENTS];
    memset(data, 0, sizeof(data));
    fd_FIR_data_t fd_fir_data_test_0;

    fd_block_fir_data_init(&fd_fir_data_test_0, data, 
        test_0_FRAME_ADVANCE, 
        test_0_TD_BLOCK_LENGTH, 
        test_0_BLOCK_COUNT);

    for(int j=0;j<16;j++)
    {
        for(int i=0;i<test_0_FRAME_ADVANCE;i++)
            new_data[i] = rand()-rand();

        int32_t __attribute__((aligned (8))) fd_processed[test_0_TD_BLOCK_LENGTH] = {0};

        fd_block_fir_add_data(new_data, &fd_fir_data_test_0);

        fd_block_fir_compute(
            fd_processed,
            &fd_fir_data_test_0,
            &fd_fir_filter_test_0);

    }
}

int main() {
  foo();
}