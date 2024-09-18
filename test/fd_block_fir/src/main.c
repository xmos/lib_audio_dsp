#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <xcore/hwtimer.h>
#include "test_0.h"

void foo(){

    int32_t __attribute__((aligned (8))) new_data[test_0_TD_BLOCK_LENGTH];
    
    //allocate a TD FIR for reference
    int32_t __attribute__((aligned (8))) data_td[debug_test_0_DATA_BUFFER_ELEMENTS];
    memset(data_td, 0, sizeof(data_td));

    //allocate a TD FIR for the example
    int32_t data[test_0_DATA_BUFFER_ELEMENTS];
    memset(data, 0, sizeof(data));
    fd_FIR_data_t fd_fir_data_test_0;
    fd_block_fir_data_init(&fd_fir_data_test_0, data, 
        test_0_FRAME_ADVANCE, 
        test_0_TD_BLOCK_LENGTH, 
        test_0_BLOCK_COUNT);

    int error_sum = 0;
    int abs_error_sum = 0;
    int count = 0;
    int32_t c = 0;
    for(int j=0;j<11;j++)
    {
        for(int i=0;i<test_0_FRAME_ADVANCE;i++)
            new_data[i] = rand()-rand();

        int32_t td_processed[test_0_FRAME_ADVANCE];

        for(int i=0;i<test_0_FRAME_ADVANCE;i++)
            td_processed[i] = td_fir_core_ref(new_data[i], &td_block_debug_fir_filter_test_0, data_td);

        int32_t __attribute__((aligned (8))) fd_processed[test_0_TD_BLOCK_LENGTH] = {0};
        fd_block_fir_add_data(new_data, &fd_fir_data_test_0);
        fd_block_fir_compute(
            fd_processed,
            &fd_fir_data_test_0,
            &fd_fir_filter_test_0);

        for(int i=0;i<test_0_FRAME_ADVANCE;i++){
            int error = td_processed[i]- fd_processed[i];
            // float ratio = (float)td_processed[i] / (float)fd_processed[i];
            printf("%ld %ld %ld %d\n", c++, td_processed[i], fd_processed[i], error);
            error_sum += error;
            if(error < 0) error = -error;
            abs_error_sum += error;
            count++;
        }

    }
    printf("avg error:%f avg abs error:%f\n", (float)error_sum / count, (float)abs_error_sum / count);
    printf("\n");
}

int main() {
  foo();
}