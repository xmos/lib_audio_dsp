
#ifndef TD_BLOCK_FIR_H
#define TD_BLOCK_FIR_H
#include <stdint.h>

// This is fixed due to the VPU
#define TD_BLOCK_FIR_LENGTH 8

typedef struct td_block_fir_data_t {
    int32_t * data; //the actual data
    uint32_t index;  //current head index of the data
    uint32_t data_stride; //the number of bytes a pointer has to have subtracted to move around the circular buffer! //TODO
} td_block_fir_data_t;

typedef struct td_block_fir_filter_t {
    int32_t * coefs; //the actual coefficients, reversed for the VPU
    uint32_t block_count;  // the actual number of blocks
    uint32_t accu_shr; //the amount to shr the accumulator after all accumulation is complete
    uint32_t accu_shl; //the amount to shr the accumulator after all accumulation is complete
} td_block_fir_filter_t;

void td_block_fir_data_init(
    td_block_fir_data_t * d, 
    int32_t *data, 
    uint32_t data_buffer_elements);

void td_block_fir_add_data(
    td_block_fir_data_t * data_struct,
    int32_t input_block[TD_BLOCK_FIR_LENGTH]);

void td_block_fir_compute(
    int32_t output_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * data_struct, 
    td_block_fir_filter_t * filter_struct);

// bring up and debug code below

typedef struct td_block_debug_fir_filter_t{
    int32_t * coefs; //the actual coefficients
    uint32_t length; //the count of coefficients
    uint32_t exponent; //the output exponent(for printing)
    uint32_t accu_shr; //the amount to shr the accumulator after all accumulation is complete
    uint32_t prod_shr; //the amount of shr the product of data and coef before accumulating
} td_block_debug_fir_filter_t;

void td_block_fir_add_data_ref(
    td_block_fir_data_t * data_struct,
    int32_t input_block[TD_BLOCK_FIR_LENGTH]);
    
void td_block_fir_compute_ref(
    int32_t output_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * data_struct, 
    td_block_fir_filter_t * filter_struct
); 

int32_t td_fir_core_ref(
    int32_t new_sample,
    td_block_debug_fir_filter_t * filter,
    int32_t * data);

#endif //TD_BLOCK_FIR_H