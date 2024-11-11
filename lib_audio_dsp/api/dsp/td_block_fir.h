
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
    uint32_t accu_shl; //the amount to shl the accumulator after all accumulation is complete
} td_block_fir_filter_t;

/*
Function to initialise the struct that manages the data, rather than coefficients, for a 
time domain block convolution. The filter generator should be run first resulting in a header 
that defines the parameters for this function. For example, running the generator with --name={NAME}
would 
    fir_data - Struct of type td_block_fir_data_t
    data - an area of memory to be used by the struct in order to hold a history of the samples. The 
           define {NAME}_DATA_BUFFER_ELEMENTS specifies exactly the number of int32_t elements to 
           allocate for the filter {NAME} to correctly function.
    data_buffer_elements - The number of samples contained in the data array, this should be 
           {NAME}_DATA_BUFFER_ELEMENTS.
*/
void td_block_fir_data_init(
    td_block_fir_data_t * fir_data, 
    int32_t *data, 
    uint32_t data_buffer_elements);

/*
Function to add samples to the FIR data structure.
    samples_in - array of int32_t samples of length expected to be fir_data->frame_advance.
    fir_data - Struct of type td_block_fir_data_t to which the samples will be added.
*/
void td_block_fir_add_data(
    int32_t samples_in[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data);

/*
Function to compute the convolution between fir_data and fir_filter.
    samples_out - Array of length TD_BLOCK_FIR_LENGTH(8), which will be used to return the 
        processed samples.
    fir_data - Struct of type td_block_fir_data_t to which the data samples will be used.
    fir_filter - Struct of type td_block_fir_filter_t to which the coefficient samples will be used.
*/
void td_block_fir_compute(
    int32_t samples_out[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data, 
    td_block_fir_filter_t * fir_filter);

// bring up and debug code below

void td_block_fir_add_data_ref(
    int32_t input_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data);
    
void td_block_fir_compute_ref(
    int32_t samples_out[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data, 
    td_block_fir_filter_t * fir_filter
); 

#endif //TD_BLOCK_FIR_H