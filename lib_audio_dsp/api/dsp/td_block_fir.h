
#ifndef TD_BLOCK_FIR_H
#define TD_BLOCK_FIR_H
#include <stdint.h>

// This is fixed due to the VPU
#define TD_BLOCK_FIR_LENGTH 8

/** 
 * @brief Time domain data struct.
 */
typedef struct td_block_fir_data_t
{
    /** The actual data samples. */
    int32_t *data;
    /** Index of the head of the FIFO data. */
    uint32_t index;  
    /** Tthe number of bytes a pointer has to have subtracted to move around the circular buffer. */
    uint32_t data_stride;
} td_block_fir_data_t;

/** 
 * @brief Time domain filter struct.
 */
typedef struct td_block_fir_filter_t {
    /** The actual coefficients, reversed for the VPU. */
    int32_t * coefs; 
    /** Count of blocks of data. */
    uint32_t block_count; 
    /** The amount to shr the accumulator after all accumulation is complete.*/
    uint32_t accu_shr;
    /** The amount to shl the accumulator after all accumulation is complete.*/
    uint32_t accu_shl; //t
} td_block_fir_filter_t;

/**
 * @brief Function to initialise the struct that manages the data, rather than coefficients, for a 
time domain block convolution. The filter generator should be run first resulting in a header 
that defines the parameters for this function. For example, running the generator with --name={NAME}
would generate defines prepended with {NAME}, i.e. {NAME}_DATA_BUFFER_ELEMENTS, {NAME}_TD_BLOCK_LENGTH,
{NAME}_BLOCK_COUNT, {NAME}_FRAME_ADVANCE, {NAME}_FRAME_OVERLAP.
 * 
 * @param fir_data Poitner to struct of type td_block_fir_data_t
 * @param data Pointer to an amount of memory to be used by the struct in order to hold a history of the samples. The 
           define {NAME}_DATA_BUFFER_ELEMENTS specifies exactly the number of int32_t elements to 
           allocate for the filter {NAME} to correctly function.
 * @param data_buffer_elements The number of samples contained in the data array, this should be 
           {NAME}_DATA_BUFFER_ELEMENTS.
 */
void td_block_fir_data_init(
    td_block_fir_data_t * fir_data, 
    int32_t *data, 
    uint32_t data_buffer_elements);

/**

 * @brief Function to add samples to the FIR data structure.
 * 
 * @param samples_in Array of int32_t samples of length TD_BLOCK_FIR_LENGTH.
 * @param fir_data Pointer to struct of type td_block_fir_data_t to which the samples will be added.
 */
void td_block_fir_add_data(
    int32_t samples_in[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data);

/**
 * @brief Function to compute the convolution between fir_data and fir_filter.
 * 
 * @param samples_out Array of length TD_BLOCK_FIR_LENGTH(8), which will be used to return the 
        processed samples.
 * @param fir_data Pointer to struct of type td_block_fir_data_t from which the data samples will be obtained.
 * @param fir_filter Pointer to struct of type td_block_fir_filter_t from which the coefficients will be obtained.
 */
void td_block_fir_compute(
    int32_t samples_out[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data, 
    td_block_fir_filter_t * fir_filter);

#endif //TD_BLOCK_FIR_H