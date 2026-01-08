// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__XS3A__)

.text
.align 16;
.issue_mode dual

/*  
void td_block_fir_add_data( 
    int32_t samples_in[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data
);

typedef struct td_block_FIR_data_t {
    int32_t * data;       // 0
    uint32_t index;       // 1
    uint32_t data_stride; // 2
} td_block_fir_data_t;

See td_block_fir_add_data_ref for reference code.
*/

#define FUNCTION_NAME   td_block_fir_add_data
#define NSTACKWORDS     4

#define input_block_p       r0
#define data_struct_p       r1

.cc_top FUNCTION_NAME.function,FUNCTION_NAME
FUNCTION_NAME:
  dualentsp NSTACKWORDS
  std r4, r5, sp[0]
  std r6, r7, sp[1]

  {vldd input_block_p[0]; ldc r0, 32}
  {ldw r3, data_struct_p[1]; nop} //load index
  {ldw r7, data_struct_p[0]; add r6, r3, r0} //load data_pointer
  {ldw r2, data_struct_p[2]; add r11, r7, r3} //load data_stride
  {vstd r11[0]; sub r3, r2, r3}

  // if this is the end of the buffer then paste it onto the front too
  {bt r3, td_block_fir_add_data_skip; nop} 
    {vstd r7[0]; mov r6, r0}
td_block_fir_add_data_skip:  
  stw r6, data_struct_p[1]

  ldd r6, r7, sp[1]
  ldd r4, r5, sp[0]
  { nop                ; retsp NSTACKWORDS           }

.L_func_end_unpack:
.cc_bottom FUNCTION_NAME.function

.global FUNCTION_NAME
.type FUNCTION_NAME,@function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS;     .global FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;                  .global FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;                 .global FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;               .global FUNCTION_NAME.maxchanends
.size FUNCTION_NAME,.L_func_end_unpack - FUNCTION_NAME

#undef NSTACKWORDS
#undef FUNCTION_NAME

#endif //defined(__XS3A__)
