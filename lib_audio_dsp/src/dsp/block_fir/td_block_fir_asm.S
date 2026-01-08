// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__XS3A__)

.text
.align 16;
.issue_mode dual

/*  
void td_block_fir_compute(
    int32_t output_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t * fir_data, 
    td_block_fir_filter_t * fir_filter, 
);

typedef struct td_block_fir_data_t {
    int32_t * data;       // 0
    uint32_t index;       // 1
    uint32_t data_stride; // 2
} td_block_fir_data_t;

typedef struct td_block_fir_filter_t {
    int32_t * coefs;      // 0
    uint32_t block_count; // 1
    uint32_t accu_shr;    // 2
} td_block_fir_filter_t;

See td_block_fir_compute_ref for reference code.

*/

#define FUNCTION_NAME   td_block_fir_compute
#define NSTACKWORDS     (6+8) + 32

#define output_block_p      r0
#define data_struct_p       r1
#define filter_struct_p     r2

#define s r4
#define data_p r6
#define data_stride r7
#define thirtytwo r9

#define t r11

.cc_top FUNCTION_NAME.function,FUNCTION_NAME
FUNCTION_NAME:
  dualentsp NSTACKWORDS
  std r5, r4, sp[0]
  std r7, r6, sp[1]
  std r9, r8, sp[2]

  {ldw data_p, data_struct_p[0] ; ldc thirtytwo, 32}
  {ldw data_stride, data_struct_p[2]; add r3, thirtytwo, thirtytwo}
  {ldw r8, data_struct_p[1] ; add data_p, data_p, data_stride}
  {ldw r5, filter_struct_p[1] ; add data_p, data_p, r8}
  {shl s, r5, 5; sub r8, r8, thirtytwo}
  {vclrdr; ldc t, 0}
  {vsetc t; sub data_p, data_p, s}
  ashr t, r8, 5
  {sub s, r5, t; lss r8, r5, t }
  {ldw r5, filter_struct_p[0] ; nop}
  {bf r8, prepare; sub r3, r3, 4}
    {add t, t, s; ldc s, 0}
  //At the point s and t are the first and second loop counters.
  prepare:
  {bf s, td_block_fir_compute_skip_first_half; nop} //if s is zero then just do t
td_block_fir_compute_first_half:
  {vldc r5[0] ; add r5, r5, thirtytwo }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub s, s, 1 }
  {bt s, td_block_fir_compute_first_half; add data_p, data_p, r3}

td_block_fir_compute_skip_first_half:
  {bf t, td_block_fir_compute_done; sub data_p, data_p, data_stride} //if t is zero then just do s

td_block_fir_compute_second_half:
  {vldc r5[0] ; add r5, r5, thirtytwo }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub data_p, data_p, 4 }
  {vlmaccr data_p[0] ; sub t, t, 1 }
  {bt t, td_block_fir_compute_second_half; add data_p, data_p, r3}

td_block_fir_compute_done:
  {ldw s, filter_struct_p[2]; ldaw t, sp[6]}

  //In order to use VLSAT we need 8 shift values from memory. This puts them into memory
  //as fast as possible. 
  std s, s, sp[3]
  std s, s, sp[4]
  std s, s, sp[5]
  std s, s, sp[6]
  
  {vlsat t[0] ; nop}

  {ldw s, filter_struct_p[3]; nop}
  {vstr t[0] ; nop}
  vlashr t[0], s
  
  vstr output_block_p[0]

  ldd r5, r4, sp[0]
  ldd r7, r6, sp[1]
  ldd r9, r8, sp[2]
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
