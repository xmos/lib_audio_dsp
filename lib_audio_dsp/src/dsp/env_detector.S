// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/defines.h"

.section .cp.rodata, "ac", @progbits
.align 4
q_sig_rnd_acc:
.word 1 << (Q_SIG - 1)

/*
void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample
)*/

.text
.issue_mode dual
.align 8

#define ED_ATT_AL       0
#define ED_REL_AL       1
#define ED_ENV          2

#define NSTACKWORDS     2
#define state           r0
#define sample          r1
#define ah              r2
#define al              r3
#define alpha           r4
#define env             r5


.cc_top adsp_env_detector_peak.function,adsp_env_detector_peak
adsp_env_detector_peak:
    { ldc al, 0                     ; dualentsp NSTACKWORDS         }
    std r5, r4, sp[0]


    // take |sample|
    ashr r2, sample, 32
    { add sample, sample, r2        ; ldw env, state[ED_ENV]        }
    xor sample, sample, r2

    // choose alpha
    { lss r11, env, sample          ; ldw alpha, state[ED_ATT_AL]   }
    { ldc ah, 0                     ; bt r11, .alpha_if_peak_end    }

    {                               ; ldw alpha, state[ED_REL_AL]   }
    .alpha_if_peak_end:

    // run q31 ema
    { sub sample, sample, env       ; mkmsk r11, 5                  }

    linsert ah, al, env, r11, 32
    maccs ah, al, alpha, sample
    lsats ah, al, r11
    lextract env, ah, al, r11, 32
    {                               ; stw env, state[ED_ENV]        }
    
    ldd r5, r4, sp[0]
    retsp NSTACKWORDS
.L_func_end_peak:
.cc_bottom adsp_env_detector_peak.function;


.cc_top adsp_env_detector_rms.function,adsp_env_detector_rms
adsp_env_detector_rms:
    { ldc ah, 0                     ; dualentsp NSTACKWORDS         }
    std r5, r4, sp[0]

    // calculate sample^2
    { ldc r11, Q_SIG                ; ldw al, cp[q_sig_rnd_acc]     }
    maccs ah, al, sample, sample
    lsats ah, al, r11
    lextract sample, ah, al, r11, 32
    { ldc ah, 0                     ; ldw env, state[ED_ENV]        }
    
    // choose alpha
    { lss r11, env, sample          ; ldw alpha, state[ED_ATT_AL]   }
    { ldc al, 0                     ; bt r11, .alpha_if_rms_end     }

    {                               ; ldw alpha, state[ED_REL_AL]   }
    .alpha_if_rms_end:

    { sub sample, sample, env       ; mkmsk r11, 5                  }

    // run q31 ema
    linsert ah, al, env, r11, 32
    maccs ah, al, alpha, sample
    lsats ah, al, r11
    lextract env, ah, al, r11, 32
    {                               ; stw env, state[ED_ENV]        }

    ldd r5, r4, sp[0]
    retsp NSTACKWORDS
.L_func_end_rms:
.cc_bottom adsp_env_detector_rms.function;

.globl adsp_env_detector_peak;
.type adsp_env_detector_peak,@function
.set adsp_env_detector_peak.nstackwords,NSTACKWORDS;     .global adsp_env_detector_peak.nstackwords; 
.set adsp_env_detector_peak.maxcores,1;                  .global adsp_env_detector_peak.maxcores; 
.set adsp_env_detector_peak.maxtimers,0;                 .global adsp_env_detector_peak.maxtimers; 
.set adsp_env_detector_peak.maxchanends,0;               .global adsp_env_detector_peak.maxchanends; 
.size adsp_env_detector_peak, .L_func_end_peak - adsp_env_detector_peak

.globl adsp_env_detector_rms;
.type adsp_env_detector_rms,@function
.set adsp_env_detector_rms.nstackwords,NSTACKWORDS;     .global adsp_env_detector_rms.nstackwords; 
.set adsp_env_detector_rms.maxcores,1;                  .global adsp_env_detector_rms.maxcores; 
.set adsp_env_detector_rms.maxtimers,0;                 .global adsp_env_detector_rms.maxtimers; 
.set adsp_env_detector_rms.maxchanends,0;               .global adsp_env_detector_rms.maxchanends; 
.size adsp_env_detector_rms, .L_func_end_rms - adsp_env_detector_rms