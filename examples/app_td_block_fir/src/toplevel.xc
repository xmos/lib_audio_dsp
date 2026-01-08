// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <platform.h>

void audio_gen(chanend c[8]);
void worker_tile(chanend c[8]);

int main(){
    chan c[8];
    par {
        on tile[0] : audio_gen(c);
        on tile[1] : worker_tile(c);
    }
    return 0;
}