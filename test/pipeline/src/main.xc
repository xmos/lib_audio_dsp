// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <platform.h>
#include <xs1.h>
#include <xscope.h>
#include <stdlib.h>
#ifdef __XC__
#define chanend_t chanend
#else
#include <xcore/chanend.h>
#endif
#include "xscope_io_device.h"

extern int main_c();

int main (void)
{
  chan xscope_chan;
  par
  {
    xscope_host_data(xscope_chan);
    on tile[0]: {
        xscope_io_init(xscope_chan);
        exit(main_c());
    }
  }
  return 0;
}
