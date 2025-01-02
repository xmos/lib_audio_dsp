app_td_block_fir
---

This demos 16 concurrent FIRs running on a single tile with a frame size of 8 and of length 4008.
Currently, it runs at 46kHz.
The application demonstrates how two channels per thread can be run in parallel over a single tile
in order to get 16 channels running concurrently.