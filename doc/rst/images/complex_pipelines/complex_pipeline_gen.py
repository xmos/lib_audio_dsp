# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Generate complex pipeline diagrams for documentation."""


from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages import Biquad, Fork, Bypass

# NOTE: The :width: in the .rst is set to {.png_width}/100 for consistent scaling,
# but the svg images are used

if __name__ == "__main__":

    file_ext = "png"

    # start with 7 input channels
    p, inputs = Pipeline.begin(7, fs=48000)

    # pass the first 2 inputs to a 2-channel Biquad
    i0 = p.stage(Biquad, inputs[0:2])
 
    # pass the third input (index 2) to a 1-channel biquad
    i1 = p.stage(Biquad, inputs[2])
 
    # pass the inputs at index 3, 5, and 6 to a 3 channel biquad
    i2 = p.stage(Biquad, inputs[3, 5, 6])
 
    # pass all of i0 and i1, as well as the first channel in i2
    # to create a 4 channel biquad
    i3 = p.stage(Biquad, i0 + i1 + i2[0]) 
 
    # The pipeline output has 6 channels:
    # - all four i3 channels 
    # - the 2nd and 3rd channel from i2
    p.set_outputs(i3 + i2[1:])
 
    p.draw(f"doc/rst/images/complex_pipelines/7_chan_biquad_pipeline.{file_ext}")

    #%%
    p, inputs = Pipeline.begin(1, fs=48000)

    # fork the input to create a 2 channel signal
    x = p.stage(Fork, inputs, count=2)

    # fork again to create a 4 channel signal
    x = p.stage(Fork, x, count=2)

    # there are now 4 channels in the pipeline output
    p.set_outputs(x)
    p.draw(f"doc/rst/images/complex_pipelines/fork_pipeline.{file_ext}")

    #%%

    p, i = Pipeline.begin(1, fs=48000)

    # thread 0
    i = p.stage(Biquad, i)

    # thread 1
    p.next_thread()
    i = p.stage(Biquad, i)

    # thread 2
    p.next_thread()
    i = p.stage(Biquad, i)

    p.set_outputs(i)

    p.draw(f"doc/rst/images/complex_pipelines/multi_thread_pipeline.{file_ext}")

    #%%
    p, inputs = Pipeline.begin(2, fs=48000)

    # inputs[1] is not used on thread 0
    x1 = p.stage(Biquad, inputs[0])

    p.next_thread()

    # inputs[1] first used on thread 1
    x = p.stage(Biquad, x1 + inputs[1])

    p.set_outputs(x)
    p.draw(f"doc/rst/images/complex_pipelines/thread_crossings_bad.{file_ext}")

    #%%
    
    p, inputs = Pipeline.begin(2, fs=48000)

    # both inputs are not used on this thread
    x1 = p.stage(Biquad, inputs[0])
    x2 = p.stage(Bypass, inputs[1])

    p.next_thread()

    x = p.stage(Biquad, x1 + x2)

    p.set_outputs(x)

    p.draw(f"doc/rst/images/complex_pipelines/thread_crossings_bypass.{file_ext}")

    #%%
    
    p, inputs = Pipeline.begin(2, fs=48000)

    x1 = p.stage(Biquad, inputs[0])
    p.next_thread()

    x2 = p.stage(Bypass, inputs[1])
    p.next_thread()

    # x1 and x2 have both crossed 1 thread already
    x = p.stage(Biquad, x1 + x2)

    p.set_outputs(x)
    p.draw(f"doc/rst/images/complex_pipelines/thread_crossings_parallel.{file_ext}")