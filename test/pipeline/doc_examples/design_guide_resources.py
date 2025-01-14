# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

# start example
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages import *

p, edge = Pipeline.begin(4)

# thread 0
e0 = p.stage(Bypass, edge[0], "a")

# thread 1
p.next_thread()
e1 = p.stage(Bypass, edge[1:], "b")
e1 = p.stage(Bypass, e1, "c")

# thread 2
p.next_thread()
e = p.stage(Bypass, e0 + e1, "d")

p.set_outputs(e)
# end example

from pathlib import Path
from utils import IMGS_PATH
output_dir = IMGS_PATH / Path(__file__).with_suffix(".gv").name
p.draw(output_dir)
