# start example
from audio_dsp.design.pipeline import Pipeline
from audio_dsp.stages import *

p, edge = Pipeline.begin(4)
edge = p.stage(VolumeControl, edge, "volume")
edge = p.stage(LimiterRMS, edge)
p.set_outputs(edge)
# end example

from pathlib import Path
mod = Path(__file__)
run_time_dir = mod.parents[3]/"doc"/"run_time_control_guide"
p.draw(run_time_dir/f"{mod.stem}.gv")
from audio_dsp.design.pipeline import generate_dsp_main
generate_dsp_main(p, Path(__file__).parent/"run_time_dsp/src/dsp")

# start config
config = p["volume"].get_config()
print(config)
# end config

(Path(__file__).parent/"run_time_dsp"/"config.txt").write_text(repr(config))

# check the app builds
from subprocess import run

kwargs = dict(check=True, cwd=Path(__file__).parent/"run_time_dsp")
run(["cmake", "-B", "build", "-G", "Unix Makefiles"], **kwargs)
run(["cmake", "--build", "build"], **kwargs)