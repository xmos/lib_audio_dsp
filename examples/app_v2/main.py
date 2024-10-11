from audio_dsp.v2 import ir, optimise, code_gen, dot
from pprint import pp

if __name__ == "__main__":
    ir = ir.load_from_yaml("dsp.yaml")
    ir = optimise.validate_input(ir)
    ir = optimise.optimise(ir)
    pp(ir.config_struct.model_dump())
    dot.render(ir, "draw.gv")
    code_gen.code_gen(ir, "src")
