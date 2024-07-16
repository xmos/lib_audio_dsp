
from pathlib import Path
import audio_dsp.design.parse_config as pc
import types 


def main():
    # parse the configs and make the control header files
    pkg_dir = Path(__file__).parents[3]
    config_dir = f"{pkg_dir}/stage_config"
    parse_config_path = Path(pkg_dir, "design", "parse_config")

    out_dir = Path(pkg_dir, "doc", "programming_guide", "gen", "control_gen")

    pc_args = types.SimpleNamespace()
    pc_args.out_dir = out_dir
    pc_args.config_dir = config_dir
    pc.main(pc_args)

    # automodule the 



    # subprocess.check_output(f"python -m {parse_config_path} -c {config_dir} -o tmp")


    # pc(config)

    # yaml_dict = yaml.load(Path(config).read_text(), Loader=yaml.Loader)
    # # module dict contains 1 entry with the name of the module as its key
    # name = next(iter(yaml_dict["module"].keys()))
    # _control_fields = {
    #     name: str() for name in yaml_dict["module"][name].keys()
    # }
if __name__ == "__main__":
    main()