import yaml
import glob
from mako.template import Template
import argparse
import os
from pathlib import Path



pkg_dir = Path(__file__).parent
templates_dir = f"{pkg_dir}/templates"

def parse_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description='Generate config struct files')
    parser.add_argument('--config-dir', '-c', required=True, type=str, default=None,
                        help="Path to the config yaml files")
    parser.add_argument('--out-dir', '-o', type=str, default=f'{str(Path.cwd())}/output_files',
                        help="output directory for host and device files")
    args = parser.parse_args()
    return args

## Parse command line arguments
print("In parse_config.py!!\n\n")
args = parse_arguments()
print(f"out_dir = {args.out_dir}")
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(f"{args.out_dir}/common", exist_ok=True)
os.makedirs(f"{args.out_dir}/device", exist_ok=True)
os.makedirs(f"{args.out_dir}/host", exist_ok=True)
os.makedirs(f"{args.out_dir}/generator", exist_ok=True)

struct_def_template = Template(filename=f'{templates_dir}/struct_def_h.mako')
struct_offset_template = Template(filename=f'{templates_dir}/gen_command_map_offsets_c.mako')
cmd_map_template = Template(filename=f'{templates_dir}/host_cmd_map.mako')
cmd_ids_template = Template(filename=f'{templates_dir}/cmds_h.mako')
module_config_offsets_template = Template(filename=f'{templates_dir}/cmd_offsets_h.mako')

cmd_map = {}

files = glob.glob(f"{args.config_dir}/*.yaml")

for fl in files:
    with open(fl, "r") as fd:
        data = yaml.safe_load(fd)

        struct_name=list(data["module"].keys())[0]
        includes = data["includes"] if "includes" in data else []
        defines = data["defines"] if "defines" in data else dict()
        with open(f"{args.out_dir}/common/{struct_name}_config.h", "w") as f_op:
            f_op.write(struct_def_template.render(name=struct_name, data=data["module"][struct_name],
                                                  includes=includes, defines=defines))

        cmd_map[struct_name] = data["module"][struct_name]

with open(f"{args.out_dir}/generator/gen_cmd_map_offset.c", "w") as f_op:
    f_op.write(struct_offset_template.render(cmd_map=cmd_map))

# Generate cmd_map used by the host
with open(f"{args.out_dir}/host/host_cmd_map.h", "w") as f_op:
    f_op.write(cmd_map_template.render(cmd_map=cmd_map))

# Generate #defines present in the cmd map used by the host
with open(f"{args.out_dir}/common/cmds.h", "w") as f_op:
    f_op.write(cmd_ids_template.render(cmd_map=cmd_map))

#generate the config offsets for every module
with open(f"{args.out_dir}/device/cmd_offsets.h", "w") as f_op:
    f_op.write(module_config_offsets_template.render(cmd_map=cmd_map))
