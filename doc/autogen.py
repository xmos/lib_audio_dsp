# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""
Generate includes for all the APIs in this repo
"""
import os
import ast
import re
import yaml
from mako.template import Template
from pathlib import Path

# Load folders
#TOOLS_USER_GUIDE_DIR    = "01_tool_user_guide"
#DESIGN_GUIDE_DIR        = "02_design_guide"
DSP_COMP_DIR            = "03_dsp_components"
#RUNTIME_CTRL_DIR        = "04_run_time_control_guide"
API_REFERENCE_DIR       = "05_api_reference"

# Define paths
ROOT_DIR = Path(__file__).parents[1]
PYTHON_ROOT = ROOT_DIR / "python"
STAGES_LIST_PATH = Path(__file__).parent / "rst"/ DSP_COMP_DIR / "gen" / "stages.rst"
DSP_GEN_DIR = Path(__file__).parent / "rst" / API_REFERENCE_DIR / "stages" / "gen"
PY_STAGE_MAKO = ROOT_DIR / "python" / "audio_dsp" / "design" / "templates" / "py_stage_doc.mako"
YAML_DIR = ROOT_DIR / "stage_config"
API_REF_GEN_DIR = Path(__file__).parent / "rst"/ API_REFERENCE_DIR / "gen"


def get_module_from_path(paths):
    # ex: manipulates path to get a module, ex: a/b/c.py -> a.b.c
    for path in paths:
        if not path.name.startswith("_"):
            path = path.with_suffix("").relative_to(PYTHON_ROOT)
            yield '.'.join(path.parts)

def get_file_info(fname):
    class_list = []

    with open(fname) as fp:
        node = ast.parse(fp.read())

    docstring = ast.get_docstring(node)
    if docstring == None:
        assert 0, f"{fname} does not have a docstring"
    docstring = docstring.replace('\n', ' ')

    for i in node.body:
        if isinstance(i, ast.ClassDef):
            if i.name.startswith("_"):
                continue
            class_list.append(i.name)

    return docstring, class_list

def python_doc(src_dir, dst_dir):
    p_design = sorted(src_dir.glob("*.py"))
    p_design_modules = list(get_module_from_path(p_design))
    gen = Template("""
% for module in modules:
${module}
${"="*len(module)}

.. automodule:: ${module}
   :noindex:
   :members:

%endfor"""
).render(modules=p_design_modules)
    (dst_dir / f"{src_dir.parts[-2]}.{src_dir.parts[-1]}.inc").write_text(gen)

def python_doc_stages(src_dir, dst_dir, list_file):
    p_design = sorted(src_dir.glob("*.py"))
    all_classes = []
    all_files = []
    all_docstrings = []
    for file in p_design:
        if file.name.startswith("_"):
            continue
        module = ".".join(file.parts[-3:])[:-3]
        module_name = (file.parts[-1])[:-3]
        title = module_name.replace("_", " ")
        # Sorry
        title = title.title().replace("Rms", "RMS").replace("Fir", "FIR").replace("Eq", "EQ")
        docstring, classes = get_file_info(file)
        all_docstrings.append(docstring)
        all_classes.append(classes)
        all_files.append(title)

        class_data = {}
        for class_name in classes:
            safe_name = class_name.replace("RMS", "Rms")
            snake_name = re.sub(r'(?<!^)(?=[A-Z]|(?<!\d)(?=\d))', '_', safe_name).lower()
            yaml_path =  Path(YAML_DIR, snake_name + ".yaml")
            if yaml_path.is_file():
                with open(yaml_path, "r") as fd:
                    data = yaml.safe_load(fd)   
                    struct_name = list(data["module"].keys())[0]
                    class_data[class_name] = data["module"][struct_name]
            else:
                class_data[class_name] = None
            pass

        gen = Template(filename=str(PY_STAGE_MAKO)).render(title = title, module = module, classes = classes, docstring = docstring, class_data = class_data)
        (dst_dir / f"{module_name}.rst").write_text(gen, newline="")
    
    # something should be really wrong if this asserts
    assert len(all_classes) == len(all_files)
    gen = Template("""

.. _dsp_stages_list:

DSP Stages List
===============

This a list of all the supported stages that can be used with the DSP pipeline tool:

% for i in range(len(titles)):
*   :ref:`${titles[i]}_stages`

    ${docstrings[i]}

% for cl in classes[i]:
    * :ref:`${cl}_stage`
% endfor ## cl in classes

% endfor ## i in len(titles)

""").render(titles = all_files, classes = all_classes, docstrings = all_docstrings)
    list_file.write_text(gen, newline="")

def c_doc(src_dir, dst_dir, glob="*.h"):
    api_dir = ROOT_DIR/"lib_audio_dsp"/"api"
    c_api_files = sorted(src_dir.glob(glob))
    c_design_modules = [p.relative_to(api_dir) for p in c_api_files if not p.name.startswith("_")]
    gen = Template("""
% for module in modules:
${str(module)}
${"="*len(str(module))}

.. doxygenfile:: ${module.name}

%endfor
""").render(modules=c_design_modules)
    output_file = (dst_dir / f"{src_dir.parts[-2]}.{src_dir.parts[-1]}.inc")
    if os.name == "nt": # if windows replace backslashes so we can reference later in docs
        gen = gen.replace("\\", "/")
    output_file.write_text(gen)

if __name__ == "__main__":
    
    python_doc(ROOT_DIR / "python" / "audio_dsp" / "design", API_REF_GEN_DIR)
    python_doc_stages(ROOT_DIR / "python" / "audio_dsp" / "stages", DSP_GEN_DIR, STAGES_LIST_PATH)
    #c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "stages", API_REF_GEN_DIR, "adsp_*.h")
    print("Done")
