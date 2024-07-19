# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Generate includes for all the APIs in this repo
"""
from mako.template import Template
from pathlib import Path
import ast

ROOT_DIR = Path(__file__).parents[1]
PYTHON_ROOT = Path(ROOT_DIR, "python")
CTRL_GEN_DIR = Path(__file__).parent / "dsp_components" / "runtime_control" / "gen"
DSP_GEN_DIR = Path(__file__).parent / "dsp_components" / "stages" / "gen"
TOOL_USER_GEN_DIR = Path(__file__).parent / "tool_user_guide" / "gen"

def python_doc(src_dir, dst_dir):
    p_design = sorted(src_dir.glob("*.py"))
    p_design_modules = [".".join(p.parts[-3:])[:-3] for p in p_design if not p.name.startswith("_")]
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

def get_file_info(fname):
    class_list = []

    with open(fname) as fp:
        node = ast.parse(fp.read())

    docstring = ast.get_docstring(node)
    if docstring == None:
        assert 0, f"{fname} does not have a docstring"

    for i in node.body:
        if isinstance(i, ast.ClassDef):
            class_list.append(i.name)

    return docstring, class_list

def python_doc_stages(src_dir, dst_dir):
    p_design = sorted(src_dir.glob("*.py"))
    for file in p_design:
        if file.name.startswith("_"):
            continue
        module = ".".join(file.parts[-3:])[:-3]
        module_name = (file.parts[-1])[:-3]
        title = module_name.replace("_", " ")
        # Sorry
        title = title.title().replace("Rms", "RMS").replace("Fir", "FIR")
        docstring, classes = get_file_info(file)
        gen = Template(
"""${"#"*len(title)}
${title}
${"#"*len(title)}

${docstring}

%for cl in classes:

%if len(classes) != 1:
${"="*len(cl)}
${cl}
${"="*len(cl)}
%endif

.. autoclass:: ${module}.${cl}
    :noindex:
    :members:

%endfor"""
).render(title = title, module = module, classes = classes, docstring = docstring)
        (dst_dir / f"{module_name}.rst").write_text(gen)


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
    (dst_dir / f"{src_dir.parts[-2]}.{src_dir.parts[-1]}.inc").write_text(gen)


python_doc(ROOT_DIR / "python" / "audio_dsp" / "design", TOOL_USER_GEN_DIR)
python_doc_stages(ROOT_DIR / "python" / "audio_dsp" / "stages", DSP_GEN_DIR)

c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "stages", TOOL_USER_GEN_DIR, "adsp_*.h")
