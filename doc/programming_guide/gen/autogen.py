# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Generate includes for all the APIs in this repo
"""
from mako.template import Template
from pathlib import Path

ROOT_DIR = Path(__file__).parents[3]

def python_doc(dir):
    p_design = sorted(dir.glob("*.py"))
    p_design_modules = [".".join(p.parts[-3:])[:-3] for p in p_design if not p.name.startswith("_")]
    gen = Template("""
% for module in modules:
${module}
${"="*len(module)}


.. automodule:: ${module}
   :members:
   :show-inheritance:
   :inherited-members:


%endfor
""").render(modules=p_design_modules)
    (Path(__file__).parent / f"{dir.parts[-2]}.{dir.parts[-1]}.inc").write_text(gen)


def python_doc_no_inheritance(dir):
    p_design = sorted(dir.glob("*.py"))
    p_design_modules = [".".join(p.parts[-3:])[:-3] for p in p_design if not p.name.startswith("_")]
    gen = Template("""
% for module in modules:
${module}
${"="*len(module)}


.. automodule:: ${module}
   :members:
   :show-inheritance:


%endfor
""").render(modules=p_design_modules)
    (Path(__file__).parent / f"{dir.parts[-2]}.{dir.parts[-1]}_no_inheritance.inc").write_text(gen)


def c_doc(dir, glob="*.h"):
    api_dir = ROOT_DIR/"lib_audio_dsp"/"api"
    c_api_files = sorted(dir.glob(glob))
    c_design_modules = [p.relative_to(api_dir) for p in c_api_files if not p.name.startswith("_")]
    gen = Template("""
% for module in modules:
${str(module)}
${"="*len(str(module))}

.. doxygenfile:: ${module.name}

%endfor
""").render(modules=c_design_modules)
    (Path(__file__).parent / f"{dir.parts[-2]}.{dir.parts[-1]}.inc").write_text(gen)



python_doc(ROOT_DIR / "python" / "audio_dsp" / "design")
python_doc(ROOT_DIR / "python" / "audio_dsp" / "stages")
python_doc(ROOT_DIR / "python" / "audio_dsp" / "dsp")
python_doc_no_inheritance(ROOT_DIR / "python" / "audio_dsp" / "stages")

c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "dsp")
c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "stages", "adsp_*.h")
