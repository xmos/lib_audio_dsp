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


%endfor
""").render(modules=p_design_modules)
    (Path(__file__).parent / f"{dir.parts[-2]}.{dir.parts[-1]}.inc").write_text(gen)


def c_doc(dir):
    api_dir = ROOT_DIR/"lib_audio_dsp"/"api"
    c_api_files = sorted(dir.glob("*.h"))
    c_design_modules = [str(p.relative_to(api_dir)) for p in c_api_files if not p.name.startswith("_")]
    gen = Template("""
% for module in modules:
${module}
${"="*len(module)}

.. doxygenfile:: ${module}

%endfor
""").render(modules=c_design_modules)
    (Path(__file__).parent / f"{dir.parts[-2]}.{dir.parts[-1]}.inc").write_text(gen)



python_doc(ROOT_DIR / "python" / "audio_dsp" / "design")
python_doc(ROOT_DIR / "python" / "audio_dsp" / "stages")
python_doc(ROOT_DIR / "python" / "audio_dsp" / "dsp")
c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "dsp")
c_doc(ROOT_DIR / "lib_audio_dsp" / "api" / "stages")
