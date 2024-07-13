# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Utility functions for building and running the application within
the Jupyter notebook
"""

import IPython
import ipywidgets as widgets
import pathlib
import shutil
import subprocess
import time

NullablePathLike = str | pathlib.Path | None


class DSPBuilder:

    log_str = """<pre style="font-family: monospace, monospace;">{output}</pre>"""

    def __init__(
        self: "DSPBuilder",
        source_dir: NullablePathLike = None,
        build_dir: NullablePathLike = None,
        bin_dir: NullablePathLike = None,
        project_name: str | None = None,
        config_name: str | None = None,
    ) -> None:
        if source_dir is None:
            self.source_dir = self._determine_source_dir()
        else:
            try:
                self.source_dir = pathlib.Path(source_dir)
            except TypeError as e:
                raise TypeError("source_dir must be pathlike or None!") from e

        if build_dir is None:
            self.build_dir = self._determine_build_dir()
        else:
            try:
                self.build_dir = pathlib.Path(build_dir)
            except TypeError as e:
                raise TypeError("build_dir must be pathlike or None!") from e

        if bin_dir is None:
            self.bin_dir = self._determine_bin_dir()
        else:
            try:
                self.bin_dir = pathlib.Path(bin_dir)
            except TypeError as e:
                raise TypeError("bin_dir must be pathlike or None!") from e

        if project_name is None:
            self.project_name = self._determine_project_name()
        elif isinstance(project_name, str):
            self.project_name = project_name
        else:
            raise TypeError("project_name parameter must be str or None!")

        if config_name is None:
            self.config_suffix = self._determine_config_suffix()
            self.config_name = self._determine_config_name()
        elif isinstance(config_name, str):
            self.config_suffix = "_" + config_name
            self.config_name = config_name
        else:
            raise TypeError("config_name parameter must be str or None!")

        self.target_name = self._determine_target_name()
        self.configure_done: bool = False

    def _determine_source_dir(self: "DSPBuilder") -> pathlib.Path:
        # We assume here that the CWD is the application directory, and that
        #     this is the desired source directory.
        return pathlib.Path.cwd()

    def _determine_build_dir(self: "DSPBuilder") -> pathlib.Path:
        # We assume here that the CWD is the application directory, and that
        #     this will contain a subdirectory /build/ which is the desired
        #     build directory.
        return pathlib.Path.cwd() / "build"

    def _determine_bin_dir(self: "DSPBuilder") -> pathlib.Path:
        # We assume here that the CWD is the application directory, and that
        #     this will contain a subdirectory /bin/ which is the bin
        #     directory.
        return pathlib.Path.cwd() / "bin"

    def _determine_project_name(self: "DSPBuilder") -> str:
        # We assume here that the name of the project is the same as the
        #     name of the enclosing directory
        return pathlib.Path.cwd().name

    def _determine_config_suffix(self: "DSPBuilder") -> str:
        # We assume here that if no config has been specified then the default
        #     target name is just the project name with no config suffix
        return ""

    def _determine_config_name(self: "DSPBuilder") -> str:
        # We assume here that if no config has been specified then the default
        #     config name is blank
        return ""

    def _determine_target_name(self: "DSPBuilder") -> str:
        # We assume here that the default target name is always the project name
        #     with the config suffix appended
        return self.project_name + self.config_suffix

    def _log(self: "DSPBuilder", process: subprocess.Popen, title: str = "") -> None:
        widget = widgets.HTML(value="")
        accordion = widgets.Accordion(children=[widget])
        accordion.set_title(0, title)
        IPython.display.display(accordion)
        output = ""
        for line in process.stdout:
            output += line
            widget.value = DSPBuilder.log_str.format(output=output)
        process.wait()
        if process.returncode:
            accordion.set_title(0, title + "  Failed ❌ (click for details)")
        else:
            accordion.set_title(0, title + "  ✔")

    def configure(self: "DSPBuilder") -> int:
        cache = self.build_dir / "CMakeCache.txt"
        makefile = self.build_dir / "Makefile"
        ninjabuild = self.build_dir / "build.ninja"
        if (
            (not self.configure_done)
            or (not cache.exists())
            or not (makefile.exists() or ninjabuild.exists())
        ):
            if cache.exists():
                # Generator is already known by CMake
                ret = subprocess.Popen(
                    [*(f"cmake -S {self.source_dir} -B {self.build_dir}".split())],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                # need to configure, default to Ninja because it's better
                generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
                ret = subprocess.Popen(
                    [
                        *(f"cmake -S {self.source_dir} -B {self.build_dir} -G".split()),
                        generator,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            self._log(ret, "Configuring...")
            if not ret.returncode:
                self.configure_done = True
            return ret.returncode

    def build(self: "DSPBuilder") -> int:
        ret = subprocess.Popen(
            f"cmake --build {self.build_dir} --target {self.target_name}".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log(ret, "Compiling...")
        return ret.returncode

    def run(self: "DSPBuilder") -> None:
        app = (
            self.bin_dir
            / self.config_name
            / (self.project_name + self.config_suffix + ".xe")
        )
        ret = subprocess.Popen(
            f"xrun {app}".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log(ret, f"Running...")
        return ret.returncode

    def configure_build_run(self: "DSPBuilder") -> None:
        returncode = self.configure()
        if returncode:
            return
        returncode = self.build()
        if returncode:
            return
        returncode = self.run()
        if returncode:
            return
        time.sleep(5)
        print("Done!\r")
