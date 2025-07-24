import os
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from pathlib import Path


def check_version(version_str: str):
    l = version_str.split(".")
    for val in l:
        if not val.isdigit():
            raise ValueError(f"Invalid version string: {version_str}")


def find_version(src_file_content: str):
    # find macro SSRJSON_BENCHMARK_VERSION
    prefix = "#define SSRJSON_BENCHMARK_VERSION"
    for line in src_file_content.splitlines():
        if line.startswith(prefix):
            version = line[len(prefix) :].strip()[1:-1]
            check_version(version)
            return version
    raise RuntimeError("Cannot find SSRJSON_BENCHMARK_VERSION in source file")


with open("./src/benchmark.c", "r", encoding="utf-8") as f:
    version_string = find_version(f.read())


class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.abspath("build")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        cmake_cmd = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            ".",
            "-B",
            "build",
        ]
        subprocess.check_call(cmake_cmd)

        if os.name == "nt":
            build_cmd = ["cmake", "--build", "build", "--config", "Release"]
        else:
            build_cmd = ["cmake", "--build", "build"]
        subprocess.check_call(build_cmd)

        if os.name == "nt":
            built_filename = "Release/ssrjson_benchmark.dll"
            target_filename = "ssrjson_benchmark.pyd"
        else:
            built_filename = "ssrjson_benchmark.so"
            target_filename = built_filename

        built_path = os.path.join(build_dir, built_filename)
        if not os.path.exists(built_path):
            raise RuntimeError(f"Built library not found: {built_path}")

        target_dir = self.build_lib
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        target_path = os.path.join(target_dir, target_filename)
        self.announce(f"Copying {built_path} to {target_path}")
        print(f"Copying {built_path} to {target_path}")
        shutil.copyfile(built_path, target_path)


setup(
    name="ssrjson_benchmark",
    version=version_string,
    description="Benchmark for ssrJSON",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    ext_modules=[
        Extension(
            "ssrjson_benchmark",
            sources=["src/benchmark.c"],
            language="c",
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
