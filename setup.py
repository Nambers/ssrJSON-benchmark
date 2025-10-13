import os
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def run_check(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"command failed: `{' '.join(cmd)}`")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"command failed: `{' '.join(cmd)}`")
        print(f"error: {e}")
        raise

class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.abspath("build")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        run_check([
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            ".",
            "-B",
            "build",
        ])

        if os.name == "nt":
            build_cmd = ["cmake", "--build", "build", "--config", "Release"]
        else:
            build_cmd = ["cmake", "--build", "build"]
        run_check(build_cmd)

        if os.name == "nt":
            built_filename = "Release/_ssrjson_benchmark.dll"
            target_filename = "_ssrjson_benchmark.pyd"
        else:
            built_filename = "_ssrjson_benchmark.so"
            target_filename = built_filename

        built_path = os.path.join(build_dir, built_filename)
        if not os.path.exists(built_path):
            raise RuntimeError(f"Built library not found: {built_path}")

        target_dir = self.build_lib + "/ssrjson_benchmark"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        target_path = os.path.join(target_dir, target_filename)
        self.announce(f"Copying {built_path} to {target_path}")
        print(f"Copying {built_path} to {target_path}")
        shutil.copyfile(built_path, target_path)


setup(
    ext_modules=[
        Extension(
            "_ssrjson_benchmark",
            sources=["src/_ssrjson_benchmark.c"],
            language="c",
        )
    ],
    packages=["ssrjson_benchmark", "ssrjson_benchmark._files"],
    package_dir={"": "src"},
    package_data={
        "ssrjson_benchmark": ["template.md"],
        "ssrjson_benchmark._files": ["*.json"],
    },
    include_package_data=True,
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
