#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""python api for shmem."""

import os
import shutil
import subprocess
from pathlib import Path
from setuptools import setup, find_namespace_packages
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py

# 消除whl压缩包的时间戳差异
os.environ['SOURCE_DATE_EPOCH'] = '0'
current_version = os.getenv('VERSION', '1.0.0')

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

# ========================
# Custom build_py: Integrate the C++ build process
# ========================
class BuildCppLibs(build_py):
    def run(self):
        self._build_cpp()

        self._copy_libraries_to_package()

        super().run()

    def _build_cpp(self):
        build_dir = Path("build")
        install_dir = Path("install/shmem")

        build_dir.mkdir(exist_ok=True)

        pyexpand_type = os.getenv("PYEXPAND_TYPE", "ON")
        build_type = os.getenv("BUILD_TYPE", "Release")
        use_cxx11_abi = os.getenv("USE_CXX11_ABI", "ON")
        compile_options_str = os.getenv("COMPILE_OPTIONS", "")
        use_mssanitizer = os.getenv("USE_MSSANITIZER", "OFF")
        soc_type = os.getenv("SOC_TYPE", "")
        enable_udma_support = os.getenv("ACLSHMEM_UDMA_SUPPORT", "OFF")

        cmake_cmd = [
            "cmake",
            f"-DBUILD_PYTHON={pyexpand_type}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DUSE_CXX11_ABI={use_cxx11_abi}",
            "-DCMAKE_INSTALL_PREFIX=../install",
            f"-DUSE_MSSANITIZER={use_mssanitizer}",
            f"-DSOC_TYPE={soc_type}",
            f"-DACLSHMEM_UDMA_SUPPORT={enable_udma_support}",
            "-DCMAKE_SKIP_RPATH=TRUE",
        ]

        if compile_options_str.strip():
            cmake_cmd.extend(compile_options_str.split())

        cmake_cmd.append("..")

        try:
            subprocess.check_call(cmake_cmd, cwd=build_dir)
            subprocess.check_call(["make", "install", "-j17"], cwd=build_dir)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C++ build failed: {e}")

        if not (install_dir / "lib").exists():
            raise RuntimeError("C++ build succeeded but 'install/shmem/lib' not found!")

    def _copy_libraries_to_package(self):
        install_lib = Path("install/shmem") / "lib"
        package_src_dir = Path("src/python") / "shmem"

        if not install_lib.exists():
            print("Warning: install/shmem/lib not found, skipping copy")
            return

        package_src_dir.mkdir(parents=True, exist_ok=True)

        for so_file in install_lib.glob("*.so"):
            dst = package_src_dir / so_file.name
            shutil.copy(so_file, dst)
            print(f"Copied {so_file} -> {dst}")

        version_file = Path("install") / "VERSION"
        if version_file.exists():
            shutil.copy(version_file, package_src_dir / "VERSION")


setup(
    name="shmem",
    version=current_version,
    author="",
    author_email="",
    description="python api for shmem",
    packages=find_namespace_packages(where="src/python", exclude=("tests*",)),
    package_dir={"": "src/python"},
    license="Apache License Version 2.0",
    install_requires=["torch-npu"],
    python_requires=">=3.7",
    package_data={
        "shmem": [
            "*.so",
            "VERSION"
        ]
    },
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildCppLibs},
)