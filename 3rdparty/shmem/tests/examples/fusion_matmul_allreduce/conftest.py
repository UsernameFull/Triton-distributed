# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import pytest


def pytest_addoption(parser):
    "Pytest hook to add command line options."
    parser.addoption(
        "--executable_path",
        action="store",
        default="tests/bin/matmul_allreduce",
        help="Path to the matmul_allreduce executable",
    )
    parser.addoption(
        "--test_data_dir",
        action="store",
        default="tests/test_data/matmul_allreduce",
        help="Directory to store persistent test data",
    )


@pytest.fixture
def executable_path(request):
    "Fixture for the matmul_allreduce executable path."
    return request.config.getoption("--executable_path")


@pytest.fixture
def test_data_dir(request):
    "Fixture for the test data directory."
    path = request.config.getoption("--test_data_dir")
    os.makedirs(path, exist_ok=True)
    return path
