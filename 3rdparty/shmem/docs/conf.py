# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import subprocess
import warnings
import os
import sphinx.locale
from sphinx.locale import admonitionlabels

admonitionlabels['remark'] = ('', '')

GIT_COMMAND = """\
git symbolic-ref -q --short HEAD \
|| git describe --tags --exact-match 2> /dev/null \
|| git rev-parse HEAD"""
try:
    branch = subprocess.check_output(["/bin/bash", "-c", GIT_COMMAND]).strip().decode()
except subprocess.CalledProcessError:
    branch = "main"
    warnings.warn("获取Git分支失败，使用默认分支main")

PROJECT = "SHMEM Guidebook"
AUTHOR = "Ascend"
COPYRIGHT_INFO = " 2025 Huawei Technologies Co., Ltd."
RELEASE = "1.0.0"
HTML_SHOW_SPHINX = False

project = PROJECT 
author = AUTHOR
copyright = COPYRIGHT_INFO
release = RELEASE
version = RELEASE.split('.')[0]

extensions = [
    'myst_parser',
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

myst_heading_anchors = 3
myst_enable_extensions = [
    "linkify",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'sticky_navigation': True,
    'logo_only': False,
}

toc_depth = 4
toc_object_entries_show_parents = 'hide'

html_static_path = ["_static"]
html_css_files = [
    'css/custom.css',
]

breathe_projects = {"ACLSHMEM_CPP_API": f"./{branch}/xml"}
breathe_default_project = "ACLSHMEM_CPP_API"

if not os.path.exists('_static/css'):
    os.makedirs('_static/css', exist_ok=True)

css_content = """
.rst-content .admonition.remark .admonition-title {
    display: none !important;
}

.rst-content .admonition.remark {
    font-weight: bold !important;
}

.rst-content .admonition.remark code,
.rst-content .admonition.remark pre {
    font-weight: normal !important;
}

.rst-content .admonition.remark {
    background: #f0f9ff !important;
    border-left: 5px solid #3b82f6 !important;
    border-radius: 8px;
    margin: 20px 0;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
}
"""

css_path = '_static/css/custom.css'
with open(css_path, 'w', encoding='utf-8') as f:
    f.write(css_content)