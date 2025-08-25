# Copyright (c) 2023, Albert Gu, Tri Dao.
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil
import glob

from setuptools import setup, find_packages, Extension
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path

PACKAGE_NAME = "wavesAI"

setup(name=PACKAGE_NAME,
      version='0.0.6',
      description='Python Modules for Wave RNNs',
      author='all authors',
      author_email='anonymous@gmail.com',
      packages=find_packages()
)
