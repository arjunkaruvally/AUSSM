import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "extension_cpp"

GPU_TARGETS = [ 75, 90 ]  # adding compile support for any of the cuda capability higher than 75

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if use_cuda:
        print("Compiling for CUDA")
    else:
        print("WARNING: Compiling only for CPU. CUDA kernels will not work")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            # "-Weverything",  # warns when implicit conversions happen in the c++ compilation (useful for debugging)
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo"
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])
    else:
        pass
        # for compute_code in GPU_TARGETS:
        #     extra_compile_args["nvcc"].insert(0, "-gencode arch=compute_"+str(compute_code)+",code=sm_"+str(compute_code))

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    print(extra_compile_args)

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.3",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="cuda kernels for adaptive mamba implementations",
    author='anonymous',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={'build_ext': BuildExtension}
)
