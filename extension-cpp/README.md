# Custom CUDA Pytorch operators for general SSMs

The repo contains operators to build general SSM models. The SSMs are generalized to arbitrary diagonal matrices
and accelerated using the work efficient parallel scan algorithm. Currently, the following kernels are written and tested
for sequence lengths upto 4000. More than that sequence lengths will require calling the cuda kernels in batches. 

- `ssm_adaptive_unitary` accessible via extension cpp after installing the package. Use `ssm_adaptive_interface` for proper type checking. In the interest of readability code/compiler level optimizations of the kernel are not performed.

# Installation

## From source

Building from source has a specific order that needs to be followed so that 
pip install does not exhibit weird dependency resolving behavior that will
break some of these packages

NOTE: the operators in this package are CUDA operators, so working CUDA is required
for installing and using the package

1. Use any python environment tool to create a fresh environment
2. Install pip
3. Install [pytorch 2.4.0 with cuda 12.4](https://pytorch.org/get-started/previous-versions/#v241).
4. First, install the extension-cpp package which contains the cuda operator `USE_CUDA=1 pip install --no-build-isolation --no-index -v extension-cpp/.`
5. Second, install mamba `pip install -c constraints.txt --no-build-isolation -v mamba-ssm`
6. Install all the other dependencies `pip install -c constraints.txt -r requirements.txt`
7. Install ai-modules which contains the python packages `pip install -c constraints.txt -v ai-modules/.`

To test the CUDA Operator:
```
pytest -s extension-cpp
```

### Notes on installing custom CUDA operator

The optimized version of the ssm operator is written as a cuda kernel. For correct compilation:

- set environment variable `USE_CUDA=1` 
- set environment variable `DEBUG=0/1` depending on choice. This will add the debug compile time flags.
- load correct gcc version (I use 9.4.0). This will come up when installing the package.
- set the environment variable `TORCH_CUDA_ARCH_LIST` to a `;` separated list of cuda architectures to compile the cuda operator for multiple GPU architectures. I use `export TORCH_CUDA_ARCH_LIST="Turing;Ampere+Tegra;Ampere;Ada;Hopper;Volta"` to compile for all the architectures available in my cluster.

## Authors

[Arjun Karuvally](https://arjunkaruvally.github.io/)
