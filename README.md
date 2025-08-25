# Bridging Expressivity and Scalability with Adaptive Unitary SSMs

This is the code base implementing the Adaptive Unitary SSM (AUSSM) introduced in our paper [Bridging Expressivity and Scalability with Adaptive Unitary SSMs](https://arxiv.org/abs/2507.05238).
We have released our custom optimized CUDA kernel for AUSSM that can be used to build SSM backbones.

The current list of packages are:

- `ai-modules` - this is the master collection of models, tasks and datasets that are used to run the experiments.
- `extension-cpp` - this package contains a collection of pytorch operators for the AUSSM kernel that extend the existing functionality of pytorch with cuda codes. Currently we have only GPU implementations.

## Usage

Once installed, the `wavesAI.model` module contains all the necessary model classes. We provide some models specialized 
for some tasks, but these are easily extended to any task. We have following model implementations: 

- `SSMTS` - processing multivariate timeseries 
- `SSMClassifier` - processes a timseries and returns the logits for classification
- `SSMSequenceClassifier` - similar to the above classifier, but with an Embedding layer to process sequences of a given language
- `SSMSeq2Seq` - processes sequences of a given language and converts into another sequence. The output sequence logits are returned

All these tasks have a layers argument that sets the configuration of the SSM backbone. Currently, for the hybrid 
AUSSM models, the layer configuration is a string of the following format "<a/m>|<a/m>..." where `a` means the AUSSM block 
and `m` means a standard Mamba block.

Any of the above models can be used directly in any compatible task. See an example below for SSMTS

```python
from wavesAI.model.aussm import SSMTS
import torch

model = SSMTS(d_model=64, input_dim=8, output_dim=8, layers="a|m")
model.to(torch.device("cuda:0"))

batch_size = 16
sequence_length = 256
x = torch.normal(0, 1, (8, sequence_length, 8), device=torch.device("cuda:0"))

print(model(x))
print(model(x).shape)  ## (8, 256, 8)
```

If you want to directly use the blocks instead of the full model, the block classes can be directly initialized.

- `MambaBlock`: creates a single standard Mamba block with the S6 SSM. The block can be configured using the class `ModelArgsMamba`
- `SSMauBlock`: The Adaptive and Unitary SSM (AUSSM) block. The block can be configured using the class `ModelArgsSSMau` 

below we provide a sample implementation of the AUSSM block.

```python
from wavesAI.model.aussm import SSMauBlock, ModelArgsSSMau
import torch

model = SSMauBlock(ModelArgsSSMau(d_model=8, d_state=4))
model.to(torch.device("cuda:0"))

batch_size = 16
sequence_length = 256
x = torch.normal(0, 1, (8, sequence_length, 8), device=torch.device("cuda:0"))

print(model(x))
print(model(x).shape)  ## (8, 256, 8)
```

## Setting the environment

To simplify setup, every required path is set using environment variables. Use the template in the `.sample_env` file to 
configure the environment to run correctly in your system. Once the variables are configured, save in a `.env` file (this is configured to be ignored by git) 
add the variables to the environment using the command `source .env`

## Installation

The CUDA operator makes creating a universal installation script almost impossible.
I have thus created makefile targets for the different installation options.
I have tested working on `pytorch==2.4.0+cu12.1`. Follow the steps below

NOTE: Virtual environments can be finicky to work with. There are some torch compilations that cause issues in the 
build process. Follow the steps exactly.

1. verify and set all the environment variables in the .env file `source .env`
2. create a **fresh** virtual environment and install the pytorch and cuda version and activate it
3. If the pytorch or cuda version is different from above, update the constraints.txt file accordingly
4. do `make build` to install all the required packages. If this does not work for any reason, check the Makefile and run each command.

## Testing

`pytest` is used to test the packages. There are currently test cases to verify if the cuda kernels are installed correctly.
Do `pytest extension-cpp/.` to run the tests. All the tests should pass.

## Data Download

The timeseries classification and regression experiments are run on publicly available datasets. The timeseries
classification dataset is downloaded on-demand using an open source library [aeon-toolkit](https://github.com/aeon-toolkit/aeon).
This is handled by the respective Datamodule, so manual download is not required.

The weather dataset can be downloaded from [here](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing).
Save it as `weather.csv` in the `DATA_ROOT` directory (this is one of the environment variables).

## Running experiments

- To run the scaling benchmarks of the cuda kernel, do `make cuda_benchmark`
- To run the algorithmic tasks, do `make algorithmic_tasks`
- To run the Time series classification tasks, do `make ts_classification`
- To run the Time series regresssion benchmark, do `make ts_regression`
