# Wave SSM

The repo is intended to train, benchmark, and test recurrent models on toy and standard benchmarks. 
The repo is modularized into three aspects of training neural network models - model, data, task. 

For rapid experiment creation, experiment train/validate/test/predict scripts are just YAML configuration files 
which are executed using the PytorchCLI library (I suggest reading the basics here: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

## Installation

I use `conda` and `pip` to manage packages. conda handles the virtual environment and most of the base packages.
pip handles the packages that are currently not in conda. To install the packages, choose one of the following options:

### Option 1

install the `requirements.txt` file provided `conda install --file requirements.txt`. This works most often, but in
case it doesn't manually install packages following Option 2.

### Option 2

- install conda and create a virtual environment
- install pytorch (>=2.4.1): https://pytorch.org/get-started/locally/
- install pythorch-lightning (>=2.4.0) https://lightning.ai/docs/pytorch/stable/starter/installation.html
- install litdata (0.2.29) https://github.com/Lightning-AI/litdata
- install wandb (0.18.3)  https://docs.wandb.ai/quickstart
- install pytorch-lightning extras (this gives the CLI I use for YAML experiments) https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html
- install the wavesAI package in editable mode (this is where all the tasks, data, model scripts are). From the 
  project root do `pip install -e .` (Dont forget the `e`, otherwise you cannot edit the contents without reinstalling eveytime)
- install `mamba_ssm` (2.2.4) https://github.com/state-spaces/mamba  (optional) required for running mamba models
- install `datasets` (3.2.0) https://huggingface.co/docs/datasets/en/index (optional, but required for running lra experiments)
- install `tokenizers` (0.20.1) https://huggingface.co/docs/tokenizers/index (optional, but required for running lra experiments)

## Organization

The folders in the project are organized as follows:

- `wavesAI`: This is the python module containing all the moving parts in the experiments. 
  - `data`: folder contains the necessary data handlers as `LightningDataModule`s. These are scalable data management routines.
  - `model`: will contain all current/future models wre are interested in testing. Each model will be a `torch.nn.Module`
  - `task`: our workflows are typically single tasks that have different data, which we want to test on different models and
    training approaches. The task folder contains tasks organized as `LightningModules`. This is where the data, models
    are combined to solve the task.

## Workflow

One a general idea of the organization is obtained, we can easily extend the framework to create new experiments. Below
I show some workflows that can be followed to implement something in the project. Each workflow shows the intention 
of the research.

Foreword: A single experiment is described as a `YAML` file which contains any and all configurations necessary to 
successfully run the experiment. See a sample config file `experiments/sample_config.yaml`. The parameters in the
config file can be overloaded, which is what current hyperparameter sweeps do in (See the sample slurm script in 
`experiments/exp_0`). The `runner.py` file in the experiments directory initializes the `PytorchCLI` which enables
setting up experiments as YAML files without any hassle (and this is how industries set up large scale experiments). 

### I want to setup an experiment that trains/validates/tests/predicts a model X, on a task Y with the dataset Z.

The goal here should be to create a YAML file with the necessary configuration to run the experiment. I recommend 
reading how the lightning CLI is set up: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

1. First, make sure the three components (X, Y, Z) are coded up and available in the `wavesAI` python module.
   Follow the workflows below for creating the components when necessary.
2. Setup a configuration file with the required setup. The provided sample_config `experiments/sample_config.yaml` can be cloned to create a new one.
   - The parameter `model.class_path` defines the task Y (note that the term model is overloaded by the lightning folks).
     The parameters for the task are given in the subsequent YAML key:value pairs
   - One of these parameters is `model.network` which will be the model we want (the X).
   - There is a YAML key called `data` which indicates the DataModule (Z) to be used.
3. now the experiment can be run using `python runner.py <fit/validate/test/predict> -c <path to YAML config>` and done.

### I want to add a new model X.

1. Networks like waveRNNs should be implemented as new `torch.nn.Module`s. 
2. The function parameters in the `__init__` function can be set in the YAML experiment configuration files.
3. create an initialize function for the module with the signature `initialize(batch_size: int, device: torch.device)`
4. implement the forward loop in the `__call__` function
5. That is it. The model can be called in the experiments by setting the `model.class_path` key with the full import name.

### I want to add a new datamodule Z.

1. See if the current `sequence.py` can be modified in some way to be used here, else
2. create the new dataset as a `LightningDataModule` object
3. That is it, the datamodule can be imported by setting the `data.class_path` key with the full import name of the Data

### I want to add a new Task Y

1. Tasks are in the `wavesAI/task` directory and implemented as `LightningModules`
2. Easiest way is to use the existing tasks as a blueprint and do only the necessary edits for the new task.

### I want to create a dataset used in the datamodule

1. The data should be prepared in advanced to avoid doing unecessary compute during training, and to avoid 
   overloading the RAM. The `data_preparation` folder has sample scripts to do this. 
2. Only a single function needs to be modified in the script to create a new type of dataset. The function should compute
   a _single_ sample in the dataset. All the other details like minibatching, loading, etc. are handled by lightning.

### I want to use Tensorboad or other loggers instead of wandb

1. Currently, I have set the wandb logger in the YAML script. There is nothing preventing changing it.
2. change the `trainer.logger` to one of the loggers offered by pytorch lightning (which covers most loggers out there) https://lightning.ai/docs/pytorch/stable/extensions/logging.html
3. Change also the logger parameters to match the chosen logger.

# Known Bugs/Weird Behaviors

Before going deep in code to debug, please check this list of known bugs or weird behaviors resulting from 
the choice of libraries I used in the project. Hopefully, this will be resolved by the the respective 
library contributors in future updates:

- **YAML file** Verify when formatting the YAML files that strings `yes, no, true, false, on, off` will be converted
  to their respective **boolean** values. Very weird, sadly it is a "feature" of the PyYAML See https://stackoverflow.com/questions/36463531/pyyaml-automatically-converting-certain-keys-to-boolean-values.
- **Raising Exceptions Programmatically** when raising exceptions programmatically, ensure some "error" message is provided,
  else pytorchCLI will raise very unintelligible error messages.
