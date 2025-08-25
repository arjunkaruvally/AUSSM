"""
All the classes required to run AUSSM in a single file

Glossary:

b - batch size
l - sequence length (time dimension)
d - input dimension
n - hidden size
"""

from __future__ import annotations
import math
import json
import sys
from typing import Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import numpy as np

from PIL.ImageOps import expand
from einops import rearrange, repeat, einsum
import matplotlib.pyplot as plt
from icecream import ic

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import extension_cpp
from sympy.polys.polyoptions import allowed_flags


@dataclass
class ModelArgsSSMau:
    """
    Configures Mamba style block with the AUSSM

    d_model (int): the input dimensions of the SSM
    d_state (int): the hidden state dimension of the SSM
    dt_rank (int): rank of the dt parameter
    d_conv (int): size of the 1d convolution kernel used in the convolution block
    conv_bias (bool): If the convolution block uses bias
    verbose (bool): verbose flag
    conv_1d (bool): whether the convolution block is used in the Mamba block
    cuda (bool): use the optimized cuda kernel. Currently only cuda=True is supported
    """
    d_model: int
    d_state: int
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    verbose: bool = False
    conv_1d: bool = True
    cuda: bool = True

    def __post_init__(self):
        self.block = SSMauBlock
        if self.dt_rank == 'auto':
            self.dt_rank = self.d_state

@dataclass
class ModelArgsMamba:
    """
    Configures a Mamba style block with the AUSSM

    d_model (int): the input dimensions of the SSM
    d_state (int): the hidden state dimension of the SSM
    expand  (int): whether the input dimensions are expanded before processing
    dt_rank (int): rank of the dt parameter
    d_conv (int): size of the 1d convolution kernel used in the convolution block
    bias (bool): If the convolution block uses bias
    verbose (bool): verbose flag
    """
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    verbose: bool = False

    def __post_init__(self):
        self.block = MambaBlock
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


@dataclass
class ModelBlockArgs:
    layers: list[Union[ModelArgsSSMau, ModelArgsMamba]]
    d_model: int


class SSMTS(nn.Module):
    """
    A timeseries processing model with a Mamba+AUSSM backbone

    d_model (int): the input dimensions of the SSM
    input_dim (int): the input dimension of the timeseries
    output_dim (int): the output dimension of the timeseries
    layers (str): a layer string with the configuration of `a` (AUSSM) and `m` (Mamba S6) blocks. eg. string `a|m|a` initializes a
                  three layer backbone with AUSSM -> Mamba (S6) -> AUSSM
    d_state (int): the hidden state dimension of the SSM
    mamba_expand (int): the expansion parameter of Mamba S6
    dt_rank (int): rank of the dt parameter
    d_conv (int): size of the 1d convolution kernel used in the convolution block of Mamba S6
    conv_bias (bool): If the convolution block uses bias
    bias (bool): is bias is used in the layers
    ssmau_cuda (bool): use the optimized cuda kernel for AUSSM. Currently only cuda=True is supported
    ssmau_conv_1d (bool): whether the convolution block is used in the AUSSM block
    embedding
    """
    def __init__(self, d_model: int,
                 input_dim: int,
                 output_dim: int,
                 layers: str=None,
                 d_state: int = 16,
                 mamba_expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 ssmau_cuda: bool = True,
                 ssmau_conv_1d: bool = True,
                 embedding_decay: bool = True,
                 verbose: bool = True):
        super().__init__()
        if layers is None:
            layers = "m|a"

        self.n_layers = len(layers.split('|'))
        self.output_dim = output_dim
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.input_dim = input_dim
        self.mamba_expand = mamba_expand
        self.pad_vocab_size_multiple = 8
        self.conv_bias = conv_bias
        self.verbose = verbose
        self.bias = bias
        self.ssmau_conv_1d = ssmau_conv_1d
        self.embedding_decay = embedding_decay
        self.ssmau_cuda = ssmau_cuda
        # self.mode = mode

        # process layers
        self.args = ModelBlockArgs(self.parse_layer_string(layers), self.d_model)
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.layers = nn.ModuleList([ResidualBlock(layer_arg) for layer_arg in self.args.layers])
        self.norm_f = RMSNorm(self.d_model)

        self.output = nn.Linear(self.d_model, self.output_dim, bias=False)

        if verbose:
            print(f"[MambaSeq2Seq] Intialized with")
            print(f"input_dim: {self.input_dim}")
            print(f"output_dim: {self.output_dim}")

    def parse_layer_string(self, layer_string:str) -> list[Union[ModelArgsSSMau, ModelArgsMamba]]:
        layer_string = layer_string.replace(" ", "")
        layers = layer_string.split("|")

        layer_args = []
        for layer in layers:
            if layer == "a":
                module_args = ModelArgsSSMau(self.d_model, self.d_state, self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose,
                                            conv_1d=self.ssmau_conv_1d, cuda=self.ssmau_cuda)
            elif layer == "m":
                module_args = ModelArgsMamba(self.d_model, self.d_state, self.mamba_expand,
                                            self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose)
            else:
                raise NotImplementedError(f"Layer type \"{layer}\" not recognized")

            layer_args.append(module_args)
        return layer_args

    def diagnostic_mode(self, device=None):
        for layer in self.layers:
            layer.diagnostic_mode(device)

    def initialize(self, *args, **kwargs):
        pass

    def get_parameter_groups_for_optimizer(self):
        if self.embedding_decay:
            return [
                { "params": self.embedding.parameters(), "weight_decay": 0.0 },
                { "params": self.layers.parameters() },
                { "params": self.norm_f.parameters() },
                { "params": self.output.parameters() }
            ]
        else:
            return [
                {"params": self.embedding.parameters()},
                {"params": self.layers.parameters()},
                {"params": self.norm_f.parameters()},
                {"params": self.output.parameters()}
            ]

    def forward(self, input_ids, **kwargs):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, output_dim)
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        logits = self.output(x)

        return logits


class SSMClassifier(nn.Module):
    """
        A sequence classification model with a Mamba+AUSSM backbone

        d_model (int): the input dimensions of the SSM
        input_dim (int): the input dimension of the timeseries
        output_dim (int): the output dimension of the timeseries
        layers (str): a layer string with the configuration of `a` (AUSSM) and `m` (Mamba S6) blocks. eg. string `a|m|a` initializes a
                      three layer backbone with AUSSM -> Mamba (S6) -> AUSSM
        d_state (int): the hidden state dimension of the SSM
        mamba_expand (int): the expansion parameter of Mamba S6
        dt_rank (int): rank of the dt parameter
        d_conv (int): size of the 1d convolution kernel used in the convolution block of Mamba S6
        conv_bias (bool): If the convolution block uses bias
        bias (bool): is bias is used in the layers
        mode (bool): "meanpool|last"
        ssmau_cuda (bool): use the optimized cuda kernel for AUSSM. Currently only cuda=True is supported
        ssmau_conv_1d (bool): whether the convolution block is used in the AUSSM block
        embedding
    """
    def __init__(self, d_model: int,
                 input_dim: int,
                 output_dim: int,
                 layers: str=None,
                 d_state: int = 16,
                 mamba_expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 mode="meanpool",
                 ssmau_conv_1d: bool = True,
                 ssmau_cuda: bool = True,
                 verbose: bool = True):  # two modes are supported - `meanpool` and `last`
        super().__init__()
        if layers is None:
            layers = "m|a"

        self.n_layers = len(layers.split('|'))
        self.num_classes = output_dim
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.mamba_expand = mamba_expand
        self.pad_vocab_size_multiple = 8
        self.conv_bias = conv_bias
        self.verbose = verbose
        self.bias = bias
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.ssmau_conv_1d = ssmau_conv_1d
        # self.mode = mode
        self.meanpool = "meanpool" in mode
        self.ssmau_cuda = ssmau_cuda

        # process layers
        self.args = ModelBlockArgs(self.parse_layer_string(layers), self.d_model)
        self.input_layer = nn.Linear(self.input_dim, self.d_model)
        self.layers = nn.ModuleList([ResidualBlock(layer_arg) for layer_arg in self.args.layers])
        self.norm_f = RMSNorm(self.d_model)

        self.output = nn.Linear(self.d_model, self.num_classes, bias=False)

        if verbose:
            print(f"[MambaSeq2Seq] Intialized with")
            print(f"input_dim: {self.input_dim}")
            print(f"num_classes: {self.num_classes}")

    def parse_layer_string(self, layer_string:str) -> list[Union[ModelArgsSSMau, ModelArgsMamba]]:
        layer_string = layer_string.replace(" ", "")
        layers = layer_string.split("|")

        layer_args = []
        for layer in layers:
            if layer == "a":
                module_args = ModelArgsSSMau(self.d_model, self.d_state, self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose,
                                            conv_1d=self.ssmau_conv_1d, cuda=self.ssmau_cuda)
            elif layer == "m":
                module_args = ModelArgsMamba(self.d_model, self.d_state, self.mamba_expand,
                                            self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose)
            else:
                raise NotImplementedError(f"Layer type \"{layer}\" not recognized")

            layer_args.append(module_args)
        return layer_args

    def get_last(self, x, lengths):
        b, l, d = x.shape
        gather_matrix = repeat(lengths-1, "b -> b l d", b=b, l=1, d=d)
        return x.gather(1, gather_matrix).squeeze()

    def masked_meanpool(self, x, lengths):
        b, l, d = x.shape
        mask = repeat(torch.arange(0, l, device=x.device), "l -> b l d", b=b, l=l, d=d)
        length_function = repeat(lengths, "b -> b l d", b=b, l=l, d=d)
        mask = mask < length_function
        return torch.sum(x*mask, dim=1) / lengths.reshape((-1, 1))

    def diagnostic_mode(self, device=None):
        for layer in self.layers:
            layer.diagnostic_mode(device)

    def initialize(self, *args, **kwargs):
        pass

    def get_parameter_groups_for_optimizer(self):
        optimizer_params = [
            {"params": self.embedding.parameters()},
            # {"params": self.layers.parameters()},
            {"params": self.norm_f.parameters()},
            {"params": self.output.parameters()}
        ]
        for layer in self.layers:
            optimizer_params.append({ "params": layer.mixer.parameters(), "weight_decay": 0.0 })
            optimizer_params.append({ "params": layer.norm.parameters() })
        return optimizer_params

    def forward(self, input_ids, lengths=None, **kwargs):
        """
        Args:
            input_ids (long tensor): shape (b, l, d) (See Glossary at top for definitions of b, l, d, n...)

        Returns:
            logits: shape (b, l, output_dim)
        """
        x = self.input_layer(input_ids)

        for layer in self.layers:
            x = layer(x)

        if lengths is not None:
            if self.meanpool:
                x = self.masked_meanpool(x, lengths)
            else:
                # choose the last index
                x = self.get_last(x, lengths)
        else:  # if lengths not given naive operations are performed
            if self.meanpool:
                x = torch.mean(x, dim=1)
            else:
                x = x[:, -1, :]

        x = self.norm_f(x)

        logits = self.output(x)

        return logits


class SSMSequenceClassifier(nn.Module):
    """
        A sequence classification model with a Mamba+AUSSM backbone, where the sequence has a specific language.

        d_model (int): the input dimensions of the SSM
        vocab_size (int): the vocabulary size of the sequence
        input_dim (int): the input dimension of the sequence
        output_dim (int): the output dimension of the sequence (number of classes)
        layers (str): a layer string with the configuration of `a` (AUSSM) and `m` (Mamba S6) blocks. eg. string `a|m|a` initializes a
                      three layer backbone with AUSSM -> Mamba (S6) -> AUSSM
        d_state (int): the hidden state dimension of the SSM
        mamba_expand (int): the expansion parameter of Mamba S6
        dt_rank (int): rank of the dt parameter
        d_conv (int): size of the 1d convolution kernel used in the convolution block of Mamba S6
        conv_bias (bool): If the convolution block uses bias
        bias (bool): is bias is used in the layers
        mode (bool): "meanpool|last"
        ssmau_cuda (bool): use the optimized cuda kernel for AUSSM. Currently only cuda=True is supported
        ssmau_conv_1d (bool): whether the convolution block is used in the AUSSM block
        embedding
    """
    def __init__(self, d_model: int,
                 vocab_size: int,
                 output_dim: int,
                 layers: str=None,
                 d_state: int = 16,
                 mamba_expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 pad_vocab_size_multiple: int = 8,
                 conv_bias: bool = True,
                 bias: bool = False,
                 mode="meanpool",
                 ssmau_conv_1d: bool = True,
                 embedding_decay: bool = True,
                 verbose: bool = True):  # two modes are supported - `meanpool` and `last`
        super().__init__()
        if layers is None:
            layers = "m|a"

        self.n_layers = len(layers.split('|'))
        self.num_classes = output_dim
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.mamba_expand = mamba_expand
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.conv_bias = conv_bias
        self.verbose = verbose
        self.bias = bias
        self.vocab_size = vocab_size
        self.ssmau_conv_1d = ssmau_conv_1d
        self.embedding_decay = embedding_decay
        # self.mode = mode
        self.meanpool = "meanpool" in mode

        # process layers
        self.args = ModelBlockArgs(self.parse_layer_string(layers), self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([ResidualBlock(layer_arg) for layer_arg in self.args.layers])
        self.norm_f = RMSNorm(self.d_model)

        self.output = nn.Linear(self.d_model, self.num_classes, bias=False)

        if verbose:
            print(f"[MambaSeq2Seq] Intialized with")
            print(f"in_vocab_size: {self.vocab_size}")
            print(f"num_classes: {self.num_classes}")

    def parse_layer_string(self, layer_string:str) -> list[Union[ModelArgsSSMau, ModelArgsMamba]]:
        layer_string = layer_string.replace(" ", "")
        layers = layer_string.split("|")

        layer_args = []
        for layer in layers:
            if layer == "a":
                module_args = ModelArgsSSMau(self.d_model, self.d_state, self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose,
                                            conv_1d=self.ssmau_conv_1d)
            elif layer == "m":
                module_args = ModelArgsMamba(self.d_model, self.d_state, self.mamba_expand,
                                            self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose)
            else:
                raise NotImplementedError(f"Layer type \"{layer}\" not recognized")

            layer_args.append(module_args)
        return layer_args

    def get_last(self, x, lengths):
        b, l, d = x.shape
        gather_matrix = repeat(lengths-1, "b -> b l d", b=b, l=1, d=d)
        return x.gather(1, gather_matrix).squeeze()

    def masked_meanpool(self, x, lengths):
        b, l, d = x.shape
        mask = repeat(torch.arange(0, l, device=x.device), "l -> b l d", b=b, l=l, d=d)
        length_function = repeat(lengths, "b -> b l d", b=b, l=l, d=d)
        mask = mask < length_function
        return torch.sum(x*mask, dim=1) / lengths.reshape((-1, 1))

    def diagnostic_mode(self, device=None):
        for layer in self.layers:
            layer.diagnostic_mode(device)

    def initialize(self, *args, **kwargs):
        pass

    def get_parameter_groups_for_optimizer(self):
        if self.embedding_decay:
            return [
                { "params": self.embedding.parameters(), "weight_decay": 0.0 },
                { "params": self.layers.parameters() },
                { "params": self.norm_f.parameters() },
                { "params": self.output.parameters() }
            ]
        else:
            return [
                {"params": self.embedding.parameters()},
                {"params": self.layers.parameters()},
                {"params": self.norm_f.parameters()},
                {"params": self.output.parameters()}
            ]

    def forward(self, input_ids, lengths=None, **kwargs):
        """
        Args:
            input_ids (long tensor): shape (b, l) (See Glossary at top for definitions of b, l, d, n...)

        Returns:
            logits: shape (b, l, output_dim)
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        if lengths is not None:
            if self.meanpool:
                x = self.masked_meanpool(x, lengths)
            else:
                # choose the last index
                x = self.get_last(x, lengths)
        else:  # if lengths not given naive operations are performed
            if self.meanpool:
                x = torch.mean(x, dim=1)
            else:
                x = x[:, -1, :]

        x = self.norm_f(x)

        logits = self.output(x)

        return logits


## There is a compile time error which occurs only during backpropagation and when compiled which is almost impossible to debug without looking into
# what pytorch does during compilation.
## It is possibly due to complex operators in the custom operator.
## DO NOT USE COMPILE HERE: @torch.compile
class SSMSeq2Seq(nn.Module):
    """
        A sequence2sequence model with a Mamba+AUSSM backbone

        d_model (int): the input dimensions of the SSM
        vocab_size (int): the vocabulary size of the input sequence
        output_vocab_size (int): the vocabulary size of the output sequence
        layers (str): a layer string with the configuration of `a` (AUSSM) and `m` (Mamba S6) blocks. eg. string `a|m|a` initializes a
                      three layer backbone with AUSSM -> Mamba (S6) -> AUSSM
        d_state (int): the hidden state dimension of the SSM
        mamba_expand (int): the expansion parameter of Mamba S6
        dt_rank (int): rank of the dt parameter
        d_conv (int): size of the 1d convolution kernel used in the convolution block of Mamba S6
        conv_bias (bool): If the convolution block uses bias
        bias (bool): is bias is used in the layers
        mode (bool): "meanpool|last"
        ssmau_cuda (bool): use the optimized cuda kernel for AUSSM. Currently only cuda=True is supported
        ssmau_conv_1d (bool): whether the convolution block is used in the AUSSM block
        embedding
    """
    def __init__(self, d_model: int,
                 vocab_size: int,
                 output_vocab_size: int,
                 layers: str=None,
                 d_state: int = 16,
                 mamba_expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 ssmau_conv_1d: bool = True,
                 embedding_decay: bool = True,
                 ssmau_cuda: bool = True,
                 verbose: bool = True):  # two modes are supported - `meanpool` and `last`
        super().__init__()
        if layers is None:
            layers = "m|a"

        self.n_layers = len(layers.split('|'))
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.mamba_expand = mamba_expand
        self.pad_vocab_size_multiple = 8
        self.conv_bias = conv_bias
        self.verbose = verbose
        self.bias = bias
        self.vocab_size = vocab_size
        self.ssmau_conv_1d = ssmau_conv_1d
        self.embedding_decay = embedding_decay
        self.ssmau_cuda = ssmau_cuda

        # process layers
        self.args = ModelBlockArgs(self.parse_layer_string(layers), self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([ResidualBlock(layer_arg) for layer_arg in self.args.layers])
        self.norm_f = RMSNorm(self.d_model)
        self.output_vocab_size = output_vocab_size  # DO NOT change this variable name: this is expected by the task

        self.output = nn.Linear(self.d_model, self.output_vocab_size, bias=False)

        if verbose:
            print(f"[SSMSeq2Seq] Intialized with")
            print(f"in_vocab_size: {self.vocab_size}")
            print(f"out_vocab_size: {self.output_vocab_size}")

    def parse_layer_string(self, layer_string:str) -> list[Union[ModelArgsSSMau, ModelArgsMamba]]:
        layer_string = layer_string.replace(" ", "")
        layers = layer_string.split("|")

        layer_args = []
        for layer in layers:
            if layer == "a":
                module_args = ModelArgsSSMau(self.d_model, self.d_state, self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose,
                                            conv_1d=self.ssmau_conv_1d, cuda=self.ssmau_cuda)
            elif layer == "m":
                module_args = ModelArgsMamba(self.d_model, self.d_state, self.mamba_expand,
                                            self.dt_rank,
                                            self.d_conv, self.conv_bias, self.bias, self.verbose)
            else:
                raise NotImplementedError(f"Layer type \"{layer}\" not recognized")

            layer_args.append(module_args)
        return layer_args

    def diagnostic_mode(self, device=None):
        for layer in self.layers:
            layer.diagnostic_mode(device)

    def initialize(self, *args, **kwargs):
        pass

    def get_parameter_groups_for_optimizer(self):
        if self.embedding_decay:
            return [
                { "params": self.embedding.parameters(), "weight_decay": 0.0 },
                { "params": self.layers.parameters() },
                { "params": self.norm_f.parameters() },
                { "params": self.output.parameters() }
            ]
        else:
            return [
                {"params": self.embedding.parameters()},
                {"params": self.layers.parameters()},
                {"params": self.norm_f.parameters()},
                {"params": self.output.parameters()}
            ]

    def forward(self, input_ids, **kwargs):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, output_dim)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        logits = self.output(x)

        return logits

"""
The following classes are adapted from https://github.com/johnma2006/mamba-minimal
"""


class ResidualBlock(nn.Module):
    def __init__(self, args: Union[ModelArgsSSMau,ModelArgsMamba]):
        """Simple block wrapping Mamba or AUSSM blocks with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = args.block(args)
        self.norm = RMSNorm(args.d_model)

    def diagnostic_mode(self, device=None):
        self.mixer.diagnostic_mode(device)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgsMamba):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        # self.initialize_model_specific_parameters(args)
        self.x_proj = nn.Linear(args.d_inner,
                                (args.dt_rank + args.d_state * 2),
                                bias=False)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(args.d_inner))
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        print(f"Available parameters in {self.__class__.__name__}:")
        for named_module in self.named_modules():
            print(named_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the mixed SSM block

        :param x:
        :return:
        """
        (b, l, d) = x.shape

        x_and_z = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, z) = x_and_z.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        y = self.ssm_mamba(x, z)

        output = self.out_proj(y)

        return output

    def ssm_mamba(self, x, z):
        (b, l, d) = x.shape

        x_proj = self.x_proj(x)

        if self.args.d_model == 0:
            return torch.Tensor([])

        b, l, d_in = x.shape
        n = self.args.d_state

        dt, B, C = x_proj.split([self.args.dt_rank,
                                 self.args.d_state,
                                 self.args.d_state], dim=-1)

        dt = self.dt_proj(dt)
        x = F.silu(x)
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        dt_bias = self.dt_proj.bias.float()

        y = selective_scan_fn(
            rearrange(x, "b l d -> b d l").contiguous(),
            rearrange(dt, "b l d -> b d l", l=l).contiguous(),
            A.contiguous(),
            rearrange(B, "b l n -> b n l", l=l).contiguous(),
            rearrange(C, "b l n -> b n l", l=l).contiguous(),
            self.D.float().contiguous(),
            z=rearrange(z, "b l d -> b d l").contiguous(),
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False
        )

        y = rearrange(y, "b d l -> b l d")
        return y


class SSMauBlock(nn.Module):
    def __init__(self, args: ModelArgsSSMau):
        """A single AUSSM block."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_model * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_model,
            out_channels=args.d_model,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_model,
            padding=args.d_conv - 1,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=args.bias)
        # self.initialize_model_specific_parameters(args)
        self.x_proj = nn.Linear(args.d_model, args.dt_rank,
                                bias=False)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_model, bias=True)
        self.D = nn.Parameter(torch.ones(args.d_model))
        self.C = nn.Parameter(torch.randn(args.d_state, dtype=torch.complex64), requires_grad=True)
        self.B = nn.Parameter(torch.randn(args.d_state, dtype=torch.complex64), requires_grad=True)
        self.xA_proj = nn.Linear(args.d_model, args.d_model * args.d_state, bias=True)
        self.xA_proj.bias.data.uniform_(-torch.pi, torch.pi)

        if self.args.cuda:
            self.optimized_SSMa_function = lambda *args_x, **kwargs_x: extension_cpp.ops.ssm_adaptive_interface(
                *args_x,
                **kwargs_x,
                unitary=True)
        else:
            self.optimized_SSMa_function = torch.compile(
                extension_cpp.ops.reference_ssm_adaptive_unitary_optimized_polar)

        print(f"Available parameters in {self.__class__.__name__}:")
        for named_module in self.named_modules():
            print(named_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the mixed SSM block

        :param x:
        :return:
        """
        (b, l, d) = x.shape

        x_and_z = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, z) = x_and_z.split(split_size=[self.args.d_model, self.args.d_model], dim=-1)

        if self.args.conv_1d:
            x = rearrange(x, 'b l d_in -> b d_in l')
            x = self.conv1d(x)[:, :, :l]
            x = rearrange(x, 'b d_in l -> b l d_in')

        y = self.ssm_au(x, z)

        output = self.out_proj(y)

        return output

    def ssm_au(self, x, z):
        x_proj = self.x_proj(x)

        b, l, d_in = x.shape
        n = self.args.d_state

        dt = self.dt_proj(x_proj)
        dt = dt + self.dt_proj.bias.float()
        dt = F.softplus(dt)

        x_param = rearrange(self.xA_proj.weight, "(n dstate1)  dstate -> dstate1 n dstate", n=n,
                            dstate=d_in,
                            dstate1=d_in)

        x_bias = rearrange(self.xA_proj.bias, "(n dstate1) -> dstate1 n", n=n, dstate1=d_in)

        self.ssm_args = {
            "u": rearrange(x, "b l d -> b d l").contiguous(),
            "dt": rearrange(dt, "b l d -> b d l", l=l).contiguous(),
            "x": x_param.contiguous() * (1 / np.sqrt(self.args.d_model)),
            "x_bias": x_bias.contiguous(),
            "B": self.B.contiguous(),
            "C": self.C.contiguous(),
            "D": self.D.float().contiguous(),
            "z": rearrange(z, "b l d -> b d l").contiguous()
        }

        y = self.optimized_SSMa_function(**self.ssm_args)

        y = rearrange(y, "b d l -> b l d")
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

