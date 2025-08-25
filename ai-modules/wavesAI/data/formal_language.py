import os
import random
import shutil
import sys
import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset, IterableDataset
import torch
import numpy as np
from pathlib import Path

# from texttable import Texttable, get_color_string, bcolors
from rich.console import Console
from rich.table import Table
from rich import print

from icecream import ic
import inspect
from datetime import datetime

import sympy
import pandas as pd

# ic.configureOutput(includeContext=True)

###################### Helper Classes

class ExpressionNode:
    def __init__(self, data):
        self.data = data
        self.left_child = None
        self.operator = None
        self.right_child = None

class Equation:
    """
    Generates a CFG of the following form and randomly sample from it

    S -> L = R
    L -> Expression
    R -> Expression
    """
    def __init__(self, number_of_literals, seed: int=0):
        self.number_of_literals = number_of_literals
        rng1 = random.Random(seed+1)
        rng2 = random.Random(seed+2)
        self.left_expression = Expression(number_of_literals, rng1)
        self.right_expression = Expression(number_of_literals, rng2)
        self.rng = random.Random(seed)

    def new_equation(self):
        self.left_expression.new_tree()
        self.right_expression.new_tree()

    def sample_equation_of_length(self, length):
        left_length = self.rng.randint(1, length-2)
        right_length = length - left_length - 1

        self.left_expression.sample_expression_of_length(left_length)
        self.right_expression.sample_expression_of_length(right_length)

        # x will only be present in the LHS
        all_literals = self.left_expression.literal_nodes

        # assign one of the literals as the unknown
        self.rng.choice(all_literals).data = "x"

        return f"{Expression.parse_tree(self.left_expression.root)} = {Expression.parse_tree(self.right_expression.root)}"

class Expression:
    """
    Generates a CFG of the following form and randomly sample from it.

    R -> ( R ) | R + R | R - R | R * R | C
    C - > 0 1 ... number_of_literals
    """
    def __init__(self, number_of_literals, rng:random.Random=None):
        self.number_of_literals = number_of_literals
        self.operators = [ "+", "-", "*" ]

        self.new_tree()

        if rng is None:
            self.rng = random.Random()
        else:
            self.rng = rng

    def new_tree(self):
        self.root = ExpressionNode("R")
        self.expandable_nodes = [self.root]
        self.literal_nodes = []
        self.expression_length = 1

    def random_expand(self):
        node_to_expand = self.rng.choice(self.expandable_nodes)
        node_to_expand.operator = self.rng.choice(self.operators)

        node_to_expand.left_child = ExpressionNode("R")
        node_to_expand.right_child = ExpressionNode("R")
        self.expandable_nodes.append(node_to_expand.left_child)
        self.expandable_nodes.append(node_to_expand.right_child)
        self.expandable_nodes.remove(node_to_expand)

        self.expression_length+=4

    @staticmethod
    def parse_tree(root):
        # ic(root)
        if root.data != "R":
            return str(root.data)

        left_expression = Expression.parse_tree(root.left_child)
        right_expression = Expression.parse_tree(root.right_child)
        return f"( {left_expression} {root.operator} {right_expression} )"

    def collapse_tree(self):
        for node_to_expand in self.expandable_nodes:
            node_to_expand.data = self.rng.randint(0, self.number_of_literals-1)
            self.literal_nodes.append(node_to_expand)
        self.expandable_nodes = [ ]

    def sample_expression_of_length(self, length):
        while self.expression_length + 4 < length:
            self.random_expand()

        self.collapse_tree()

        return Expression.parse_tree(self.root)

###################################### Formal Language Tasks


class Majority(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 64
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 1)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 1

        self.baseline_accuracy = 1/self.output_vocab_size

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = length - 1
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length] = torch.randint(0, self.output_vocab_size, (input_length,), generator=rand_gen)

            unique_items, counts = seq[:input_length].unique(return_counts=True)
            seq[input_length] = unique_items[counts.argmax()]

            # repetition logic
            seq[input_length + 1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Majoritycount(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 64
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 1)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 1

        self.baseline_accuracy = 1/self.output_vocab_size

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = length - 1
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length] = torch.randint(0, self.output_vocab_size, (input_length,), generator=rand_gen)

            unique_items, counts = seq[:input_length].unique(return_counts=True)
            seq[input_length] = counts.max()

            # repetition logic
            seq[input_length + 1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Solveequation(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 14
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 9)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1
        self.vocab_token_to_idx["="] = self.vocab_size - 2
        self.vocab_token_to_idx["+"] = self.vocab_size - 3
        self.vocab_token_to_idx["-"] = self.vocab_size - 4
        self.vocab_token_to_idx["*"] = self.vocab_size - 5
        self.vocab_token_to_idx["("] = self.vocab_size - 6
        self.vocab_token_to_idx[")"] = self.vocab_size - 7
        self.vocab_token_to_idx["x"] = self.vocab_size - 8
        self.vocab_token_to_idx["<ACT>"] = self.vocab_size - 9

        self.baseline_accuracy = 1 / (self.vocab_size - 9)

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 9

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]

    def tokenize_string(self, x_string):
        return [ self.vocab_token_to_idx[val] for val in x_string.split(" ") ]

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        equation = Equation(self.vocab_size - 9, seed=self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = length - 2
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            equation.new_equation()

            sampled_equation = equation.sample_equation_of_length(input_length)
            # ic(sampled_equation)
            # equation_length = len(sampled_equation.split(" "))

            equation_lhs, equation_rhs = sampled_equation.split("=")
            expr = sympy.parse_expr(f"{equation_lhs} - ( {equation_rhs} )")

            # this means the expression is invalid, sample another one
            if isinstance(expr, sympy.core.numbers.Integer):
                continue

            x = sympy.Symbol("x")
            solution = None
            solution = int(sympy.solvers.solve(expr, x)[0]) % self.output_vocab_size

            tokenized_expression = torch.Tensor(self.tokenize_string(f"{sampled_equation} <ACT> {solution}")).long()
            expression_length = tokenized_expression.shape[0]

            seq[:expression_length] = tokenized_expression

            # repetition logic
            seq[expression_length:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[expression_length-1] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Modarithmetic(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 12
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 7)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1
        self.vocab_token_to_idx["="] = self.vocab_size - 2
        self.vocab_token_to_idx["+"] = self.vocab_size - 3
        self.vocab_token_to_idx["-"] = self.vocab_size - 4
        self.vocab_token_to_idx["*"] = self.vocab_size - 5
        self.vocab_token_to_idx["("] = self.vocab_size - 6
        self.vocab_token_to_idx[")"] = self.vocab_size - 7

        self.baseline_accuracy = 1 / (self.vocab_size - 7)

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 7

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]


    def tokenize_string(self, x_string):
        return [ self.vocab_token_to_idx[val] for val in x_string.split(" ") ]

    def generate(self):
        rand_gen = torch.Generator()
        python_rand_gen = random.Random(self.seed)
        rand_gen.manual_seed(self.seed)
        expression = Expression(self.vocab_size - 7, python_rand_gen)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = length
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            expression.new_tree()
            random_expression = expression.sample_expression_of_length(input_length-1)
            # ic(random_expression)
            tokenized_expression = torch.Tensor(self.tokenize_string(random_expression)).long()
            expression_length = tokenized_expression.shape[0]
            expression_result = eval(random_expression)

            seq[:expression_length] = tokenized_expression
            seq[expression_length] = self.vocab_token_to_idx["="]
            seq[expression_length+1] = self.vocab_token_to_idx[str(expression_result % self.output_vocab_size)]

            # repetition logic
            seq[expression_length+2:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[expression_length+1] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Cyclenav(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 9
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = { str(key): key for key in range(self.vocab_size-4) }
        self.vocab_token_to_idx["+1"] = self.vocab_size - 4
        self.vocab_token_to_idx["-1"] = self.vocab_size - 3
        self.vocab_token_to_idx["<STAY>"] = self.vocab_size - 2
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        # self.vocab_token_to_idx = {"+1": 0, "-1": 1, "<STAY>": 2, "<PAD>": 3}
        self.baseline_accuracy = 1 / (self.vocab_size - 4)

        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 4

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}

    def decode(self, tokens):
        raw_decode = " ".join([ self.vocab_idx_to_token[idx] for idx in tokens ])
        raw_decode = raw_decode.replace("<STAY>", "+0")
        return raw_decode

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length-1, (1, ), generator=rand_gen)
            input_length = length - 1
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            seq[:input_length] = torch.randint(self.vocab_size-4, self.vocab_size-1, (input_length, ), generator=rand_gen)

            # compute the result
            result = eval(self.decode(seq[:input_length].data.tolist()))

            seq[input_length] = result % self.output_vocab_size

            # repetition logic
            seq[input_length+1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Modarithmeticwobraces(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 10
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 2)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1
        self.vocab_token_to_idx["="] = self.vocab_size - 2
        self.vocab_token_to_idx["+"] = self.vocab_size - 3
        self.vocab_token_to_idx["-"] = self.vocab_size - 4
        self.vocab_token_to_idx["*"] = self.vocab_size - 5

        self.baseline_accuracy = 1 / (self.vocab_size - 5)

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 5

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]
        self.ACT_TOKEN = self.vocab_token_to_idx.get("<ACT>")

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = int(np.floor(length / 2) - 2)
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length*2:2] = torch.randint(0, self.vocab_size - 5, (input_length,), generator=rand_gen)
            seq[1:input_length * 2:2] = torch.randint(self.vocab_size - 5, self.vocab_size-2, (input_length,), generator=rand_gen)
            seq[input_length*2+1] = self.vocab_token_to_idx["="]

            ## logic for computing result
            running_output = None
            cur_op = None
            for ele_id, ele in enumerate(seq[:input_length*2+1]):
                ele = ele.item()
                if running_output is None:
                    running_output = int(self.vocab_idx_to_token[ele])
                elif ele == self.vocab_token_to_idx["+"]:
                    cur_op = lambda x: running_output + x
                elif ele == self.vocab_token_to_idx["-"]:
                    cur_op = lambda x: running_output - x
                elif ele == self.vocab_token_to_idx["*"]:
                    cur_op = lambda x: running_output * x
                elif ele < self.output_vocab_size:
                    running_output = cur_op(ele)
                    cur_op = None
                else:
                    raise ValueError(f"Illegal operator {ele}")

            assert cur_op is None, "there is a dangling operator"

            seq[input_length*2+2] = running_output % self.output_vocab_size

            # repetition logic
            seq[input_length*2+3:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length*2+2] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Bucketsort(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):

        self.vocab_size = vocab_size if vocab_size is not None else 11
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 2)}
        self.vocab_token_to_idx["<ACT>"] = self.vocab_size - 2
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        self.baseline_accuracy = 1 / (self.vocab_size - 2)

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 2

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]
        self.ACT_TOKEN = self.vocab_token_to_idx.get("<ACT>")

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = int(np.floor(length / 2) - 1)
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length] = torch.randint(0, self.vocab_size - 2, (input_length,), generator=rand_gen)
            seq[input_length] = self.ACT_TOKEN

            seq[input_length+1:2*input_length+1], _ = torch.sort(seq[:input_length])

            # repetition logic
            seq[2 * input_length + 1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length+1:2*input_length+1] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Repetition(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 12
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 2)}
        self.vocab_token_to_idx["<ACT>"] = self.vocab_size - 2
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        self.baseline_accuracy = 1 / (self.vocab_size - 2)

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 2

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]
        self.ACT_TOKEN = self.vocab_token_to_idx.get("<ACT>")
        self.EOS_TOKEN = self.vocab_token_to_idx.get("<EOS>")

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = int(np.floor(length / 2) - 1)
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length] = torch.randint(0, self.vocab_size - 2, (input_length,), generator=rand_gen)
            seq[input_length] = self.ACT_TOKEN

            seq[input_length+1:2*input_length+1] = seq[:input_length]

            # repetition logic
            seq[2 * input_length + 1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length+1:2*input_length+1] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

class Parity(IterableDataset):
    def __init__(self, vocab_size, min_length, max_length, seed=0):
        self.vocab_size = vocab_size if vocab_size is not None else 3
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

        # build vocabulary
        self.vocab_token_to_idx = {str(val): val for val in range(self.vocab_size - 1)}
        self.vocab_token_to_idx["<PAD>"] = self.vocab_size - 1

        self.baseline_accuracy = 0.5

        self.input_vocab_size = self.vocab_size
        self.output_vocab_size = self.vocab_size - 1

        self.vocab_idx_to_token = {val: key for key, val in self.vocab_token_to_idx.items()}
        self.PAD_TOKEN = self.vocab_token_to_idx["<PAD>"]

    def generate(self):
        rand_gen = torch.Generator()
        rand_gen.manual_seed(self.seed)
        while True:
            # ic(self.max_length)
            length = torch.randint(self.min_length, self.max_length+1, (1, ), generator=rand_gen)
            input_length = length - 1
            seq = torch.zeros(self.max_length, dtype=torch.long)
            mask = torch.zeros(self.max_length, dtype=torch.bool)

            # for i in range(num_sequences):
            seq[:input_length] = torch.randint(0, 2, (input_length,), generator=rand_gen)
            seq[input_length] = 1 if torch.sum(seq[:input_length]) % 2 == 0 else 0

            # repetition logic
            seq[input_length + 1:] = self.PAD_TOKEN

            # only the repeated sequence when <ACT> needs to be propagated
            mask[input_length] = True

            yield seq, mask, length

    def __iter__(self):
        return iter(self.generate())

##################### DATA Modules

class FormalLanguageStreaming(L.LightningDataModule):
    def __init__(self,
                 seed=0,
                 min_train_length: int = 6,
                 max_train_length: int = 40,
                 min_val_length: int = 41,
                 max_val_length: int = 256,
                 min_test_length: int = 41,
                 max_test_length: int = 256,
                 task_name: str = "repetition",
                 batch_size: int = 256,
                 num_workers: int = 4,
                 vocab_size: int = None,
                 pin_memory: bool = True):
        super().__init__()
        self.min_train_length = min_train_length
        self.max_train_length = max_train_length
        self.min_val_length = min_val_length
        self.max_val_length = max_val_length
        self.min_test_length = min_test_length
        self.max_test_length = max_test_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.baseline_accuracy = 0.0
        self.vocab_size = vocab_size
        self.seed = seed

        # self.PAD_TOKEN = None
        # self.ACT_TOKEN = None
        # self.EOS_TOKEN = None
        self.vocab_token_to_idx = {}
        self.vocab_idx_to_token = {}

        self.task_name = task_name

        # this should be called to initialize the vocabulary size and number of classes
        self.prepare_data()

        self.save_hyperparameters(ignore=['_class_path'])

    def tokens_to_string(self, tokens):
        out = ""
        for token in tokens:
            out = f"{out} {self.vocab_idx_to_token[token]}"
        return out[1:]  # removes the starting space

    def timeline(self, **kwargs):
        assert len(kwargs) > 0, "need at least one argument"

        table = Table(show_header=True, header_style="bold magenta",
                      row_styles=["dim", ""], title=f"Task: {self.task_name}")

        table.add_column("time", no_wrap=True)
        for t in list(range(len(kwargs[list(kwargs.keys())[0]]))):
            table.add_column(f"{t}", justify="center", width=4)
        # table_list = [ ["time"] + list(range(len(kwargs[list(kwargs.keys())[0]]))) ]

        for arg_key, val in kwargs.items():
            cols = [ f"{arg_key}" ]
            # print(val)
            for token in val:
                if not isinstance(token, bool):
                    token_str = self.vocab_idx_to_token.get(token, token)  # vocabulary convert if available
                    token_str = str(token_str)
                else:
                    token_str = "1" if token else "0"
                # cols.append(token_str[1] if len(token_str) > 1 else token_str )
                token_str = token_str.replace("<", "")
                token_str = token_str.replace(">", "")
                cols.append(token_str)
            table.add_row(*cols)
            # table_list.append(cols)

        console = Console()
        console.print(table)

    def prepare_data(self):

        classname = self.task_name.capitalize()

        try:
            self.task_class = globals()[classname]
        except KeyError:
            raise NotImplementedError(f"Task {classname} not implemented")

        obj = self.task_class(self.vocab_size, self.min_train_length, self.max_train_length)

        self.vocab_size = obj.vocab_size
        self.vocab_token_to_idx = obj.vocab_token_to_idx
        self.vocab_idx_to_token = obj.vocab_idx_to_token
        self.output_vocab_size = obj.output_vocab_size
        self.baseline_accuracy = obj.baseline_accuracy

    def setup(self, stage=None):
        # Load cached data
        self.train_dataset = self.task_class(self.vocab_size, self.min_train_length,
                                             self.max_train_length, seed=self.seed)
        self.val_dataset = self.task_class(self.vocab_size, self.min_val_length,
                                           self.max_val_length, seed=self.seed+1)
        self.test_dataset = self.task_class(self.vocab_size, self.min_test_length,
                                            self.max_test_length, seed=self.seed+2)

        def collate_batch(batch):
            xs, masks, lengths = zip(*batch)
            xs = torch.stack(xs, dim=0)

            ## y needs to be set first for correctness
            ys = xs[:, 1:]
            xs = xs[:, :-1]

            lengths = torch.stack(lengths, dim=0)
            masks = torch.stack(masks, dim=0)[:, 1:]
            return torch.Tensor(xs), torch.Tensor(ys), {"lengths": lengths, "masks": masks}

        self._collate_fn = collate_batch

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

if __name__ == "__main__":
    for taskname in [ "repetition", "bucketsort", "modarithmeticwobraces",
                      "cyclenav", "modarithmetic", "solveequation",
                      "parity", "majoritycount", "majority" ]:
        if taskname == "parity":
            vocab_size = 3
        elif taskname in [ "majoritycount", "majority" ]:
            vocab_size = 64
        else:
            vocab_size = 12
        datamodule = FormalLanguageStreaming(0,
                                       10,
                                       20,
                                       11,
                                       20,
                                       task_name=taskname,
                                       batch_size=1,
                                       num_workers=1,
                                       vocab_size=vocab_size)

        datamodule.setup()
        datamodule.prepare_data()

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        x, y, options = batch

        # ic(x[0].shape)
        # ic(options["lengths"][0])

        ## this will be one less because of the sequential training paradigm
        ic(datamodule.vocab_size)
        ic(datamodule.output_vocab_size)
        ic(options["lengths"][0])
        datamodule.timeline(input=x[0].tolist(), output=y[0].tolist(), mask=options["masks"][0].tolist())
