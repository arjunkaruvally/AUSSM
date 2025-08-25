import itertools
import time

import torch
from icecream import ic
from numpy.ma.core import zeros_like
from rich.progress import track
from rich import print
from torch.cuda import OutOfMemoryError
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
from torch.overrides import is_tensor_like
import unittest
import extension_cpp
import collections
from torch import Tensor, dtype
from typing import Tuple
import numpy as np
import torch.nn.functional as F
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from torch.autograd.gradcheck import get_numerical_jacobian_wrt_specific_input, GradcheckError

from einops import rearrange, einsum

from tqdm import tqdm

from extension_cpp.ops import reference_ssm_adaptive_unitary_optimized_polar


def sample_inputs(device, requires_grad=False, sequence_length=14, b=16, d=16, n=16):
    ## Realistic
    # b = 1
    l = sequence_length
    # d = 16
    # n = 16

    # ## DEBUG
    # b = 1
    # l = sequence_length
    # d = 1
    # n = 1

    torch.manual_seed(10)

    kwargs = {"dtype": torch.float32, "device": device, "requires_grad": requires_grad}
    complex_kwargs = {"dtype": torch.complex64, "device": device, "requires_grad": requires_grad}
    # u = torch.randn(b, d, l, **kwargs)
    u = torch.randint(0, 2, size=(b, d, l), **kwargs)
    dt = torch.randn(b, d, l, **kwargs)
    x = torch.randn(d, n, d, **kwargs) ## no need of division by 100s
    x_bias = torch.randn(d, n,
                    **kwargs)

    B = torch.randn(n, **complex_kwargs)
    C = torch.randn(n, **complex_kwargs)
    D = torch.randn(d, **kwargs)
    z = torch.randn(b, d, l, **kwargs)

    # # DEBUG
    # u = torch.ones_like(u)
    # dt = torch.ones_like(dt) * 0
    # x = torch.ones_like(x)
    # x_bias = torch.ones_like(x_bias) * 0
    # B = torch.ones_like(B)
    # C = torch.ones_like(C)
    # D = torch.ones_like(D) * 0
    # z = torch.ones_like(z)

    return u.contiguous(), dt.contiguous(), x.contiguous(), x_bias.contiguous(), B.contiguous(), C.contiguous(), D.contiguous(), z.contiguous()


def call_custom_ssm(*args):
    return extension_cpp.ops.ssm_adaptive(*args)[0]


class TestSSMa(TestCase):
    def _test_correctness(self, device, unitary=False, preal=False, chunk_length=2048):
        if unitary or preal:
            args = sample_inputs(device)
        else:
            raise NotImplementedError
            # args = sample_inputs_complex(device)
        result = extension_cpp.ops.ssm_adaptive_interface(*args,
                                                          unitary=unitary, preal=preal, chunk_length=chunk_length)
        if unitary:
            expected = reference_ssm_adaptive_unitary_optimized_polar(*args)
        else:
            raise NotImplementedError
        # torch.set_printoptions(threshold=10_000)
        print("from cpp", result)
        print("refL ", expected)
        self.assertEqual(len(result), len(expected))
        # print(result[torch.abs(result - expected) > 0])
        torch.testing.assert_close(result, expected, atol=7e-5, rtol=5e-5)

    def test_correctness_unitary_cuda(self):
        self._test_correctness("cuda", True)

    def test_correctness_unitary_cuda_chunked(self):
        self._test_correctness("cuda", True, chunk_length=5)

    #
    # def test_correctness_preal_cuda(self):
    #     self._test_correctness("cuda", False, True)
    # #
    # def test_correctness_complex_cuda(self):
    #     self._test_correctness("cuda", False)

    def _test_gradients(self, device, unitary=False, preal=False, chunk_length=2048):
        INPUT_NAMES = ["u", "dt", "x", "x_bias", "B", "C", "D", "z"]
        INPUTS_TO_SHOW = [ ]

        if unitary or preal:
            args = sample_inputs(device, requires_grad=True)
        else:
            raise NotImplementedError

        func = lambda *args: extension_cpp.ops.ssm_adaptive_interface(*args,
                                                                      unitary, preal, chunk_length=chunk_length)
        if unitary:
            func_reference = reference_ssm_adaptive_unitary_optimized_polar
        else:
            raise NotImplementedError

        y_random = torch.randn_like(args[0])

        y_ref = func_reference(*args)
        torch.mean((y_ref - y_random)**2).backward()

        reference_gradients = {}
        for arg_id, arg in enumerate(args):
            # arg.grad *= 0
            reference_gradients[arg_id] = arg.grad.clone()
            arg.grad *= 0
            # print(f"{INPUT_NAMES[arg_id]} {arg.grad}")

        y_cuda = func(*args)
        torch.mean((y_cuda - y_random) ** 2).backward()

        n_errors_raised = 0
        for arg_id, arg in enumerate(args):
            try:
                torch.testing.assert_close(arg.grad, reference_gradients[arg_id])
                print(f"!!!!!!!!!! Congratulations, error wrt {INPUT_NAMES[arg_id]} within tolerance of 1e-5")
                # print(arg.grad)
            except AssertionError as e:
                n_errors_raised = n_errors_raised + 1
                print(f"xxxxxxxxxxxxxxxxxxxxx Error found for gradient wrt {INPUT_NAMES[arg_id]}")
                print(e)
                if INPUT_NAMES[arg_id] in INPUTS_TO_SHOW:
                    print(f"reference: \n {reference_gradients[arg_id]} \n computed: \n {arg.grad}")
            print("=============================================================================")

        # assert False
        assert n_errors_raised == 0, f"{n_errors_raised} Errors found in gradient computation. See output for more details."

    def test_gradients_unitary_cuda(self):
        self._test_gradients("cuda", True)

    def test_gradients_unitary_cuda_chunked(self):
        self._test_gradients("cuda", True, chunk_length=5)

    def _test_scaling(self, unitary=False, preal=False):
        sequence_lengths = [ 2048, 1024, 512, 256, 128, 64 ]
        bs = [ 256, 128, 64, 32, 16 ]
        ds = [ 128, 64, 32, 16, 8 ]
        ns = [ 128, 64, 32, 16, 8 ]

        combinations = list(itertools.product(sequence_lengths, bs, ds, ns))

        no_error = True
        cur_seq = 0
        for combination in track(combinations):
            if cur_seq != combination[0]:
                print(combination)
            # ic(combination)
            try:
                args = sample_inputs("cuda", True,
                                     *combination)
                result = extension_cpp.ops.ssm_adaptive_interface(*args, unitary=unitary, preal=preal)
                result.sum().backward()
            except OutOfMemoryError as e:
                print(f"[red] OOM for (l, b, d, n): {combination} [/red]")
                no_error = False
            except RuntimeError as e:
                print(f"[red] RuntimeError for (l, b, d, n): {combination} [/red]")
                no_error = False
                # time.sleep(2)

        assert no_error, "Errors in certain combination of bldn parameters. see output for details"
        pass

    # CAUTION: This is a scaling test, which is often not required to test correctness
    # if required to run, uncomment but the test can take a long time to run
    # def test_scaling_unitary_cuda(self):
    #     self._test_scaling(unitary=True)


    # def test_gradients_preal_cuda(self):
    #     self._test_gradients("cuda", False, True)
    #
    # def test_gradients_complex_cuda(self):
    #     self._test_gradients("cuda", False)

    # def _opcheck(self, device):
    #     args = sample_inputs(device)
    #     # Use opcheck to test that the operator was written correctly.
    #     # opcheck(torch.ops.extension_cpp.ssm_adaptive.default, args)
    #     opcheck(torch.ops.extension_cpp.ssm_adaptive_unitary.default, args)
    #
    # def test_opcheck_cuda(self):
    #     self._opcheck("cuda")


if __name__ == "__main__":
    unittest.main()
