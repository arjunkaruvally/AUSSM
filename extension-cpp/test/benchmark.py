from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

from einops import rearrange, einsum
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

import extension_cpp
from tqdm import tqdm

from argparse import Namespace

import matplotlib.pyplot as plt


def run_evaluation(options):
    TIME_SCALES = {"s": 0.000001, "ms": 0.001, "us": 1}

    if options.example == "py":
        ssm_adaptive = extension_cpp.ops.reference_ssm_adaptive_unitary_optimized_polar
    else:
        if options.not_adaptive:
            ssm_adaptive = selective_scan_fn
        else:
            from extension_cpp.ops import ssm_adaptive_interface as ssm_adaptive
    if options.example == "cuda":
        options.cuda = True

    device = torch.device("cuda") if options.cuda else torch.device("cpu")
    dtype = torch.float32

    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}
    complex_kwargs = {"dtype": torch.complex64, "device": device, "requires_grad": True}
    b = options.batch_size
    l = options.sequence_length
    d = options.d_inner
    n = options.d_state

    forward_memory = 0
    backward_memory = 0

    if options.not_adaptive:
        u = torch.randn(b, d, l, **kwargs)
        dt = torch.randn(b, d, l, **kwargs)
        if options.not_adaptive:
            A = torch.randn(d, n, **kwargs)
        else:
            x = torch.randn(d, n, d, **kwargs)
        B = torch.randn(b, n, l, **kwargs)
        C = torch.randn(b, n, l, **kwargs)
        D = torch.randn(d, **kwargs)
        z = torch.randn(b, d, l, **kwargs)
    else:
        u = torch.randn(b, d, l, **kwargs)
        dt = torch.randn(b, d, l, **kwargs)
        x = torch.randn(d, n, d,
                        **kwargs) / 10000  ## /100 factor removes some of the overflow issues when computing gradients
        x_bias = torch.randn(d, n,
                             **kwargs) / 10000

        B = torch.randn(n, **complex_kwargs)
        C = torch.randn(n, **complex_kwargs)
        D = torch.randn(d, **kwargs)
        z = torch.randn(b, d, l, **kwargs)

    # print(run_id)
    u.grad = None
    dt.grad = None
    if options.not_adaptive:
        A.grad = None
    else:
        x.grad = None
        x_bias.grad = None
    B.grad = None
    C.grad = None
    D.grad = None
    z.grad = None

    with torch.autograd.profiler.profile(use_device="cuda", profile_memory=True) as prof:
        # first 5 runs are warmup
        for run_id in tqdm(range(options.runs+5), desc="trials", leave=False):
            torch.cuda.reset_peak_memory_stats()
            if options.not_adaptive:
                y = ssm_adaptive(u.float(), dt.float(), A.float(), B.float(), C.float(), D.float(), z.float())
            else:
                y = ssm_adaptive(u.contiguous(), dt.contiguous(), x.contiguous(), x_bias.contiguous(), B.contiguous(),
                                                             C.contiguous(), D.contiguous(), z.contiguous(), unitary=True)

            current_forward_memory = torch.cuda.max_memory_allocated() * 1e-6

            if run_id > 5:  # warmup time
                forward_memory += current_forward_memory

            if not options.no_backward:
                y.sum().backward(retain_graph=True)
            current_backward_memory = torch.cuda.max_memory_allocated() * 1e-6

            if options.cleanup:
                # cleanup the memory
                del y
            if options.cleanup:
                torch.cuda.empty_cache()

            if run_id > 5:
                backward_memory += current_backward_memory
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1, max_src_column_width=5, max_name_column_width=30))

    forward_average = 0
    backward_average = 0
    for item in prof.key_averages():  # computes both the time spent in cuda kernel and any aten functions
        # print(item.key, " ", item.cuda_time)
        if "extension_cpp::" in item.key or "SelectiveScanFn" in item.key:
            if "ackward" in item.key:
                backward_average += item.device_time
            else:
                forward_average += item.device_time

    if forward_average <= 0:
        print("WARNING: no known kernels detected, falling back to counting all the aten events")
        for item in prof.key_averages():  # computes both the time spent in cuda kernel and any aten functions
            if "aten::" in item.key:
                if "ackward" in item.key:
                    backward_average += item.device_time
                else:
                    forward_average += item.device_time
            elif "ackward" in item.key:
                backward_average += item.device_time

    # if forward_average == 0:
    scale = TIME_SCALES[options.scale]

    return (forward_average * scale,
            backward_average * scale,
            forward_memory / options.runs,
            backward_memory / options.runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## DEBUG
    parser.add_argument("example", choices=["py", "cuda"])
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-l", "--sequence_length", type=int, default=50)
    parser.add_argument("-d", "--d_inner", type=int, default=32)
    parser.add_argument("-n", "--d_state", type=int, default=64)
    parser.add_argument("-r", "--runs", type=int, default=50)
    parser.add_argument("--scale", choices=["s", "ms", "us"], default="us")
    parser.add_argument("-c", "--cuda", action="store_true")
    # parser.add_argument("-i", "--unitary", action="store_true")  ## deprecated, verify the code to use non unitary
    parser.add_argument("-na", "--not_adaptive", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-nb", "--no_backward", action="store_true")
    parser.add_argument("-rm", "--record_memory", action="store_true")

    options = parser.parse_args()

    if options.record_memory:
        torch.cuda.memory._record_memory_history()
    forward_average, backward_average, forward_memory, backward_memory = run_evaluation(options)

    if options.record_memory:
        torch.cuda.memory._dump_snapshot("memory_history.pickle")

    print(
        "Forward: {0:.3f} {2} {3:.2f}MB | Backward {1:.3f} {2} {4:.2f}MB".format(
            forward_average, backward_average, options.scale, forward_memory, backward_memory
        )
    )

    # print("Running evaluation across sequence lengths")
    # options_dict = vars(options)
    #
    # forward_list = []
    # backward_list = []
    # seq_lengths = list(range(50, options.sequence_length, 50))
    #
    # if options.record_memory:
    #     torch.cuda.memory._record_memory_history()
    #
    # for seq_length in tqdm(seq_lengths, desc="Sequence lengths"):
    #     options_dict["sequence_length"] = seq_length
    #
    #     forward_average, backward_average, _, _ = run_evaluation(Namespace(**options_dict))
    #
    #     forward_list.append(forward_average)
    #     backward_list.append(backward_average)
    #
    # if options.record_memory:
    #     torch.cuda.memory._dump_snapshot("memory_history.pickle")
    #
    # plt.title("Forward")
    # plt.plot(seq_lengths, forward_list)
    # plt.savefig("forward.png")
    #
    # plt.clf()
    # plt.title("Backward")
    # plt.plot(seq_lengths, backward_list)
    # plt.savefig("backward.png")
