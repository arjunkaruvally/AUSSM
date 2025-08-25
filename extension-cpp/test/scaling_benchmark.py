from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import numpy as np
import torch

from einops import rearrange, einsum
from icecream import ic
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

import extension_cpp

from rich import print
from rich.progress import Progress, track, TaskID

from argparse import Namespace

import matplotlib.pyplot as plt

import scienceplots

plt.style.use('science')

def run_evaluation(options, progressbar: Progress=None, progress_task_id: TaskID=None):
    if progress_task_id is None:
        print("Progress will not be displayed")

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

    if progress_task_id is not None:
        progressbar.update(task_id=progress_task_id, total=options.runs+5)
        progressbar.update(task_id=progress_task_id, completed=0)

    with torch.autograd.profiler.profile(use_device="cuda", profile_memory=True) as prof:
        # first 5 runs are warmup
        for run_id in range(options.runs+5):
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

            if progress_task_id is not None:
                progressbar.update(task_id=progress_task_id, advance=1)
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1, max_src_column_width=5, max_name_column_width=30))

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



def get_scaling(options, progressbar: Progress=None, progress_task_id: TaskID=None, progress_sub_task_id: TaskID=None):
    if progressbar is None:
        print("Progress will not be displayed")

    options = vars(options)
    forward_averages = []
    backward_averages = []
    forward_memories = []
    backward_memories = []
    sequence_lengths = np.logspace(np.log10(10), np.log10(2048), 20)
    sequence_lengths = [ int(val) for val in sequence_lengths ]

    if progress_task_id is not None:
        progressbar.update(task_id=progress_task_id, total=len(sequence_lengths))
        progressbar.update(task_id=progress_task_id, completed=0)
    for sequence_length in sequence_lengths:
        options["sequence_length"] = sequence_length
        forward_average, backward_average, forward_memory, backward_memory = run_evaluation(Namespace(**options), progressbar, progress_sub_task_id)
        forward_averages.append(forward_average)
        backward_averages.append(backward_average)
        forward_memories.append(forward_memory)
        backward_memories.append(backward_memory)

        if progress_task_id is not None:
            progressbar.update(task_id=progress_task_id, advance=1)

    return sequence_lengths, forward_averages, backward_averages, forward_memories, forward_memories


def compare_resource_utilization(options):
    models = [ "mamba", "SSMau-cuda", "SSMau-pytorch"]
    # models = ["mamba"]
    results = []

    with Progress() as progress:
        task1 = progress.add_task("[red]Model...", total=len(models))
        task2 = progress.add_task("[green]Sequence Length...")
        task3 = progress.add_task("[cyan]Running...")
        for model in models:
            progress.update(task1, description=f"[red]{model}")
            # print("Running " + model)
            options = vars(options)
            if model == "mamba":
                options["cuda"] = True
                options["not_adaptive"] = True
                options["example"] = "cuda"
            elif model == "SSMau-pytorch":
                options["cuda"] = True
                options["not_adaptive"] = False
                options["example"] = "py"
            elif model == "SSMau-cuda":
                options["cuda"] = True
                options["not_adaptive"] = False
                options["example"] = "cuda"
            else:
                raise NotImplementedError

            options = Namespace(**options)
            sequence_lengths, forward_averages, backward_averages, forward_memories, backward_memories = get_scaling(options,
                                                                                                                     progressbar=progress,
                                                                                                                     progress_task_id=task2,
                                                                                                                     progress_sub_task_id=task3)

            result_dict = {
                "sequence length": sequence_lengths,
                "Average Time for the Forward Pass": forward_averages,
                "Average Time for the Backward Pass": backward_averages,
                "Peak GPU Memory Usage for the Forward Pass": forward_memories,
                "Peak GPU Memory Usage for the Backward Pass": backward_memories
            }

            results.append((model, result_dict))
            progress.update(task1, advance=1)

    for key in track(results[0][1].keys(), description="Plotting Results..."):
        plt.clf()
        # plt.figure(figsize=(3, 2))
        if "sequence length" in key:
            continue
        for model, result_dict in results:
            plt.plot(result_dict["sequence length"], result_dict[key], label=model)

        plt.xlabel("sequence length")
        plt.ylabel(f"time (in {options.scale})" if "Time" in key else "GPU Memory (in MB)")
        plt.title(key)

        plt.tight_layout()
        plt.legend()
        # plt.show()
        plt.savefig(f"{key}.svg", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## DEBUG
    # parser.add_argument("example", choices=["py", "cuda"])
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-d", "--d_inner", type=int, default=32)
    parser.add_argument("-n", "--d_state", type=int, default=64)
    parser.add_argument("-r", "--runs", type=int, default=50)
    parser.add_argument("--scale", choices=["s", "ms", "us"], default="us")
    # parser.add_argument("-c", "--cuda", action="store_true")
    # parser.add_argument("-na", "--not_adaptive", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-nb", "--no_backward", action="store_true")

    options = parser.parse_args()

    compare_resource_utilization(options)
