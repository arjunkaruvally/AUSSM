from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from einops import rearrange, einsum
import torch.nn.functional as F

from tqdm import tqdm

import sys

previous_backward_shapes = {}

__all__ = ["ssm_adaptive_unitary", "ssm_adaptive_interface", "reference_ssm_adaptive_unitary",
           "reference_ssm_adaptive_complex", "reference_ssm_adaptive_preal",
           "reference_ssm_adaptive_preal_optimized", "reference_ssm_adaptive_complex_optimized",
           "reference_ssm_adaptive_unitary_optimized"]

def pad_to_multiple_of(tensor: Tensor, multiple_of: int) -> Tensor:
    initial_length = tensor.shape[-1]
    padded_length = int(np.ceil(initial_length / multiple_of))*multiple_of

    return F.pad(tensor, tuple([0, padded_length - initial_length] + [0]*(2*(len(tensor.shape)-1))), value=0)


def ssm_adaptive_interface(
    u: Tensor, dt: Tensor, x: Tensor, x_bias:Tensor, B: Tensor, C: Tensor, D: Tensor, z: Tensor, unitary=True, preal=False, max_parallel_length=2048, **kwargs) -> Tensor:
    """This is the frontend interface for ssm_adaptive kernel which ignores the G matrix return required only for backward"""
    # print("in ssm_adaptive_interface")

    assert not (unitary and preal), "Both unitary and preal cannot be used at the same time"

    # logic for padding the inpus to the correct length
    initial_length = u.shape[-1]
    b, d, l = u.shape
    _, n, _ = x.shape
    u = pad_to_multiple_of(u, 16).contiguous()
    dt = pad_to_multiple_of(dt, 16).contiguous()
    z = pad_to_multiple_of(z, 16).contiguous()

    if unitary:
        # ## END DEBUG CODE
        # # print("choosing the unitary kernel")
        # assert u.dtype == torch.float32, f"Expected u to be float32 got {u.dtype}"
        # assert dt.dtype == torch.float32, f"Expected dt to be float32 got {dt.dtype}"
        # assert x.dtype == torch.float32, f"Expected x to be float32 got {x.dtype}"
        # assert B.dtype == torch.complex64, f"Expected B to be complex64 got {B.dtype}"
        # assert C.dtype == torch.complex64, f"Expected C to be complex64 got {C.dtype}"
        # assert D.dtype == torch.float32, f"Expected D to be float32 got {D.dtype}"
        # assert z.dtype == torch.float32, f"Expected z to be float32 got {z.dtype}"
        # ## DEBUG CODE

        B_r = F.softplus(B.real).contiguous()
        B_theta = B.imag.contiguous()
        C_r = F.softplus(C.real).contiguous()
        C_theta = C.imag.contiguous()

        # ## DEBUG CODE
        # assert u.is_contiguous(), f"Expected u to be contiguous, got {u.is_contiguous()}"
        # assert dt.is_contiguous(), f"Expected dt to be contiguous, got {dt.is_contiguous()}"
        # assert x.is_contiguous(), f"Expected x to be contiguous, got {x.is_contiguous()}"
        # assert x_bias.is_contiguous(), f"Expected x_bias to be contiguous, got {x_bias.is_contiguous()}"
        # assert B_r.is_contiguous(), f"Expected B_r to be contiguous, got {B_r.is_contiguous()}"
        # assert B_theta.is_contiguous(), f"Expected B_theta to be contiguous, got {B_theta.is_contiguous()}"
        # assert C_r.is_contiguous(), f"Expected C_r to be contiguous, got {C_r.is_contiguous()}"
        # assert C_theta.is_contiguous(), f"Expected C_theta to be contiguous, got {C_theta.is_contiguous()}"
        # assert D.is_contiguous(), f"Expected D to be contiguous, got {D.is_contiguous()}"
        # assert z.is_contiguous(), f"Expected z to be contiguous, got {z.is_contiguous()}"
        # ## END DEBUG CODE
        previous_hidden_state = torch.zeros((b, d, n), device=u.device, dtype=torch.complex64)
        b, d, l = u.shape
        out = torch.zeros_like(u)

        for i in range(0, l, max_parallel_length):
            u_local = u[:, :, i: min(i+max_parallel_length, l)].contiguous()
            dt_local = dt[:, :, i: min(i + max_parallel_length, l)].contiguous()
            z_local = z[:, :, i: min(i + max_parallel_length, l)].contiguous()
            ret = ssm_adaptive_unitary(u_local, dt_local, x, x_bias, B_r, B_theta,
                                       C_r, C_theta, D, z_local, previous_hidden_state.real.contiguous(),
                                       previous_hidden_state.imag.contiguous())
            out[:, :, i: min(i + max_parallel_length, l)] = ret[0]
            previous_hidden_state.real = ret[1]
            previous_hidden_state.imag = ret[2]
    else:
        raise NotImplementedError("other kernels not implemented")

    # logic to remove the padded length
    # contiguous is necessary here for the backward pass
    out = out[:, :, :initial_length]

    return out

#############################################################3 Unitary operators

def ssm_adaptive_unitary(
    u: Tensor, dt: Tensor, x: Tensor, x_bias:Tensor, B_r: Tensor, B_theta: Tensor, C_r: Tensor, C_theta: Tensor,
        D: Tensor, z: Tensor, prev_hidden_state_real: Tensor, prev_hidden_state_imag: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """The ssm_adaptive API"""
    outputs = torch.ops.extension_cpp.ssm_adaptive_unitary.default(
        u, dt, x, x_bias, B_r, B_theta, C_r, C_theta, D, z, prev_hidden_state_real, prev_hidden_state_imag
    )
    return outputs

# This is the backward for ssm_adaptive_unitary_forward.
# ssm_adaptive_forward has 2 returns so there are 2 gradients
def backward_unitary(ctx, grad_y, grad_next_hidden_state_real, grad_next_hidden_state_imag):
    # print("in backward")
    # print(*ctx.saved_tensors)
    u, dt, x, x_bias, B_r, B_theta, C_r, C_theta, D, z, prev_hidden_state_real, prev_hidden_state_imag = ctx.saved_tensors

    # ## DEBUG CODE
    # assert u.is_contiguous(), f"Expected u to be contiguous, got {u.is_contiguous()}"
    # assert dt.is_contiguous(), f"Expected dt to be contiguous, got {dt.is_contiguous()}"
    # assert x.is_contiguous(), f"Expected x to be contiguous, got {x.is_contiguous()}"
    # assert x_bias.is_contiguous(), f"Expected x_bias to be contiguous, got {x_bias.is_contiguous()}"
    # assert B_r.is_contiguous(), f"Expected B_r to be contiguous, got {B_r.is_contiguous()}"
    # assert B_theta.is_contiguous(), f"Expected B_theta to be contiguous, got {B_theta.is_contiguous()}"
    # assert C_r.is_contiguous(), f"Expected C_r to be contiguous, got {C_r.is_contiguous()}"
    # assert C_theta.is_contiguous(), f"Expected C_theta to be contiguous, got {C_theta.is_contiguous()}"
    # assert D.is_contiguous(), f"Expected D to be contiguous, got {D.is_contiguous()}"
    # assert z.is_contiguous(), f"Expected z to be contiguous, got {z.is_contiguous()}"
    # assert grad_y.is_contiguous(), f"Expected grad_y to be contiguous, got {grad_y.is_contiguous()}"
    #
    # global previous_backward_shapes
    # ## END DEBUG CODE
    try:
        ( grad_u, grad_dt, grad_x, grad_x_bias, grad_B_r, grad_B_theta, grad_C_r, grad_C_theta, grad_D, grad_z, grad_prev_hidden_state_real, grad_prev_hidden_state_imag ) = \
            torch.ops.extension_cpp.ssm_adaptive_backward_unitary.default(grad_y, grad_next_hidden_state_real, grad_next_hidden_state_imag,
                                                                          u, dt, x, x_bias, B_r, B_theta, C_r, C_theta, D, z, prev_hidden_state_real, prev_hidden_state_imag)
    except RuntimeError as e:
        current_backward_shapes = { "u": u.shape, "dt": dt.shape, "x": x.shape, "x_bias": x_bias.shape, "B_r": B_r.shape,
                                    "B_theta": B_theta.shape, "C_r": C_r.shape, "C_theta": C_theta.shape,
                                    "D": D.shape, "z": z.shape, "grad_y": grad_y.shape }
        # # DEBUG CODE
        print(f"Runtime Error for {current_backward_shapes}")
        # print(f"Previous backward shapes: {previous_backward_shapes}")
        # previous_backward_shapes = current_backward_shapes
        # # END DEBUG CODE

        raise e

    # there are 7 inputs and they all get gradients
    return grad_u, grad_dt, grad_x, grad_x_bias, grad_B_r, grad_B_theta, grad_C_r, grad_C_theta, grad_D, grad_z, grad_prev_hidden_state_real, grad_prev_hidden_state_imag

def setup_context_unitary(ctx, inputs, output):
    # tensors_to_save = list(inputs) + list(output)[1:]
    tensors_to_save = list(inputs)
    ctx.save_for_backward(*tensors_to_save)

torch.library.register_autograd(
    "extension_cpp::ssm_adaptive_unitary", backward_unitary, setup_context=setup_context_unitary)

## TODO: have to do this later
@torch.library.register_fake("extension_cpp::ssm_adaptive_unitary")
def _(u, dt, x, x_bias, B_r, B_theta, C_r, C_theta, D, z):
    b = u.shape[0]
    L = u.shape[2]
    d = u.shape[1]
    n = B.shape[1]
    fake_y = torch.empty_like(u)
    # fake_hidden_states = torch.empty(b, d, n, number_of_chunks+1, 2, u.options())
    return fake_y

############################### Reference implementations


def reference_ssm_adaptive_unitary(u, dt, x, x_bias, B, C, D, z):
    """
    This is the reference function for complex SSMs. B C and D are interpreted as polar angles and
    A = x @ u is interpreted as polar angles.

    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (d, n, d)
    :param B: torch.Tensor(torch.float32) (b, n, l)
    :param C: torch.Tensor(torch.float32) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    print("in reference_ssm_adaptive_unitary")
    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")
    B = rearrange(B, "b n l -> b l n")
    C = rearrange(C, "b n l -> b l n")
    z = rearrange(z, "b d l -> b l d")

    B = torch.polar(torch.ones_like(B), B)
    C = torch.polar(torch.ones_like(C), C)

    delta = dt

    A = einsum(x, u, "i j r, b l r -> b l i j") + x_bias
    A = torch.polar(torch.ones_like(A), A)

    # u = F.silu(u)  # done before calling the function
    # print(A)

    (b, l, d_in) = u.shape
    n = A.shape[-1]

    # deltaA = torch.exp(einsum(delta, A, 'b l d_in, b l d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

    x = torch.polar(torch.zeros((b, d_in, n), device=u.device), torch.zeros((b, d_in, n), device=u.device))
    ys = []
    for i in tqdm(range(l)):
        x = A[:, i] * x + deltaB_u[:, i]  # to remove
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in').real
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    y = y + u * D
    y = y * F.silu(z)

    y = rearrange(y, "b l d -> b d l")

    return y.real


def reference_ssm_adaptive_complex(u, dt, x, B, C, D, z):
    """
    This is the reference function for complex SSMs.
    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (2, d, n, d). one dimension each for rho and theta
    :param B: torch.Tensor(torch.complex64) (b, n, l). why 64? see https://pytorch.org/docs/2.4/generated/torch.complex.html#torch-complex
    :param C: torch.Tensor(torch.complex64) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """
    print("in reference_ssm_adaptive_complex")
    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")
    B = rearrange(B, "b n l -> b l n")
    C = rearrange(C, "b n l -> b l n")
    z = rearrange(z, "b d l -> b l d")

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32
    assert len(x.shape) == 4
    assert B.dtype == torch.complex64
    assert C.dtype == torch.complex64
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    (b, l, d_in) = u.shape

    delta = dt

    A = einsum(x, u, "p i j r, b l r -> p b l i j")

    u = F.silu(u)
    # print(A)

    A = torch.polar(torch.exp(A[0]), A[1])

    n = A.shape[-1]

    # deltaA = torch.exp(einsum(delta, A, 'b l d_in, b l d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

    assert deltaB_u.dtype == torch.complex64

    h = torch.zeros((b, d_in, n), device=u.device, dtype=torch.complex64)
    ys = []
    for i in tqdm(range(l)):
        h = A[:, i] * h + deltaB_u[:, i]  # to remove
        assert h.dtype == torch.complex64
        assert A.dtype == torch.complex64
        assert C[:, i, :].dtype == torch.complex64
        y = einsum(h, C[:, i, :], 'b d_in n, b n -> b d_in').real
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    y = y + u * D
    y = y * F.silu(z)

    print("uz", u * F.silu(z))

    y = rearrange(y, "b l d -> b d l")

    return y


def reference_ssm_adaptive_preal(u, dt, x, B, C, D, z):
    """
    This is the reference function for complex SSMs.
    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (d, n, d).
    :param B: torch.Tensor(torch.float32) (b, n, l).
    :param C: torch.Tensor(torch.float32) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """
    print("in reference_ssm_adaptive_preal")
    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")
    B = rearrange(B, "b n l -> b l n")
    C = rearrange(C, "b n l -> b l n")
    z = rearrange(z, "b d l -> b l d")

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32
    assert len(x.shape) == 3
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    (b, l, d_in) = u.shape

    delta = dt

    A = torch.exp(einsum(x, u, "i j r, b l r -> b l i j"))

    u = F.silu(u)
    # print(A)

    n = A.shape[-1]

    # deltaA = torch.exp(einsum(delta, A, 'b l d_in, b l d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

    assert deltaB_u.dtype == torch.float32

    h = torch.zeros((b, d_in, n), device=u.device, dtype=torch.float32)
    ys = []
    for i in tqdm(range(l)):
        h = A[:, i] * h + deltaB_u[:, i]  # to remove
        y = einsum(h, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    y = y + u * D
    y = y * F.silu(z)

    y = rearrange(y, "b l d -> b d l")

    return y


#################################  Optimized Versions

def reference_ssm_adaptive_complex_optimized(u, dt, x, x_bias, B, C, D, z=None,
                                             diagnostic=False, z_prod=True, binary=False, no_graded_rotation=False):
    """
    This is the reference function for complex SSMs.
    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.complex64) (d, n, d). one dimension each for rho and theta
    :param B: torch.Tensor(torch.complex64) (b, n, l). why 64? see https://pytorch.org/docs/2.4/generated/torch.complex.html#torch-complex
    :param C: torch.Tensor(torch.complex64) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """
    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.complex64
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    u = u.contiguous()
    dt = dt.contiguous()
    x = x.contiguous()
    x_bias = x_bias.contiguous()

    constantB = False
    constantC = False
    if len(B.shape) == 1:
        constantB = True

    if len(C.shape) == 1:
        constantC = True

    if diagnostic:
        diagnostic_dict = {}

    if constantB:
        assert B.dtype == torch.complex64
    else:
        assert B.dtype == torch.float32

    if constantC:
        assert C.dtype == torch.complex64
    else:
        assert C.dtype == torch.float32

    if z_prod:
        assert z.dtype == torch.float32
        z = rearrange(z, "b d l -> b l d")

    u = rearrange(u, "b d l -> b l d")
    # u = torch.complex(u, torch.zeros_like(u))
    dt = rearrange(dt, "b d l -> b l d")

    if not constantB:
        B = rearrange(B, "b n l -> b l n")

    if not constantC:
        C = rearrange(C, "b n l -> b l n")

    delta = torch.complex(dt, torch.zeros_like(dt)).contiguous()

    # deltaA = torch.exp(einsum(delta, A, 'b l d_in, b l d_in n -> b l d_in n'))
    # deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

    A = einsum(x, torch.complex(u, torch.zeros_like(u)).contiguous(), "i j r, b l r -> b l i j").contiguous()
    b, l, d, n = A.shape

    if diagnostic:
        diagnostic_dict["A_adaptive"] = A.detach().clone()

    u = F.silu(u)
    A = A + x_bias.reshape((1, 1, d, n)).contiguous()

    A_r = F.sigmoid(A.real)
    A_theta = A.imag

    if diagnostic:
        diagnostic_dict["A"] = A.detach().clone()

    G_r = torch.cumprod(A_r, dim=1)
    G_theta = torch.cumsum(A_theta, dim=1) % (2*torch.pi)
    # print(G_rho)
    # G = torch.complex(torch.exp(G_rho), G_theta).contiguous()
    # G_inv = torch.complex(torch.exp(-G_rho), -G_theta).contiguous()

    if constantB:
        f_var = einsum(G_r * torch.complex(torch.ones_like(G_theta), G_theta), delta, B, u,
                       "b l i j, b l i, j, b l i -> b l i j")
    else:
        raise NotImplementedError("Current setting not supported in the SSM.")

    if constantC:
        y = einsum(C, 1/(G_r + 1e-10) * torch.complex(torch.ones_like(G_theta), -G_theta), f_var, "j, b l i j, b l i j -> b l i").real
    else:
        raise NotImplementedError("Current setting not supported in the SSM.")

    y = y + u * D
    y = y * F.silu(z)
    y = rearrange(y, "b l d -> b d l")

    return y


def reference_ssm_adaptive_preal_optimized(u, dt, x, x_bias, B, C, D, z):
    """
    This is the reference function for complex SSMs.
    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (d, n, d).
    :param B: torch.Tensor(torch.float32) (b, n, l).
    :param C: torch.Tensor(torch.float32) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """
    # print("in reference_ssm_adaptive_preal")
    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")
    B = rearrange(B, "b n l -> b l n")
    C = rearrange(C, "b n l -> b l n")
    z = rearrange(z, "b d l -> b l d")

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32
    assert len(x.shape) == 3
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    (b, l, d_in) = u.shape
    n = B.shape[-1]
    delta = dt

    log_A = einsum(x, u, "i j r, b l r -> b l i j")
    log_A += x_bias.reshape((1, 1, d_in, n))
    log_A = einsum(delta, log_A, "b l i, b l i j -> b l i j")
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
    G = torch.cumsum(log_A, dim=1)
    re = torch.exp(- G) * deltaB_u
    re = torch.cumsum(re, dim=1)
    y = einsum(C, torch.exp(G), re, "b l j, b l i j, b l i j -> b l i")

    y = y + u * D
    y = y * F.silu(z)
    y = rearrange(y, "b l d -> b d l")

    return y


@torch.compile
def reference_ssm_adaptive_unitary_optimized(u, dt, x, x_bias, B, C, D, z):
    """
    This is the reference function for complex SSMs. B C and D are interpreted as polar angles and
    A = x @ u is interpreted as polar angles.

    :param u: torch.Tensor(torch.float32) (b, d, l)  (CAUTION: Do not do silu before calling this function)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (d, n, d)
    :param B: torch.Tensor(torch.float32) (b, n, l)
    :param C: torch.Tensor(torch.float32) (b, n, l)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32

    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    constantB = False
    constantC = False
    if len(B.shape) == 1:
        constantB = True
    if len(C.shape) == 1:
        constantC = True

    if constantB:
        assert B.dtype == torch.complex64
    else:
        assert B.dtype == torch.float32

    if constantC:
        assert C.dtype == torch.complex64
    else:
        assert C.dtype == torch.float32

    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")
    z = rearrange(z, "b d l -> b l d")

    if not constantB:
        B = rearrange(B, "b n l -> b l n")
    if not constantC:
        C = rearrange(C, "b n l -> b l n")

    delta = dt

    # stability can be improved by considering r and theta separately
    A = einsum(x, u, "i j r, b l r -> b l i j")

    u = F.silu(u)

    b, l, d, n = A.shape
    A_theta = A + x_bias.reshape((1, 1, d, n))

    A_theta = F.tanh(A_theta)
    # print(A_theta)
    A_theta = A_theta * torch.pi
    A_r = torch.cos(A_theta)
    A_im = torch.sin(A_theta)
    # A_r = einsum(delta, A_r, "b l i, b l i j -> b l i j")
    # A_im = einsum(delta, A_im, "b l i, b l i j -> b l i j")

    G_r = torch.cumsum(A_r, dim=1)
    G_im = torch.cumsum(A_im, dim=1)

    G = torch.complex(G_r, G_im)

    print(G)

    if G.isnan().any():
        sys.exit(1)

    # ################ Test
    # G = F.sigmoid(G) * 2 * torch.pi
    # ############### END TEST

    # print((torch.cos(B.unsqueeze(-2) - G)).shape)
    if constantB:
        g = einsum(B, torch.exp(-G), delta, u, "j, b l i j, b l i, b l i -> b l i j")
    else:
        raise NotImplementedError

    g = torch.cumsum(g, dim=1)

    if constantC:
        y = einsum(C, torch.exp(G), g, "j, b l i j, b l i j -> b l i").real
    else:
        raise NotImplementedError
    # print(y.shape, u.shape, x.shape)
    y = y + u * D
    y = y * F.silu(z)
    y = rearrange(y, "b l d -> b d l")

    return y


def reference_ssm_adaptive_unitary_optimized_polar(u, dt, x, x_bias, B, C, D, z=None, diagnostic=False, z_prod=True,
                                                   binary=False, no_graded_rotation=False, unitary=True):
    """
    This is the reference function for complex SSMs. B C and D are interpreted as polar angles and
    A = x @ u is interpreted as polar angles.

    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, d, l)
    :param x: torch.Tensor(torch.float32) (d, n, d)
    :param B: torch.Tensor(torch.complex64) (n)
    :param C: torch.Tensor(torch.complex64) (n)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)

    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """

    # assert u.dtype == torch.float32
    # assert dt.dtype == torch.float32
    # assert x.dtype == torch.float32
    #
    # assert D.dtype == torch.float32

    constantB = False
    constantC = False
    if len(B.shape) == 1:
        constantB = True
    if len(C.shape) == 1:
        constantC = True

    if diagnostic:
        diagnostic_dict = {}

    # if constantB:
    #     assert B.dtype == torch.complex64
    # else:
    #     assert B.dtype == torch.float32
    #
    # if constantC:
    #     assert C.dtype == torch.complex64
    # else:
    #     assert C.dtype == torch.float32

    if z_prod:
        # assert z.dtype == torch.float32
        z = rearrange(z, "b d l -> b l d")

    u = rearrange(u, "b d l -> b l d")
    dt = rearrange(dt, "b d l -> b l d")

    if not constantB:
        B = rearrange(B, "b n l -> b l n")
    if not constantC:
        C = rearrange(C, "b n l -> b l n")

    delta = dt

    # stability can be improved by considering r and theta separately
    A = einsum(x, u, "i j r, b l r -> b l i j")

    u = F.silu(u)

    b, l, d, n = A.shape
    if diagnostic:
        diagnostic_dict["A_adaptive"] = A.detach().clone()

    A = A + x_bias.reshape((1, 1, d, n))

    # print("A", A[:, :, :, :])

    # print("A[0]", A[:, 0, :, :])
    # print("A[3]", A[:, 3, :, :])

    if diagnostic:
        diagnostic_dict["A"] = A.detach().clone()

    G = torch.cumsum(A, dim=1)

    if diagnostic:
        diagnostic_dict["G"] = G.detach().clone()

    # ################ Test
    G = G % (2 * torch.pi)   # alternative to having the sigmoid squashing
    # ############### END TEST

    # print("G", G)

    # print("G[0]", G[:, 0, :, :])
    # print("G[3]", G[:, 3, :, :])

    if constantB:
        B_theta = B.imag.reshape((1, 1, 1, -1))
        B_r = F.softplus(B.real.reshape((1, 1, 1, -1)))

        if binary:
            if no_graded_rotation:
                re = einsum(B_r * torch.sign(torch.cos(B_theta - G)), delta, u,
                            "b l i j, b l i, b l i -> b l i j")
                im = einsum(B_r * torch.sign(torch.sin(B_theta - G)), delta, u,
                            "b l i j, b l i, b l i -> b l i j")
            else:
                re = einsum(B_r * torch.sign(torch.cos(G)) * torch.cos(B_theta), delta, u, "b l i j, b l i, b l i -> b l i j")
                im = einsum(B_r * torch.sign(torch.cos(G)) * torch.sin(B_theta), delta, u, "b l i j, b l i, b l i -> b l i j")
        else:
            # print("delta", B_r)
            re = einsum(B_r * torch.cos(B_theta - G), delta, u, "b l i j, b l i, b l i -> b l i j")
            im = einsum(B_r * torch.sin(B_theta - G), delta, u, "b l i j, b l i, b l i -> b l i j")
    else:
        re = einsum(torch.cos(B.unsqueeze(-2) - G), delta, u, "b l i j, b l i, b l i -> b l i j")
        im = einsum(torch.sin(B.unsqueeze(-2) - G), delta, u, "b l i j, b l i, b l i -> b l i j")

    # print("re", re, "im", im)
    # print("re[0]", (re*F.silu(z).unsqueeze(3))[:, 0, :, :])
    # print("re[3]", (re*F.silu(z).unsqueeze(3))[:, 3, :, :])

    re = torch.cumsum(re, dim=1)
    im = torch.cumsum(im, dim=1)

    if diagnostic:
        diagnostic_dict["re"] = re.detach().clone()
        diagnostic_dict["im"] = im.detach().clone()

    if constantC:
        C_theta = C.imag.reshape((1, 1, 1, -1))
        C_r = F.softplus(C.real.reshape((1, 1, 1, -1)))

        if binary:
            y = torch.sum(C_r * (torch.sign(torch.cos(G)) * torch.cos(C_theta) * re
                               - torch.sign(torch.cos(G)) * torch.sin(C_theta) * im), dim=-1)
        else:
            y = torch.sum(C_r * ( torch.cos(C_theta + G) * re - torch.sin(C_theta + G) * im), dim=-1)
    else:
        y = torch.sum((torch.cos(C.unsqueeze(-2) + G) * re - torch.sin(C.unsqueeze(-2) + G) * im), dim=-1)

    if diagnostic:
        if binary:
            diagnostic_dict["h"] = torch.complex(torch.sign(torch.cos(G)) * re,
                                                 torch.sign(torch.cos(G)) * im)
        else:
            diagnostic_dict["h"] = torch.complex(torch.cos(G) * re - torch.sin(G) * im,
                                                 torch.cos(G) * im + torch.sin(G) * re)

    y = y + u * D

    if z_prod:
        # print(y.shape, z.shape)
        y = y * F.silu(z)

    # print("uz", u*F.silu(z))

    y = rearrange(y, "b l d -> b d l")

    if diagnostic:
        return y, diagnostic_dict

    return y



# def reference_mamba(u, dt, A, B, C, D, z=None, delta_bias=None,
#                     diagnostic=False, z_prod=True, binary=False, no_graded_rotation=False,
#                     delta_softplus=True,
#                     return_last_state=False):
#     """
#     This is the reference function for complex SSMs. B C and D are interpreted as polar angles and
#     A = x @ u is interpreted as polar angles.
#
#     :param u: torch.Tensor(torch.float32) (b, d, l)
#     :param dt: torch.Tensor(torch.float32) (b, d, l)
#     :param x: torch.Tensor(torch.float32) (d, n, d)
#     :param B: torch.Tensor(torch.float32) (b, n, l)
#     :param C: torch.Tensor(torch.float32) (b, n, l)
#     :param D: torch.Tensor(torch.float32) (d)
#     :param z: torch.Tensor(torch.float32) (b, d, l)
#
#     :return:  y: torch.Tensor(torch.float32) (b, d, l)
#     """
#
#     assert u.dtype == torch.float32
#     assert dt.dtype == torch.float32
#     assert x.dtype == torch.float32
#
#     assert D.dtype == torch.float32
#
#     if diagnostic:
#         diagnostic_dict = {}
#
#     if z_prod:
#         assert z.dtype == torch.float32
#         z = rearrange(z, "b d l -> b l d")
#
#     u = rearrange(u, "b d l -> b l d")
#     dt = rearrange(dt, "b d l -> b l d")
#
#     if not constantB:
#         B = rearrange(B, "b n l -> b l n")
#     if not constantC:
#         C = rearrange(C, "b n l -> b l n")
#
#     delta = dt
#
#     # stability can be improved by considering r and theta separately
#     A = einsum(x, u, "i j r, b l r -> b l i j")
#
#     u = F.silu(u)
#
#     # b, l, d, n = A.shape
#     # if diagnostic:
#     #     diagnostic_dict["A_adaptive"] = A.detach().clone()
#     #
#     # A = A + x_bias.reshape((1, 1, d, n))
#     #
#     # if diagnostic:
#     #     diagnostic_dict["A"] = A.detach().clone()
#     #
#     # G = torch.cumsum(A, dim=1)
#     #
#     # if diagnostic:
#     #     diagnostic_dict["G"] = G.detach().clone()
#     #
#     # # ################ Test
#     # G = G % (2 * torch.pi)   # alternative to having the sigmoid squashing
#     # # ############### END TEST
#     #
#     # y = y + u * D
#     #
#     # if z_prod:
#     #     # print(y.shape, z.shape)
#     #     y = y * F.silu(z)
#     #
#     # y = rearrange(y, "b l d -> b d l")
#     #
#     # if diagnostic:
#     #     return y, diagnostic_dict
#
#     return y

############################# Alternate models with BC fixed

def reference_ssm_adaptive_unitary_BC_fixed(u, dt, x, x_bias, B, C, D, z):
    """
    This is the reference function for complex SSMs. B C and D are interpreted as polar angles and
    A = x @ u is interpreted as polar angles.

    :param u: torch.Tensor(torch.float32) (b, d, l)
    :param dt: torch.Tensor(torch.float32) (b, l)  this should be softplus'd before calling this function
    :param x: torch.Tensor(torch.float32) (d, n, d)
    :param B: torch.Tensor(torch.float32) (n d)
    :param C: torch.Tensor(torch.float32) (d n)
    :param D: torch.Tensor(torch.float32) (d)
    :param z: torch.Tensor(torch.float32) (b, d, l)
    :return:  y: torch.Tensor(torch.float32) (b, d, l)
    """

    assert u.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert x.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert D.dtype == torch.float32
    assert z.dtype == torch.float32

    u = rearrange(u, "b d l -> b l d")
    z = rearrange(z, "b d l -> b l d")

    delta = dt

    # stability can be improved by considering r and theta separately
    A = einsum(x, u, "i j r, b l r -> b l i j")
    b, l, d, n = A.shape
    A = A + x_bias.reshape((1, 1, d, n))
    # u = F.silu(u)  ## This is done outside the kernel
    G = torch.cumsum(A, dim=1)

    re = einsum(torch.cos(B.unsqueeze(-2) - G), delta, u, "b l i j, b l i, b l i -> b l i j")
    im = einsum(torch.sin(B.unsqueeze(-2) - G), delta, u, "b l i j, b l i, b l i -> b l i j")
    re = torch.cumsum(re, dim=1)
    im = torch.cumsum(im, dim=1)
    # print(torch.cos(C.unsqueeze(-2) + G).shape, re.shape, u.shape, x.shape)
    y = torch.sum(torch.cos(C.unsqueeze(-2) + G) * re - torch.sin(C.unsqueeze(-2) + G) * im, dim=-1)
    # print(y.shape, u.shape, x.shape)
    y = y + u * D
    y = y * F.silu(z)
    y = rearrange(y, "b l d -> b d l")

    return y

