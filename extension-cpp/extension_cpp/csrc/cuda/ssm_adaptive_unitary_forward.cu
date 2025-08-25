// #pragma once

/*
Roadmap
- [x] forward dynamics
- [x] backward dynamics
- [x] forward with hidden state
- [x] backward with hidden state
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>
#include <cub/cub.cuh>

#include "ssm_adaptive_commons.h"

#define SHARED_MEMORY_CONSTANT 0
#define SHARED_MEMORY_FACTOR 6
// #define NUM_THREADS 128
// #define kNItems 16

namespace extension_cpp{

//////////////////////////////////////////////////////////////////////////////////////////// Complex Logic

template <typename kAttrs>
__global__ void ssm_adaptive_forward_kernel_unitary(SSMaParams params) {

    const int b = blockIdx.z;
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    const int uyz_dim1_stride = params.seq_length;
    const int uyz_dim2_stride = params.seq_length*params.d;

    const int x_dim1_stride = params.d;
    const int x_dim2_stride = params.n*params.d;

    // time index
    const int thread_id = threadIdx.x;

    const int y_local_offset = b*uyz_dim2_stride*params.n + i*uyz_dim1_stride*params.n + j*params.seq_length;
    const int y_max = min(params.n*(params.batch_size*uyz_dim2_stride + params.d*uyz_dim1_stride + params.seq_length) + params.seq_length,
                          params.n*(params.batch_size*uyz_dim2_stride + params.d*uyz_dim1_stride + params.seq_length) + (thread_id+1)*kAttrs::kNItems);

    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename kAttrs::BlockLoadT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename kAttrs::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = reinterpret_cast<typename kAttrs::BlockScanT::TempStorage&>(smem_);
    auto& smem_exchange = reinterpret_cast<typename kAttrs::BlockExchangeT::TempStorage&>(smem_);

    //// Computing A_{l i j} from u
    float u_local[kAttrs::kNItems];
    float dt_local[kAttrs::kNItems];
    float z_local[kAttrs::kNItems];
    float y_local[kAttrs::kNItems];
    float re_local[kAttrs::kNItems];
    float im_local[kAttrs::kNItems];
    float Aij_local[kAttrs::kNItems] = { 0 };
    float* x_local = (float*) (smem_ + kAttrs::smem_size);
    const float local_B_r = params.B_r[j];
    const float local_B_theta = params.B_theta[j];
    const float local_C_r = params.C_r[j];
    const float local_C_theta = params.C_theta[j];
    const float local_D = params.D[i];
    float local_x_bias = params.x_bias[i*params.n + j];
    float local_hid_real = params.prev_hid_real[b*params.d*params.n + i*params.n + j];
    float local_hid_imag = params.prev_hid_imag[b*params.d*params.n + i*params.n + j];

    // load x into shared memory
    for(int p=thread_id; p<params.d; p+=blockDim.x){
//         printf("[%d %d %d - %d] p: %d", b, i, j, thread_id, p);
        x_local[p] = params.x[i*x_dim2_stride + j*x_dim1_stride + p];
    }

//     __syncthreads();

    // A = einsum(x, u, "i j r, b l r -> b l i j") + x_bias
    // This full section computes Aij and stores in local memory.
    // Also after computation, u_local should contain u_i
    for(int p=0; p<i; p++){
        __syncthreads();
        kAttrs::BlockLoadT(smem_load).Load(params.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, params.seq_length);
//         local_x_temp = params.x[i*x_dim2_stride + j*x_dim1_stride + p];
        #pragma unroll
        for(int k=0; k<kAttrs::kNItems; k++){
            Aij_local[k] += u_local[k] * x_local[p];
        }
    }

    for(int p=i+1; p<params.d; p++){
        __syncthreads();
        kAttrs::BlockLoadT(smem_load).Load(params.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, params.seq_length, 0);
//         local_x_temp = params.x[i*x_dim2_stride + j*x_dim1_stride + p];
        #pragma unroll
        for(int k=0; k<kAttrs::kNItems; k++){
            Aij_local[k] += u_local[k] * x_local[p];
        }
    }
    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(params.u + (b*uyz_dim2_stride + i*uyz_dim1_stride), u_local, params.seq_length, 0);
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        Aij_local[k] += u_local[k] * x_local[i] + local_x_bias;
    }
    // end of Aij computation

    // u = F.silu(u)
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        u_local[k] = silu(u_local[k]);
    }

    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(params.dt + (b*uyz_dim2_stride + i*uyz_dim1_stride), dt_local, params.seq_length, 0);

    // G = torch.cumsum(A, dim=1)
    __syncthreads();
    kAttrs::BlockScanT(smem_scan).InclusiveSum(Aij_local, Aij_local);

    float (&Gij_local)[kAttrs::kNItems] = Aij_local;  // Aij_local -> Gij_local -----------------------------------
    // G = G % (2 * torch.pi)
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        Gij_local[k] = fmodf(Gij_local[k], 2*M_PI);
    }

    // re = einsum(B_r * torch.cos(B_theta - G), delta, u, "b l i j, b l i, b l i -> b l i j")
    // im = einsum(B_r * torch.sin(B_theta - G), delta, u, "b l i j, b l i, b l i -> b l i j")
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
//         printf("[%d %d %d - %d] uz[%d]: %f \n", b, i, j, thread_id, (thread_id*kNItems) + k, uz_local[k]);
        re_local[k] = local_B_r * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * u_local[k];
        im_local[k] = local_B_r * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * u_local[k];
    }
    __syncthreads();
    kAttrs::BlockScanT(smem_scan).InclusiveSum(re_local, re_local);
    __syncthreads();
    kAttrs::BlockScanT(smem_scan).InclusiveSum(im_local, im_local);

    // y = (y + u * D) * silu(z)
    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(params.z + (b*uyz_dim2_stride + i*uyz_dim1_stride), z_local, params.seq_length, 0);
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        z_local[k] = silu(z_local[k]);
        if(j == 0){ y_local[k] = u_local[k] * local_D * z_local[k]; } else { y_local[k] = 0; }
    }

    // y = torch.sum(C_r * ( torch.cos(C_theta + G) * re - torch.sin(C_theta + G) * im), dim=-1)
    float temp_hidden_real = 0.0;
    float temp_hidden_imag = 0.0;

    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        y_local[k] += z_local[k] * local_C_r * cosf(local_C_theta + Gij_local[k]) * ( re_local[k] + local_hid_real ) -
                      z_local[k] * local_C_r * sinf(local_C_theta + Gij_local[k]) * ( im_local[k] + local_hid_imag );

//         temp_hidden_real = cosf(Gij_local[k]) * ( re_local[k] + local_hid_real ) -
//                            sinf(Gij_local[k]) * ( im_local[k] + local_hid_imag );
//         temp_hidden_imag = sinf(Gij_local[k]) * ( re_local[k] + local_hid_real ) +
//                            cosf(Gij_local[k]) * ( im_local[k] + local_hid_imag );
//
//         re_local[k] = temp_hidden_real;
//         im_local[k] = temp_hidden_imag;
    }

    int ultimate_id = params.seq_length - (thread_id * kAttrs::kNItems) - 1;
    if(ultimate_id >= 0 && ultimate_id < kAttrs::kNItems){
        params.next_hid_real[b*params.d*params.n + i*params.n + j] =
                                    cosf(Gij_local[ultimate_id]) * ( re_local[ultimate_id] + local_hid_real ) -
                                    sinf(Gij_local[ultimate_id]) * ( im_local[ultimate_id] + local_hid_imag );
        params.next_hid_imag[b*params.d*params.n + i*params.n + j] =
                                    sinf(Gij_local[ultimate_id]) * ( re_local[ultimate_id] + local_hid_real ) +
                                    cosf(Gij_local[ultimate_id]) * ( im_local[ultimate_id] + local_hid_imag );
    }

    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(y_local, y_local);
    for(int k=0; (k < kAttrs::kNItems) && (k*blockDim.x + thread_id < params.seq_length); k++){
        atomicAdd(params.y + b*uyz_dim2_stride + i*uyz_dim1_stride + k*blockDim.x + thread_id, y_local[k]);
    }
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> ssm_adaptive_cuda_forward_unitary(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor x,
    torch::Tensor x_bias,
    torch::Tensor B_r,
    torch::Tensor B_theta,
    torch::Tensor C_r,
    torch::Tensor C_theta,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor prev_hidden_state_real,
    torch::Tensor prev_hidden_state_imag) {
    /*
    :param u (b d l):
    :param dt (b d l):
    :param x (d n d):
    :param B (n):
    :param C (n):
    :param D (d):
    :param z (b d l):

    :return y (b d l): Float32
   */

    auto batch_size = u.size(0);
    auto L = u.size(2);
    auto d = u.size(1);
    auto n = B_r.size(0);

//     auto y = torch::zeros_like(u);
    auto y = torch::zeros({ batch_size, d, L }, u.options());

    auto next_hid_real = torch::zeros_like(prev_hidden_state_real);
    auto next_hid_imag = torch::zeros_like(prev_hidden_state_imag);

    // how blocks are arranged in a grid
    const dim3 GRID_DIM(n, d, batch_size);

    SSMaParams params = {
        u.data_ptr<float>(),  // u
        dt.data_ptr<float>(), // dt
        x.data_ptr<float>(),  // x
        x_bias.data_ptr<float>(), //x_bias
        B_r.data_ptr<float>(),  // B_r
        B_theta.data_ptr<float>(), // B_theta
        C_r.data_ptr<float>(),  // C_r
        C_theta.data_ptr<float>(),  // C_theta
        D.data_ptr<float>(),  // D
        z.data_ptr<float>(),  // z
        prev_hidden_state_real.data_ptr<float>(), // previous_hidden_state_real
        prev_hidden_state_imag.data_ptr<float>(), // previous_hidden_state_imag
        y.data_ptr<float>(),  // y
        next_hid_real.data_ptr<float>(), //next_hid_real
        next_hid_imag.data_ptr<float>(), //next_hid_imag
        batch_size,           // batch_size
        L,                    // seq_length
        d,                    // d
        n,                    // n
    };

    if (params.seq_length <= 128) {
        using kAttrs = SSMKernelAttrs<32, 4>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_forward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(params);
    } else if (params.seq_length <= 256) {
        using kAttrs = SSMKernelAttrs<32, 8>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_forward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(params);
    } else if (params.seq_length <= 512) {
        using kAttrs = SSMKernelAttrs<32, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_forward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(params);
    } else if (params.seq_length <= 1024) {
        using kAttrs = SSMKernelAttrs<64, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_forward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(params);
    } else {
        using kAttrs = SSMKernelAttrs<128, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_forward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(params);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, next_hid_real, next_hid_imag};
}

// Registers _C as an extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
 m.def("ssm_adaptive_unitary(Tensor u,Tensor dt,Tensor x, Tensor x_bias, Tensor B_r, Tensor B_theta, Tensor C_r, Tensor C_theta, Tensor D, Tensor z, Tensor prev_hid_real, Tensor prev_hidden_state_imag) -> (Tensor, Tensor, Tensor)");
 m.def("ssm_adaptive_backward_unitary(Tensor grad_y,Tensor grad_next_hid_real,Tensor grad_next_hid_imag,Tensor u,Tensor dt,Tensor x,Tensor x_bias,Tensor B_r,Tensor B_theta,Tensor C_r,Tensor C_theta,Tensor D,Tensor z, Tensor prev_hid_real, Tensor prev_hid_imag) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

// Register the torch library as CUDA
// somehow, torch has compiled CPU -> CUDA, CUDA-> HIPA and so on
// I have no explanation for this!!!
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("ssm_adaptive_unitary", &ssm_adaptive_cuda_forward_unitary);
}

} // END OF namespace