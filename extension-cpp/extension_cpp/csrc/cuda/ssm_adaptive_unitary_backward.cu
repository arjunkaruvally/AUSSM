#pragma once

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>
#include <cub/cub.cuh>

#include "ssm_adaptive_commons.h"


#define SHARED_MEMORY_CONSTANT 0
#define SHARED_MEMORY_FACTOR 6


namespace extension_cpp{


template <typename kAttrs>
__global__ void ssm_adaptive_backward_kernel_unitary(SSMaParams fparams, SSMaBackwardParams bparams) {
    //batch index
    const int b = blockIdx.z;
    const int j = blockIdx.x;
    const int i = blockIdx.y;

    // time index
    const int thread_id = threadIdx.x;

    const int uyz_dim1_stride = fparams.seq_length;
    const int uyz_dim2_stride = fparams.seq_length*fparams.d;

    const int x_dim1_stride = fparams.d;
    const int x_dim2_stride = fparams.n*fparams.d;

    const int y_max = min(fparams.n*(fparams.batch_size*uyz_dim2_stride + fparams.d*uyz_dim1_stride + fparams.seq_length) + fparams.seq_length,
                          fparams.n*(fparams.batch_size*uyz_dim2_stride + fparams.d*uyz_dim1_stride + fparams.seq_length) + (thread_id+1)*kAttrs::kNItems);

    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename kAttrs::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_striped = reinterpret_cast<typename kAttrs::BlockLoadStripedT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename kAttrs::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = reinterpret_cast<typename kAttrs::BlockScanT::TempStorage&>(smem_);
    auto& smem_reduce = reinterpret_cast<typename kAttrs::BlockReduceT::TempStorage&>(smem_);
    auto& smem_revscan = reinterpret_cast<typename kAttrs::BlockReverseScanT::TempStorage&>(smem_);
    auto& smem_exchange = reinterpret_cast<typename kAttrs::BlockExchangeT::TempStorage&>(smem_);

    ///////////////////// Recomputing the Forward Pass

    //// Computing A_{l i j} from u
    float u_local[kAttrs::kNItems];
    float dt_local[kAttrs::kNItems];
    float z_local[kAttrs::kNItems];
    float y_local[kAttrs::kNItems];
    float re_local[kAttrs::kNItems];
    float im_local[kAttrs::kNItems];
    float Aij_local[kAttrs::kNItems] = { 0 };
    float* x_local = (float*) (smem_ + kAttrs::smem_size);
    const float local_B_r = fparams.B_r[j];
    const float local_B_theta = fparams.B_theta[j];
    const float local_C_r = fparams.C_r[j];
    const float local_C_theta = fparams.C_theta[j];
    const float local_D = fparams.D[i];
    float local_x_bias = fparams.x_bias[i*fparams.n + j];
    float loc_hid_real = fparams.prev_hid_real[b*fparams.d*fparams.n + i*fparams.n + j];
    float loc_hid_imag = fparams.prev_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j];

    const int k_max_striped = min((fparams.seq_length - thread_id) / blockDim.x, kAttrs::kNItems-1);

    // load x onto shared memory
    for(int p=thread_id; p<fparams.d; p+=blockDim.x){
        x_local[p] = fparams.x[i*x_dim2_stride + j*x_dim1_stride + p];
    }

    // A = einsum(x, u, "i j r, b l r -> b l i j") + x_bias
    // This full section computes Aij and stores in local memory.
    // Also after computation, u_local should contain u_i
    for(int p=0; p<i; p++){
        __syncthreads();
        kAttrs::BlockLoadT(smem_load).Load(fparams.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, fparams.seq_length);

        #pragma unroll
        for(int k=0; k<kAttrs::kNItems; k++){
            Aij_local[k] += u_local[k] * x_local[p];
        }
    }

//     __syncthreads();

    for(int p=i+1; p<fparams.d; p++){
        __syncthreads();
        kAttrs::BlockLoadT(smem_load).Load(fparams.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, fparams.seq_length, 0);
//         local_x_temp = fparams.x[i*x_dim2_stride + j*x_dim1_stride + p];
        #pragma unroll
        for(int k=0; k<kAttrs::kNItems; k++){
            Aij_local[k] += u_local[k] * x_local[p];
        }
    }
//     __syncthreads();

    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(fparams.u + (b*uyz_dim2_stride + i*uyz_dim1_stride), u_local, fparams.seq_length, 0);
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){
        Aij_local[k] += u_local[k] * x_local[i] + local_x_bias;
    }
    // end of Aij computation

    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(fparams.dt + (b*uyz_dim2_stride + i*uyz_dim1_stride), dt_local, fparams.seq_length, 0);

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
        re_local[k] = local_B_r * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]);
        im_local[k] = local_B_r * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]);
    }
    __syncthreads();
    kAttrs::BlockScanT(smem_scan).InclusiveSum(re_local, re_local);
    __syncthreads();
    kAttrs::BlockScanT(smem_scan).InclusiveSum(im_local, im_local);

    // y = (y + u * D) * silu(z)
    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(fparams.z + (b*uyz_dim2_stride + i*uyz_dim1_stride), z_local, fparams.seq_length, 0);
    // NOTE: the rest of the forward pass is not required for backward pass

    ///////////////// END Recomputing forward pass

    float local_d_var[kAttrs::kNItems] = { 0 };
    float local_d_var1[kAttrs::kNItems] = { 0 };
    float local_d_var2[kAttrs::kNItems] = { 0 };
    float local_d_var3[kAttrs::kNItems] = { 0 };
    float local_d_G[kAttrs::kNItems] = { 0 };
    float local_d_u[kAttrs::kNItems] = { 0 };
    float local_grad_out[kAttrs::kNItems] = { 0 };
    float local_grad_out_1[kAttrs::kNItems] = { 0 };

    ////////////////// Level 1

    // dL/dD - VERIFIED
    float (&local_grad_y)[kAttrs::kNItems] = local_grad_out;
    float (&local_grad_D)[kAttrs::kNItems] = local_d_var;

    __syncthreads();
    kAttrs::BlockLoadT(smem_load).Load(bparams.grad_y + (b*uyz_dim2_stride + i*uyz_dim1_stride),
                                   local_grad_y,
                                   fparams.seq_length, 0);
    for(int k=0; (k < kAttrs::kNItems) && thread_id*kAttrs::kNItems + k < fparams.seq_length; k++){
        if(j == 0){
//             printf("[%d %d %d - %d] k: %d, t: %d grad_y: %f z: %f u: %f \n",
//                      b, i, j, thread_id, k, thread_id*kAttrs::kNItems+k, local_grad_y[k], z_local[k], u_local[k]);
            local_grad_D[k] = local_grad_y[k] * silu(z_local[k]) * silu(u_local[k]);
            local_d_u[k] = local_grad_y[k] * d_silu(u_local[k]) * local_D * silu(z_local[k]) ;
        }
    }
    __syncthreads();
    float gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_grad_D);

    __syncthreads();
    if(thread_id == 0){
        atomicAdd(bparams.grad_D+i, gradient_aggregate);
    }  // aggregate is only valid on thread0

    // dL/dz - VERIFIED
    float (&local_grad_z)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_grad_z[k] = 0; // reset the gradients to 0
        if(j == 0){
            local_grad_z[k] = local_grad_y[k] * local_D * silu(u_local[k]) * d_silu(z_local[k]);
        }
        local_grad_z[k] += local_grad_y[k] *
                           ( local_C_r * cosf(local_C_theta + Gij_local[k]) * ( re_local[k] + loc_hid_real ) -
                             local_C_r * sinf(local_C_theta + Gij_local[k]) * ( im_local[k] + loc_hid_imag ) ) *
                           d_silu(z_local[k]);
    }

    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(local_grad_z, local_grad_z);
    for(int k=0; (k < kAttrs::kNItems) && (k*blockDim.x + thread_id < fparams.seq_length); k++){
        atomicAdd(bparams.grad_z + (b*uyz_dim2_stride + i*uyz_dim1_stride + k*blockDim.x + thread_id),
              local_grad_z[k]);
    }

    #pragma unroll
    for(int k=0; k < kAttrs::kNItems; k++){
        local_grad_z[k] = 0;
    }

    // dL/dC_r
    float (&local_grad_C_r)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_grad_C_r[k] = local_grad_y[k] *
                                ( cosf(local_C_theta + Gij_local[k]) * (re_local[k] + loc_hid_real ) -
                                  sinf(local_C_theta + Gij_local[k]) * (im_local[k] + loc_hid_imag) ) *
                                  silu(z_local[k]);
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_grad_C_r);

//     #pragma unroll
//     for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
//         printf("[%d %d %d - %d] k:%d grad_C_r[%d]: %f \n", b, i, j, thread_id, k, thread_id*kAttrs::kNItems + k, local_grad_C_r[k]);
    if(thread_id == 0){
        atomicAdd(bparams.grad_C_r+j, gradient_aggregate);
    }

    // dL/dC_theta
    float (&local_grad_C_theta)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_grad_C_theta[k] = -1 * local_grad_y[k] *
                                ( local_C_r * sinf(local_C_theta + Gij_local[k]) * (re_local[k] + loc_hid_real) +
                                  local_C_r * cosf(local_C_theta + Gij_local[k]) * (im_local[k] + loc_hid_imag) ) *
                                  silu(z_local[k]);
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_grad_C_theta);

    if(thread_id == 0){
        atomicAdd(bparams.grad_C_theta+j, gradient_aggregate);
    }

    // dL/dre
    float (&local_d_re)[kAttrs::kNItems] = local_d_var1;
    float (&local_d_prev_hidden_re)[kAttrs::kNItems] = local_d_var2;
    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_re[k] = local_grad_y[k] *
                        ( local_C_r * cosf(local_C_theta + Gij_local[k]) ) *
                          silu(z_local[k]);
        local_d_prev_hidden_re[k] = local_d_re[k];
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_prev_hidden_re);
    if(thread_id == 0){
        atomicAdd(bparams.grad_prev_hid_real+(b*fparams.d*fparams.n + i*fparams.n + j), gradient_aggregate);
    }

    // dL/dim
    float (&local_d_im)[kAttrs::kNItems] = local_d_var2;
    float (&local_d_prev_hidden_im)[kAttrs::kNItems] = local_d_var3;
    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_im[k] = local_grad_y[k] *
                        ( -1 * local_C_r * sinf(local_C_theta + Gij_local[k]) ) *
                          silu(z_local[k]);
        local_d_prev_hidden_im[k] = local_d_im[k];
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_prev_hidden_im);
    if(thread_id == 0){
        atomicAdd(bparams.grad_prev_hid_imag+(b*fparams.d*fparams.n + i*fparams.n + j), gradient_aggregate);
    }

    int ultimate_id = fparams.seq_length - (thread_id * kAttrs::kNItems) - 1;
    if(ultimate_id >= 0 && ultimate_id < kAttrs::kNItems){
        atomicAdd(bparams.grad_prev_hid_real+(b*fparams.d*fparams.n + i*fparams.n + j),
                 ( bparams.grad_next_hid_real[b*fparams.d*fparams.n + i*fparams.n + j] * cosf(Gij_local[ultimate_id])  +
                   bparams.grad_next_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j] * sinf(Gij_local[ultimate_id]) ) ) ;

        atomicAdd(bparams.grad_prev_hid_imag+(b*fparams.d*fparams.n + i*fparams.n + j),
                 ( -1 * bparams.grad_next_hid_real[b*fparams.d*fparams.n + i*fparams.n + j] * sinf(Gij_local[ultimate_id])  +
                   bparams.grad_next_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j] * cosf(Gij_local[ultimate_id]) ) );

        local_d_re[ultimate_id] += bparams.grad_next_hid_real[b*fparams.d*fparams.n + i*fparams.n + j] * cosf(Gij_local[ultimate_id]) +
                                   bparams.grad_next_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j] * sinf(Gij_local[ultimate_id]) ;

        local_d_im[ultimate_id] += -1 * bparams.grad_next_hid_real[b*fparams.d*fparams.n + i*fparams.n + j] * sinf(Gij_local[ultimate_id]) +
                                   bparams.grad_next_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j] * cosf(Gij_local[ultimate_id]) ;

        local_d_G[ultimate_id] += bparams.grad_next_hid_real[b*fparams.d*fparams.n + i*fparams.n + j] *
                                  ( ( re_local[ultimate_id] + loc_hid_real ) * (-1) * sinf(Gij_local[ultimate_id]) -
                                    ( im_local[ultimate_id] + loc_hid_imag ) * cosf(Gij_local[ultimate_id]) );
        local_d_G[ultimate_id] += bparams.grad_next_hid_imag[b*fparams.d*fparams.n + i*fparams.n + j] *
                                  ( ( re_local[ultimate_id] + loc_hid_real ) * cosf(Gij_local[ultimate_id]) +
                                    ( im_local[ultimate_id] + loc_hid_imag ) * (-1) * sinf(Gij_local[ultimate_id]) );
    }

    ///////////////////////////////////////////////////// Level 2
    __syncthreads();

    // dL/dre
    SSMScanPrefixCallbackOp<float> prefix_op((float)0.0);
    __syncthreads();
    kAttrs::BlockReverseScanT(smem_revscan).InclusiveReverseScan(local_d_re, local_d_re,
                                                                 [](float a, float b){ return a+b; },
                                                                 prefix_op
                                                                 );
    float (&local_d_r)[kAttrs::kNItems] = local_d_re;

    // dL/ds

//     SSMScanPrefixCallbackOp<float> prefix_op1((float)0.0);
    prefix_op.running_prefix = 0 ;
    __syncthreads();
    kAttrs::BlockReverseScanT(smem_revscan).InclusiveReverseScan(local_d_im, local_d_im,
                                                                 [](float a, float b){ return a+b; },
                                                                 prefix_op
                                                                 );
    float (&local_d_s)[kAttrs::kNItems] = local_d_im;

    ///////////////////////////////////////////////////// Level 3
    // dL/dt
    float (&local_d_dt)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_dt[k] = local_d_r[k] * local_B_r * cosf(local_B_theta - Gij_local[k]) * silu(u_local[k]) +
                        local_d_s[k] * local_B_r * sinf(local_B_theta - Gij_local[k]) * silu(u_local[k]);
    }

    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(local_d_dt, local_d_dt);
    for(int k=0; (k < kAttrs::kNItems) && (k*blockDim.x + thread_id < fparams.seq_length); k++){
        atomicAdd(bparams.grad_dt + (b*uyz_dim2_stride + i*uyz_dim1_stride + k*blockDim.x + thread_id),
              local_d_dt[k]);
    }

    // reset memory to zero for B_r and B_theta
    #pragma unroll
    for(int k=0; k < kAttrs::kNItems; k++){
        local_d_dt[k] = 0;
    }

    // dL/dB_r
    float (&local_d_Br)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_Br[k] = local_d_r[k] * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]) +
                        local_d_s[k] * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]);

        local_d_u[k] += local_d_r[k] * local_B_r * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * d_silu(u_local[k]) +
                        local_d_s[k] * local_B_r * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * d_silu(u_local[k]);
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_Br);

    if(thread_id == 0){ atomicAdd(bparams.grad_B_r+j, gradient_aggregate); }

    // dL/dB_theta
    float (&local_d_Btheta)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_Btheta[k] = -1 * local_d_r[k] * local_B_r * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]) +
                            local_d_s[k] * local_B_r * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]);
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_Btheta);

    if(thread_id == 0){ atomicAdd(bparams.grad_B_theta+j, gradient_aggregate); }

    // dL/dG_rho
//     float (&local_d_G)[kAttrs::kNItems] = local_d_var;

    #pragma unroll
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_G[k] += -1 * local_grad_y[k] *
                            ( local_C_r * sinf(local_C_theta + Gij_local[k]) * (re_local[k] + loc_hid_real ) +
                              local_C_r * cosf(local_C_theta + Gij_local[k]) * (im_local[k] + loc_hid_imag ) ) *
                              silu(z_local[k]);

        local_d_G[k] += local_d_r[k] * local_B_r * sinf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]) -
                        local_d_s[k] * local_B_r * cosf(local_B_theta - Gij_local[k]) * dt_local[k] * silu(u_local[k]);
    }

    //////////////////////////////// Level 4

    prefix_op.running_prefix = 0 ;
    __syncthreads();
    kAttrs::BlockReverseScanT(smem_revscan).InclusiveReverseScan(local_d_G, local_d_G,
                                                                 [](float a, float b){ return a+b; },
                                                                 prefix_op
                                                                 );
    float (&local_d_A)[kAttrs::kNItems] = local_d_G;

    /////////////////////////////////  Level 5

    // dL/dx_bias
    float (&local_d_x_bias)[kAttrs::kNItems] = local_d_var1;
    for(int k=0; (k < kAttrs::kNItems) && (thread_id*kAttrs::kNItems + k < fparams.seq_length); k++){
        local_d_x_bias[k] = local_d_A[k];
//         atomicAdd(bparams.grad_x_bias+(i*fparams.n + j), local_d_A[k]);
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_x_bias);

    if(thread_id == 0){ atomicAdd(bparams.grad_x_bias+(i*fparams.n + j), gradient_aggregate); }

    // dL/dx and dL/du  - computed together
    float (&local_d_x)[kAttrs::kNItems] = local_d_var1;
    float local_grad_temp = 0;

    // blockexchange to coalesce atomics
    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(local_d_A, local_d_A);
    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(u_local, u_local);
    __syncthreads();
    kAttrs::BlockExchangeT(smem_exchange).BlockedToStriped(local_d_u, local_d_u);

    // resets the local values to 0
    #pragma unroll
    for(int k=0; k<kAttrs::kNItems; k++){ local_d_x[k] = 0; }

    // after exchange, k indexing changes
//     printf("[%d %d %d - %d] seq_length: %d k_max_striped: %d \n", b, i, j, thread_id, fparams.seq_length, k_max_striped);
    #pragma unroll
    for(int k=0; k <= k_max_striped; k++){
        local_d_x[k] = local_d_A[k] * u_local[k];
        local_grad_temp = local_d_u[k] + (local_d_A[k] * x_local[i]) ;

        atomicAdd(bparams.grad_u + (b*uyz_dim2_stride + i*uyz_dim1_stride + k * blockDim.x + thread_id ),
                   local_grad_temp
                  );
    }

    __syncthreads();
    gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_x);
    if(thread_id == 0){
        atomicAdd(bparams.grad_x+(i*x_dim2_stride + j*x_dim1_stride + i), gradient_aggregate);
    }

    for(int p=0; p<i; p++){
        __syncthreads();
        kAttrs::BlockLoadStripedT(smem_load_striped).Load(fparams.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, fparams.seq_length, 0);
        #pragma unroll
        for(int k=0; (k <= k_max_striped); k++){
            local_d_x[k] = local_d_A[k] * u_local[k];

            local_grad_temp = (local_d_A[k] * x_local[p]) ;

            atomicAdd(bparams.grad_u + (b*uyz_dim2_stride + p*uyz_dim1_stride + k * blockDim.x + thread_id ),
                       local_grad_temp
                      );
        }
        __syncthreads();
        gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_x);
        if(thread_id == 0){
            atomicAdd(bparams.grad_x+(i*x_dim2_stride + j*x_dim1_stride + p), gradient_aggregate);
        }
    }
    for(int p=i+1; p<fparams.d; p++){
        __syncthreads();
        kAttrs::BlockLoadStripedT(smem_load_striped).Load(fparams.u + (b*uyz_dim2_stride + p*uyz_dim1_stride), u_local, fparams.seq_length, 0);
        #pragma unroll
        for(int k=0; (k <= k_max_striped); k++){
            local_d_x[k] = local_d_A[k] * u_local[k];

            local_grad_temp = (local_d_A[k] * x_local[p]) ;

            atomicAdd(bparams.grad_u + (b*uyz_dim2_stride + p*uyz_dim1_stride + k * blockDim.x + thread_id ),
                       local_grad_temp
                      );
        }
        __syncthreads();
        gradient_aggregate = kAttrs::BlockReduceT(smem_reduce).Sum(local_d_x);
        if(thread_id == 0){
            atomicAdd(bparams.grad_x+(i*x_dim2_stride + j*x_dim1_stride + p), gradient_aggregate);
        }
    }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
ssm_adaptive_cuda_backward_unitary(
    torch::Tensor grad_y,
    torch::Tensor grad_next_hid_real,
    torch::Tensor grad_next_hid_imag,
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
    torch::Tensor prev_hid_real,
    torch::Tensor prev_hid_imag
    ) {

//     std::cout<<"grad_y "<<grad_y<<std::endl;

    auto batch_size = u.size(0);
    auto L = u.size(2);
    auto d = u.size(1);
    auto n = B_r.size(0);

    grad_y = grad_y.contiguous();
    u = u.contiguous();
    dt = dt.contiguous();
    x = x.contiguous();
    x_bias = x_bias.contiguous();
    B_r = B_r.contiguous();
    B_theta = B_theta.contiguous();
    C_r = C_r.contiguous();
    C_theta = C_theta.contiguous();
    D = D.contiguous();
    z = z.contiguous();

    auto grad_u = torch::zeros_like(u).contiguous();
//     auto grad_u = torch::zeros({ batch_size, d, n, L }, u.options());
    auto grad_dt = torch::zeros_like(dt).contiguous();
    auto grad_x = torch::zeros_like(x).contiguous();
    auto grad_x_bias = torch::zeros_like(x_bias).contiguous();
    auto grad_B_r = torch::zeros_like(B_r).contiguous();
    auto grad_B_theta = torch::zeros_like(B_theta).contiguous();
    auto grad_C_r = torch::zeros_like(C_r).contiguous();
    auto grad_C_theta = torch::zeros_like(C_theta).contiguous();
    auto grad_D = torch::zeros_like(D).contiguous();
    auto grad_z = torch::zeros_like(z).contiguous();
    auto grad_prev_hid_real = torch::zeros_like(prev_hid_real).contiguous();
    auto grad_prev_hid_imag = torch::zeros_like(prev_hid_imag).contiguous();

    // how blocks are arranged in a grid
    const dim3 GRID_DIM(n, d, batch_size);

    SSMaParams fparams = {
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
        prev_hid_real.data_ptr<float>(),
        prev_hid_imag.data_ptr<float>(),
        nullptr,  // y
        nullptr, // next_hidden_state_real
        nullptr, // next_hidden_state_imag
        batch_size,           // batch_size
        L,                    // seq_length
        d,                    // d
        n,                    // n
    };

    SSMaBackwardParams bparams = {
        grad_y.data_ptr<float>(),  // grad_y
        grad_next_hid_real.data_ptr<float>(),
        grad_next_hid_imag.data_ptr<float>(),
        grad_u.data_ptr<float>(),  // grad_u
        grad_dt.data_ptr<float>(),  // grad_dt
        grad_x.data_ptr<float>(),  // grad_x
        grad_x_bias.data_ptr<float>(),  // grad_x_bias
        grad_B_r.data_ptr<float>(),  // grad_B_r
        grad_B_theta.data_ptr<float>(),   // grad_B_theta
        grad_C_r.data_ptr<float>(),  // grad_C_r
        grad_C_theta.data_ptr<float>(),  // grad_C_theta
        grad_D.data_ptr<float>(),  // grad_D
        grad_z.data_ptr<float>(),  // grad_z
        grad_prev_hid_real.data_ptr<float>(),
        grad_prev_hid_imag.data_ptr<float>(),
    };

    if (fparams.seq_length <= 128) {
        using kAttrs = SSMBackwardKernelAttrs<32, 4>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_backward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(fparams, bparams);
    } else if (fparams.seq_length <= 256) {
        using kAttrs = SSMBackwardKernelAttrs<32, 8>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_backward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(fparams, bparams);
    } else if (fparams.seq_length <= 512) {
        using kAttrs = SSMBackwardKernelAttrs<32, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_backward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(fparams, bparams);
    } else if (fparams.seq_length <= 1024) {
        using kAttrs = SSMBackwardKernelAttrs<64, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_backward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(fparams, bparams);
    } else {
        using kAttrs = SSMBackwardKernelAttrs<128, 16>;
        constexpr dim3 BLOCK_DIM(kAttrs::kNThreads);
        ssm_adaptive_backward_kernel_unitary<kAttrs><<<GRID_DIM, BLOCK_DIM, kAttrs::smem_size + d*sizeof(float)>>>(fparams, bparams);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaDeviceSynchronize();  // wait for kernel to finish

//     std::cout<<"START Cr "<<grad_C_r<<" END Cr"<<std::endl;

    return {grad_u, grad_dt, grad_x, grad_x_bias, grad_B_r, grad_B_theta, grad_C_r, grad_C_theta, grad_D, grad_z, grad_prev_hid_real, grad_prev_hid_imag};
}

// Register the torch library as CUDA
// somehow, torch has compiled CPU -> CUDA, CUDA-> HIPA and so on
// I have no explanation for this!!!
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("ssm_adaptive_backward_unitary", &ssm_adaptive_cuda_backward_unitary);
}

} // END OF namespace
