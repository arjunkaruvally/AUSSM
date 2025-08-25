#pragma once

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>
#include <cub/cub.cuh>

#include "reverse_scan.h"

#define MAX_THREADS_PER_BLOCK 128
// #define MAX_SHARED_MEMORY 65536  // in bytes - this is the shared memory that is addressable
// maximum possible floating values: 16384  (int bytes/4)
// for forward number of floats: (3L) maximum L: 5461
// for backward number of floats: (4L+2) maximum L: 4095
// NOTE that when training, the full forward length cannot be used as backward will still be bottlenecked

// #define MAX_SHARED_MEMORY 50176  // in bytes - this is the shared memory that is addressable
// // maximum possible floating values: 12544  (int bytes/4)
// // for forward number of floats: (3L) maximum L: 4181
// // for backward number of floats: (4L+2) maximum L: 3135
// // NOTE that when training, the full forward length cannot be used as backward will still be bottlenecked

#define MAX_SHARED_MEMORY 49128  // in bytes - this is the shared memory that is addressable in a 2080Ti
#define NUM_BANKS 16
#define _USE_MATH_DEFINES

#include <math.h>


constexpr size_t custom_max(std::initializer_list<size_t> ilist){
    return *std::max_element(ilist.begin(), ilist.end());
}

/*
    These are some custom device functions used by the kernels
*/

namespace constants
{
    const float GLOBAL_FLOAT_ONE = 1.0;
}

template <typename scalar_t>
__device__ void cumulative_sum(scalar_t* arr, const int size, const int thread_id, const int num_threads) {
    // up sweep
    int m = 0;
    int s = 2;

    for(; s<=size; s=s<<1){
        for(int m = (thread_id+1) * s - 1; m < size; m += s * (num_threads) ){
            arr[m] += arr[m - (s >> 1)];
        }
        __syncthreads();
    }

    s = s>>1;
    for(; s>1; s=s>>1){
        for(int m = (thread_id+1) * s - 1; m + (s>>1) < size; m += s * (num_threads)){
            arr[m + (s >> 1)] += arr[m];
        }
        __syncthreads();
    }
}

// in right cumulative sum, all the indices are anchored to the right and moves from the right
template <typename scalar_t>
__device__ void cumulative_right_sum(scalar_t* arr, const int size, const int thread_id, const int num_threads) {
    // up sweep
    int m = 0;
    int s = 2;

    for(; s<=size; s=s<<1){
        for(int m = (thread_id+1) * s - 1; m < size; m += s * (num_threads) ){
            arr[(size - 1) - m] += arr[(size - 1) - (m - (s >> 1))];
        }
        __syncthreads();
    }

    s = s>>1;
    for(; s>1; s=s>>1){
        for(int m = (thread_id+1) * s - 1; m + (s>>1) < size; m += s * (num_threads)){
            arr[(size - 1) - (m + (s >> 1))] += arr[(size - 1) - m];
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
//  float one = 1.0;
  return constants::GLOBAL_FLOAT_ONE / (constants::GLOBAL_FLOAT_ONE + expf(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (constants::GLOBAL_FLOAT_ONE - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t silu(scalar_t z) {
  return z * sigmoid(z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_silu(scalar_t z) {
  return (z * d_sigmoid(z) + sigmoid(z));
}


/*
    This sets some of the variables requied by the kernel. Variables used for assigning work to each thread can be
    set here as templates so that they are stored in the local memory rather than the slow global memory of the kernel
*/

template<int NThreads_, int NItems_>
struct SSMKernelAttrs {
    static_assert(NItems_ % 4 == 0, "The kernel expects each thread to hold a multiple of 4 compute items.");
    static constexpr int kNThreads = NThreads_;
    static constexpr int kNItems = NItems_;

    using BlockLoadT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<float, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT = cub::BlockScan<float, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;

    static constexpr int smem_size = custom_max({
                                                 sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockScanT::TempStorage),
                                                 sizeof(typename BlockExchangeT::TempStorage)
                                                 });

    static constexpr int chunk_size = kNItems * kNThreads;
};


template<int NThreads_, int NItems_>
struct SSMBackwardKernelAttrs {
    static_assert(NItems_ % 4 == 0, "The kernel expects each thread to hold a multiple of 4 compute items.");
    static constexpr short kNThreads = NThreads_;
    static constexpr short kNItems = NItems_;

    using BlockLoadT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadStripedT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_STRIPED>;
    using BlockStoreT = cub::BlockStore<float, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT = cub::BlockScan<float, kNThreads>;
    using BlockReduceT = cub::BlockReduce<float, kNThreads>;
    using BlockReverseScanT = BlockReverseScan<float, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;

    static constexpr int smem_size = custom_max({
                                                 sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadStripedT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockScanT::TempStorage),
                                                 sizeof(typename BlockReduceT::TempStorage),
                                                 sizeof(typename BlockReverseScanT::TempStorage),
                                                 sizeof(typename BlockExchangeT::TempStorage)
                                                 });

    static constexpr short chunk_size = kNItems * kNThreads;
};


struct SSMaBackwardParams {
    const float *__restrict__ grad_y;
    const float *__restrict__ grad_next_hid_real;
    const float *__restrict__ grad_next_hid_imag;
    float *__restrict__ grad_u;
    float *__restrict__ grad_dt;
    float *__restrict__ grad_x;
    float *__restrict__ grad_x_bias;
    float *__restrict__ grad_B_r;
    float *__restrict__ grad_B_theta;
    float *__restrict__ grad_C_r;
    float *__restrict__ grad_C_theta;
    float *__restrict__ grad_D;
    float *__restrict__ grad_z;
    float *__restrict__ grad_prev_hid_real;
    float *__restrict__ grad_prev_hid_imag;
};


struct SSMaParams {
    const float *__restrict__ u;
    const float *__restrict__ dt;
    const float *__restrict__ x;
    const float *__restrict__ x_bias;
    const float *__restrict__ B_r;
    const float *__restrict__ B_theta;
    const float *__restrict__ C_r;
    const float *__restrict__ C_theta;
    const float *__restrict__ D;
    const float *__restrict__ z;
    const float *__restrict__ prev_hid_real;
    const float *__restrict__ prev_hid_imag;
    float *__restrict__ y;
    float *__restrict__ next_hid_real;
    float *__restrict__ next_hid_imag;
    const int batch_size;
    const int seq_length;
    const int d;
    const int n;
};
