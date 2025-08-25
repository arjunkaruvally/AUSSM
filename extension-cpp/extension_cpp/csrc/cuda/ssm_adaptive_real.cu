#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

#define MAX_THREADS_PER_BLOCK 128
// #define MAX_SHARED_MEMORY 65536  // in bytes - this is the shared memory that is addressable
// maximum possible floating values: 16384  (int bytes/4)
// for forward number of floats: (4L) maximum L: 4096
// for backward number of floats: (6L+2) maximum L: 2730
// NOTE that when training, the full forward length cannot be used as backward will still be bottlenecked

#define MAX_SHARED_MEMORY 49128  // in bytes - this is the shared memory that is addressable in a 2080Ti
// maximum possible floating values: 12282  (int bytes/4)
// for forward number of floats: (4L) maximum L: 3070
// for backward number of floats: (6L+2) maximum L: 2046
// NOTE that when training, the full forward length cannot be used as backward will still be bottlenecked

// #define MAX_SHARED_MEMORY 48  // debug size
// // For SHARED_MEMORY: 48
// // maximum possible floats: 12 (int bytes/4)
// // forward max L: 5
// // backward max L: 3

namespace extension_cpp{

template <typename scalar_t>
__device__ void cumulative_sum(scalar_t* arr, const int size, const int thread_id, const int num_threads) {
    // up sweep
    int m = 0;
    int s = 2;

    for(; s<=size; s=s<<1){
        for(m=thread_id; m<size; m+=num_threads){
            if( (m+1)%s == 0){
                arr[m] += arr[m - (s >> 1)];
            }
        }
        __syncthreads();
    }
    int adding_index = 0;
    s = s>>1;
    for(; s>1; s=s>>1){
        for(m=thread_id; m<size; m+=num_threads){
            if( (m+1)%s == 0){
                if( (m + (s >> 1)) < size){
                    arr[m + (s >> 1)] += arr[m];
                }
            }
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
        for(m=thread_id; m<size; m+=num_threads){
//             printf("[%d - %d] in up sweep loop with s = %d, %d \n", thread_id, m, s, (m+1)%s);
            if( (m+1)%s == 0){
//                 printf("[%d - %d] in up sweep arr[%d] += arr[%d] => %f \n", thread_id, m, m, m-(s >> 1), arr[m] + arr[m - (s >> 1)]);
                arr[(size - 1) - m] += arr[(size - 1) - (m - (s >> 1))];
            }
        }
        __syncthreads();
    }
    int adding_index = 0;
    s = s>>1;
//     #pragma unroll
    for(; s>1; s=s>>1){
//         #pragma unroll
        for(m=thread_id; m<size; m+=num_threads){
            if( (m+1)%s == 0){
                if( (m + (s >> 1)) < size){
                    arr[(size - 1) - (m + (s >> 1))] += arr[(size - 1) - m];
                }
            }
        }
        __syncthreads();
    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + expf(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t silu(scalar_t z) {
  return z * sigmoid(z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_silu(scalar_t z) {
  return (z * d_sigmoid(z) + sigmoid(z));
}

//////////////////////////////////////////////////////////////////////////////////////////// Complex Logic

template <typename scalar_t>
__global__ void ssm_adaptive_forward_kernel_preal(
    torch::PackedTensorAccessor32<scalar_t,3> u,
    torch::PackedTensorAccessor32<scalar_t,3> dt,
    torch::PackedTensorAccessor32<scalar_t,3> x_rho,
    torch::PackedTensorAccessor32<scalar_t,3> B_r,
    torch::PackedTensorAccessor32<scalar_t,3> C_r,
    torch::PackedTensorAccessor32<scalar_t,1> D,
    torch::PackedTensorAccessor32<scalar_t,3> z,
    torch::PackedTensorAccessor32<scalar_t,3> y,
    torch::PackedTensorAccessor32<scalar_t,5> hidden_states,
    const int batch_size,
    const int seq_length,
    const int d,
    const int n,
    const int chunk_size) {
    //batch index
    const int b = blockIdx.z;
    const int i = blockIdx.y;
    const int j = blockIdx.x;

    // cumulative array is in a memory location shared by
    extern __shared__ float smem[];

    float* G_rho = &smem[0];
    float* A_rho = &smem[0];
    float* r = &smem[0];
    float* re = &smem[0];

    float last_hidden_re = hidden_states[b][i][j][0][0];
    // time index
    const int thread_id = threadIdx.x;

    // handling longer sequence lengths
    int local_seq_length = 0;
    int local_offset = 0;

    // total time: O( d(l/P) [A] + O(log(L)) [G] + (L/P)*log(L) [parallel scans] ) - O( max(L/P , log(L)) )
    int chunk_id = 0;
    for(local_offset = 0; local_offset < seq_length; local_offset += chunk_size){
        // two variables - local_offset and local_seq_length indicate the start and end of local sequence processing
        local_seq_length = min(chunk_size, seq_length - local_offset);
        if(local_seq_length < 1){
            break;
        }
        chunk_id += 1;

        G_rho = &smem[0];
        A_rho = &smem[0];
        r = &smem[local_seq_length];
        re = &smem[local_seq_length];

        // O(d(l/P)) time where P is the number of parallel processes
        for(int l=thread_id; l < local_seq_length; l += blockDim.x){
            A_rho[l] = 0;
            for(int r=0; r<d; r++){
                A_rho[l] += x_rho[i][j][r] * u[b][r][local_offset + l];
            }
        }
        __syncthreads();  // sync threads after computing A

        // O(log(L) L/P) time
        cumulative_sum(A_rho, local_seq_length, thread_id, blockDim.x);  // this computes G - there is a syncthreads at the end of the scan

        // O(L/P)
        for(int k=thread_id; k<local_seq_length; k += blockDim.x){
            r[k] = B_r[b][j][local_offset + k] * expf( - G_rho[k])  * dt[b][i][local_offset + k] * silu(u[b][i][local_offset + k]) ;
        }
        __syncthreads();

        // O(2 log(L) L/P) time
        cumulative_sum(r, local_seq_length, thread_id, blockDim.x);  // there is a syncthreads at the end of the scan algorithm

        // O(L/P)
        for(int t=thread_id; t<local_seq_length; t += blockDim.x){
            if(j == 0){ atomicAdd(&y[b][i][local_offset + t], silu(z[b][i][local_offset + t])*silu(u[b][i][local_offset + t])*D[i]); }
            atomicAdd(&y[b][i][local_offset + t],
                        C_r[b][j][local_offset + t] * expf(G_rho[t]) * silu(z[b][i][local_offset + t]) * (re[t] + last_hidden_re)
            ) ;
        }

        float local_hidden_re = last_hidden_re;
        int t = local_seq_length - 1;

        // IMPORTANT: this is necessary for the threads to get the correct values of the hidden state.
        // directly accessing hidden_state array leads to race conditions possibly imposed by compiler optimizing out stuff
        // and mistakenly using outdated data in the local registers
        last_hidden_re = expf(G_rho[t]) * (re[t] + local_hidden_re);
        if(thread_id == 0){
            hidden_states[b][i][j][chunk_id][0] = last_hidden_re;
        }
        __syncthreads();  // wait for all threads before starting the forward loop as G, re and im may get overwritten in the next loop start
    }
}


std::tuple<torch::Tensor, torch::Tensor> ssm_adaptive_cuda_forward_preal(
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor x_rho,
    torch::Tensor B_r,
    torch::Tensor C_r,
    torch::Tensor D,
    torch::Tensor z) {

    auto batch_size = u.size(0);
    auto L = u.size(2);
    auto d = u.size(1);
    auto n = B_r.size(1);

    const int maximum_floats = MAX_SHARED_MEMORY >> 2;  // this is the number of floats that can be stored in shared memory without overflow
//     const int shared_memory_constant = 0;
//     const int shared_memory_factor = 2;
    const int shared_memory_constant = 2;  // We use the backward bottleneck shared mmeory size
    const int shared_memory_factor = 3;
    const int required_shared_floats = shared_memory_factor*L + shared_memory_constant;
    const int chunk_size = (int) std::floor((maximum_floats - shared_memory_constant)*1.0 / shared_memory_factor);
    const int number_of_chunks = (int)std::ceil(L*1.0/chunk_size);  // IMPORTANT: in c++, the *1.0 will convert the int L to float so that we get the correct floating point chunk size
    const int shared_memory_size = std::min((int)required_shared_floats, (int)maximum_floats)*sizeof(float);
    // how threads are arranged within a block
    // the maximum of x*y*z as 1024
    const dim3 BLOCK_DIM(std::min((int)L, (int)MAX_THREADS_PER_BLOCK));

    // how blocks are arranged in a grid
    const dim3 GRID_DIM(n, d, batch_size);

    auto y = torch::zeros_like(u);

//     std::cout << "forward with number_of_chunks: " << number_of_chunks << " chunk_size: "<< chunk_size << " shared_memory: "<< shared_memory_size << "\n";

    auto hidden_states = torch::zeros({batch_size, d, n, number_of_chunks+1, 1}, u.options());  // only 1 for real part

    ssm_adaptive_forward_kernel_preal<float><<<GRID_DIM, BLOCK_DIM, shared_memory_size>>>(
        u.packed_accessor32<float,3>(),
        dt.packed_accessor32<float,3>(),
        x_rho.packed_accessor32<float,3>(),
        B_r.packed_accessor32<float,3>(),
        C_r.packed_accessor32<float,3>(),
        D.packed_accessor32<float,1>(),
        z.packed_accessor32<float,3>(),
        y.packed_accessor32<float,3>(),
        hidden_states.packed_accessor32<float,5>(),
        batch_size,
        L,
        d,
        n,
        chunk_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {y, hidden_states};
}

template <typename scalar_t>
__global__ void ssm_adaptive_backward_kernel_preal(
    torch::PackedTensorAccessor32<scalar_t,3> grad_y,
    torch::PackedTensorAccessor32<scalar_t,5> grad_h,
    torch::PackedTensorAccessor32<scalar_t,3> u,
    torch::PackedTensorAccessor32<scalar_t,3> dt,
    torch::PackedTensorAccessor32<scalar_t,3> x_rho,
    torch::PackedTensorAccessor32<scalar_t,3> B_r,
    torch::PackedTensorAccessor32<scalar_t,3> C_r,
    torch::PackedTensorAccessor32<scalar_t,1> D,
    torch::PackedTensorAccessor32<scalar_t,3> z,
    torch::PackedTensorAccessor32<scalar_t,3> grad_u,
    torch::PackedTensorAccessor32<scalar_t,3> grad_dt,
    torch::PackedTensorAccessor32<scalar_t,3> grad_x_rho,
    torch::PackedTensorAccessor32<scalar_t,3> grad_B_r,
    torch::PackedTensorAccessor32<scalar_t,3> grad_C_r,
    torch::PackedTensorAccessor32<scalar_t,1> grad_D,
    torch::PackedTensorAccessor32<scalar_t,3> grad_z,
    torch::PackedTensorAccessor32<scalar_t,5> hidden_states,
    const int batch_size,
    const int seq_length,
    const int d,
    const int n,
    const int chunk_size) {
    //batch index
    const int b = blockIdx.z;

    // index for dimension d
    const int q = blockIdx.x;
    const int p = blockIdx.y;

    // time index
    const int thread_id = threadIdx.x;

    const int j = q;
    const int i = p;

    float ugradUpdated = 0;
    float dtgradUpdated = 0;
    float xgradUpdated = 0;
    float BgradUpdated = 0;
    float CgradUpdated = 0;
    float DgradUpdated = 0;
    float zgradUpdated = 0;

    extern __shared__ float smem[];
    float* grad_re = &smem[0];
    float* grad_r = &smem[0];
    float* grad_G_rho = &smem[0];  // gradG needs auxiliary space as G is required for level 1 and 2
    float* grad_A_rho = &smem[0];
    float* grad_next_hidden = &smem[min(3*chunk_size, 3*seq_length)];  // this will be either the minimum of chunk size or seq length

    float* G_rho = &smem[0];
    float* A_rho = &smem[0];
    float* r = &smem[0];
    float* re = &smem[0];

    const int last_chunk_id = hidden_states.size(3)-1;

//     float grad_next_hidden_real = grad_h[b][i][j][last_chunk_id][0];
//     float grad_next_hidden_imag = grad_h[b][i][j][last_chunk_id][1];

    if(thread_id == 0){
        grad_next_hidden[0] = grad_h[b][i][j][last_chunk_id][0];
    }

    float grad_cur_hidden_real = 0;

    int local_offset = 0;
    int local_seq_length = 0;
    int chunk_id = last_chunk_id+1;

    // This implements the BPTT algorithm
    for(local_offset = last_chunk_id*chunk_size; local_offset >= 0; local_offset -= chunk_size, chunk_id -= 1){
        __syncthreads();  // call syncthreads before doing anything

        // two variables - local_offset and local_seq_length indicate the start and end of local sequence processing
        local_seq_length = min(chunk_size, seq_length - local_offset);
        if(local_seq_length < 1){
            continue;
        }

//         printf("[%d %d %d %d] in loop with local_offset: %d local_seq_length: %d chunk_size: %d chunk_id: %d grad_next_real: %f grad_next_imag: %f \n",
//                  b, i, j, thread_id,                 local_offset,     local_seq_length,   chunk_size,   chunk_id,    grad_next_hidden[0],  grad_next_hidden[1]);

        // in the order of shared memory space allocated
        // real variables get the earlier spaces if we want to switch the kernel to have pure real eigenvalues
        G_rho = &smem[0];
        A_rho = &smem[0];
        grad_re = &smem[local_seq_length];
        grad_r = &smem[local_seq_length];
        r = &smem[local_seq_length];
        re = &smem[local_seq_length];
        grad_G_rho = &smem[2*local_seq_length];
        grad_A_rho = &smem[2*local_seq_length];

        // reset the grads for the next BPTT pass
        ugradUpdated = 0;
        dtgradUpdated = 0;
        xgradUpdated = 0;
        BgradUpdated = 0;
        CgradUpdated = 0;
        DgradUpdated = 0;
        zgradUpdated = 0;

        // O(d(l/P)) time where P is the number of parallel processes
        for(int l=thread_id; l < local_seq_length; l += blockDim.x){
            A_rho[l] = 0;
            for(int r=0; r<d; r++){
                A_rho[l] += x_rho[i][j][r] * u[b][r][local_offset + l];
            }
        }
        __syncthreads();  // sync threads after computing A

        // O(log(L) 2L/P) time
        cumulative_sum(A_rho, local_seq_length, thread_id, blockDim.x);  // this computes G

        // O(L/P)
        for(int k=thread_id; k<local_seq_length; k += blockDim.x){
            r[k] = B_r[b][j][local_offset + k] * expf(- G_rho[k]) * dt[b][i][local_offset +k] * silu(u[b][i][local_offset + k]) ;
        }
        __syncthreads();

        // O(2 log(L)) time
        cumulative_sum(r, local_seq_length, thread_id, blockDim.x);

        /////////////////// Backpropagating gradients

        /////// Level 1 - O(L/P)
        // implement backpropagation algorithm for better time and space efficiency
        for(int m=thread_id; m<local_seq_length; m += blockDim.x){
            grad_G_rho[m] = 0;

            grad_G_rho[m] += grad_y[b][p][local_offset + m] * silu(z[b][p][local_offset + m]) *  C_r[b][j][local_offset + m] * expf(G_rho[m]) *
                        ( (re[m] + hidden_states[b][i][j][chunk_id-1][0])  ) ;

            // gradients from the D term
            if(j == 0) {
                zgradUpdated = d_silu(z[b][p][local_offset + m]) * D[p] * silu(u[b][p][local_offset + m]) * grad_y[b][p][local_offset + m];
                atomicAdd(&grad_u[b][p][local_offset + m],
                    d_silu(u[b][p][local_offset + m]) * D[p] * silu(z[b][p][local_offset + m]) * grad_y[b][p][local_offset + m]);
                atomicAdd(&grad_D[p], silu(z[b][p][local_offset + m]) * silu(u[b][p][local_offset + m]) * grad_y[b][p][local_offset + m]);
            } // only the first j thread does this as there is only 1 term along that dimension

            atomicAdd(&grad_C_r[b][q][local_offset + m],
                      grad_y[b][i][local_offset + m] * silu(z[b][p][local_offset + m]) * expf(G_rho[m]) *
                      ( (re[m] + hidden_states[b][i][j][chunk_id-1][0]) ) );
            atomicAdd(&grad_z[b][p][local_offset + m],
                      zgradUpdated + grad_y[b][p][local_offset + m] * C_r[b][j][local_offset + m] * expf(G_rho[m]) *
                      ( (re[m] + hidden_states[b][i][j][chunk_id-1][0] ) )
                      * d_silu(z[b][p][local_offset + m]) );

            // this needs to be computed for each thread separately to avoid compiler register optimizations
            grad_cur_hidden_real += grad_y[b][p][local_offset + m] * silu(z[b][p][local_offset + m]) * C_r[b][j][local_offset + m] * expf(G_rho[m]);

            // O(l/P)  - these overwrites re and im
            if(m == local_seq_length - 1){  // gradients from the next hidden state BPTT

                grad_G_rho[m] +=      expf(G_rho[m]) * grad_next_hidden[0] *
                                        ( (re[m] + hidden_states[b][i][j][chunk_id-1][0]) );

                grad_re[m] = grad_next_hidden[0] *  expf(G_rho[m]) ;

                grad_cur_hidden_real += grad_next_hidden[0] *  expf(G_rho[m]) ;

            } else {
                grad_re[m] = 0 ;
            }
            grad_re[m] += grad_y[b][p][local_offset + m] * silu(z[b][p][local_offset + m]) * C_r[b][j][local_offset + m] * expf(G_rho[m]) ;
        }

        __syncthreads();  // sync threads as computing grads for re and im will overwrite re and im

        // reset these after using and before doing other computations
        grad_next_hidden[0] = 0;

        /////// Level 2 - O(log(L) L/P)

        // O(log(L) L/P)
        cumulative_right_sum(grad_re, local_seq_length, thread_id, blockDim.x);  // this will give grad_r

        /////// Level 3 - O(L/P)

        for(int m=thread_id; m<local_seq_length; m += blockDim.x){
            atomicAdd(&grad_dt[b][i][local_offset + m],
                       grad_r[m] * B_r[b][j][local_offset+m] * expf(-G_rho[m]) * silu(u[b][i][local_offset + m])
            );

            atomicAdd(&grad_B_r[b][j][local_offset + m],
                        grad_r[m] * expf(-G_rho[m]) * dt[b][i][local_offset + m] * silu(u[b][i][local_offset + m])
            );

            atomicAdd(&grad_u[b][i][local_offset + m],
                       grad_r[m] *  B_r[b][j][local_offset+m] * expf(-G_rho[m]) * dt[b][i][local_offset + m] * d_silu(u[b][i][local_offset + m])
            );
            grad_G_rho[m] +=
                      - grad_r[m] * B_r[b][j][local_offset+m] * expf(-G_rho[m]) * dt[b][i][local_offset + m] * silu(u[b][i][local_offset + m]) ;
        }
        __syncthreads();

        /////// Level 4 - O(logL L/P)
        cumulative_right_sum(grad_G_rho, local_seq_length, thread_id, blockDim.x);

        /////// Level 5 - O(d L/P)
        for(int m=thread_id; m<local_seq_length; m += blockDim.x){
            for(int r=0; r<d; r++){
                atomicAdd(&grad_x_rho[i][j][r],
                           grad_A_rho[m]*u[b][r][local_offset + m]
                );

                atomicAdd(&grad_u[b][r][local_offset + m],
                           grad_A_rho[m]*x_rho[i][j][r]
                );
            }
        }

        ///////  reset the hidden gradients for the next BPTT steps
        atomicAdd(&grad_next_hidden[0], grad_cur_hidden_real);

        grad_cur_hidden_real = 0;
    }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> ssm_adaptive_cuda_backward_preal(
    torch::Tensor grad_y,
    torch::Tensor grad_h,
    torch::Tensor u,
    torch::Tensor dt,
    torch::Tensor x_rho,
    torch::Tensor B_r,
    torch::Tensor C_r,
    torch::Tensor D,
    torch::Tensor z,
    torch::Tensor hidden_states
    ) {

    auto batch_size = u.size(0);
    auto L = u.size(2);
    auto d = u.size(1);
    auto n = B_r.size(1);

    auto grad_u = torch::zeros_like(u);
    auto grad_dt = torch::zeros_like(dt);
    auto grad_x_rho = torch::zeros_like(x_rho);
    auto grad_B_r = torch::zeros_like(B_r);
    auto grad_C_r = torch::zeros_like(C_r);
    auto grad_D = torch::zeros_like(D);
    auto grad_z = torch::zeros_like(z);

    const int maximum_floats = MAX_SHARED_MEMORY >> 2;  // this is the number of floats that can be stored in shared memory without overflow
    const int shared_memory_constant = 2;
    const int shared_memory_factor = 3;
    const int required_shared_floats = shared_memory_factor*L + shared_memory_constant;
    const int chunk_size = (int) std::floor((maximum_floats - shared_memory_constant)*1.0 / shared_memory_factor);
    const int shared_memory_size = std::min((int)required_shared_floats, (int)maximum_floats)*sizeof(float);
    const int number_of_chunks = (int)std::ceil(L*1.0/chunk_size);

    // how threads are arranged within a block
    // the maximum of x*y*z as 1024
    const dim3 BLOCK_DIM(std::min((int)L, (int)MAX_THREADS_PER_BLOCK));  // TODO: current assumption L < MAX_THREADS_PER_BLOCK

    // how blocks are arranged in a grid
    const dim3 GRID_DIM(n, d, batch_size);

//     std::cout << "backward with number_of_chunks: " << number_of_chunks << " chunk_size: "<< chunk_size << " shared_memory: "<< shared_memory_size << "\n";

//     AT_DISPATCH_FLOATING_TYPES(A.type(), "ssm_adaptive_backward_cuda", ([&] {
    ssm_adaptive_backward_kernel_preal<float><<<GRID_DIM, BLOCK_DIM, shared_memory_size>>>(
        grad_y.packed_accessor32<float,3>(),
        grad_h.packed_accessor32<float,5>(),
        u.packed_accessor32<float,3>(),
        dt.packed_accessor32<float,3>(),
        x_rho.packed_accessor32<float,3>(),
        B_r.packed_accessor32<float,3>(),
        C_r.packed_accessor32<float,3>(),
        D.packed_accessor32<float,1>(),
        z.packed_accessor32<float,3>(),
        grad_u.packed_accessor32<float,3>(),
        grad_dt.packed_accessor32<float,3>(),
        grad_x_rho.packed_accessor32<float,3>(),
        grad_B_r.packed_accessor32<float,3>(),
        grad_C_r.packed_accessor32<float,3>(),
        grad_D.packed_accessor32<float,1>(),
        grad_z.packed_accessor32<float,3>(),
        hidden_states.packed_accessor32<float,5>(),
        batch_size,
        L,
        d,
        n,
        chunk_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_u, grad_dt, grad_x_rho, grad_B_r, grad_C_r, grad_D, grad_z};
}

// Register the torch library as CUDA
// somehow, torch has compiled CPU -> CUDA, CUDA-> HIPA and so on
// I have no explanation for this!!!
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("ssm_adaptive_preal", &ssm_adaptive_cuda_forward_preal);
  m.impl("ssm_adaptive_backward_preal", &ssm_adaptive_cuda_backward_preal);
}

} // END OF namespace
