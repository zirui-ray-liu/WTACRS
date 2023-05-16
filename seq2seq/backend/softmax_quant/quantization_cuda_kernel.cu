/*
 * Cuda kernels for quantization and mixed-precision packing.
 */

#include <torch/extension.h>
// #include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)
#define fmax(a, b) ((a) > (b) ? (a): (b))

using torch::IntArrayRef;
using torch::Tensor;

/****************************************/
/***** Pack/Unpack Single Precision *****/
/****************************************/
template <typename scalar_t>
__global__ void compute_scale_single_precision_kernel(int32_t bits,
                                                      const scalar_t* __restrict__ min,
                                                      const scalar_t* __restrict__ max,
                                                      scalar_t* __restrict__ scale,
                                                      int64_t N,
                                                      int64_t num_groups) {
  int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups) {
    scale[id] = ((scalar_t)((1 << bits) - 1)) / (max[id] - min[id] + 2e-6);
  }
}

// Pack float16/32 data into int8 bit stream
template<typename scalar_t, bool boundary_check>
__global__ void pack_single_precision_kernel(int32_t bits,
                                             const scalar_t* __restrict__ data,
                                             const scalar_t* __restrict__ scale,
                                             const scalar_t* __restrict__ min,
                                             int8_t* __restrict__ packed,
                                             std::pair<uint64_t, uint64_t> seeds,
                                             int64_t N,
                                             int64_t num_groups,
                                             int64_t group_size,
                                             int64_t block_idx_y_base) {
  const int64_t no = blockIdx.y + block_idx_y_base;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int work_per_thread = 8 / bits;
  const int64_t global_thread_id = (no * num_groups + group_id) * group_size + d;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);

  uint8_t local_packed = 0;
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int64_t id = (n * num_groups + group_id) * group_size + d;
    const float noise = curand_uniform(&state);
    const int32_t val = __float2int_rn(fmax((data[id] - min[n * num_groups + group_id]) * scale[n * num_groups + group_id] + noise - 0.5, 0.0f));
    local_packed |= (val << (ni * bits));
  }

  packed[global_thread_id] = local_packed;
}

// Pack float16/32 data into int8 bit stream
std::pair<Tensor, Tensor> pack_single_precision_cuda(Tensor data,
                                                     Tensor min,
                                                     Tensor max,
                                                     int bits,
                                                     bool stochastic) {
  int64_t N = data.size(0);
  int64_t num_groups = data.size(1);
  int64_t group_size = data.size(2);

  // Compute total bits
  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  int64_t N_round = N + (work_per_thread - N % work_per_thread) % work_per_thread;
  int64_t total_bits = (int64_t)bits * (N_round * num_groups * group_size);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8,}, options);

  // Compute scale
  options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
  Tensor scale = torch::empty({N, num_groups, 1}, options);
  int threads = 256;
  int blocks = (N * num_groups + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "compute_scale_single_precision", ([&] {
    compute_scale_single_precision_kernel<<<blocks, threads>>>(
      bits, min.data_ptr<scalar_t>(), max.data_ptr<scalar_t>(),
      scale.data_ptr<scalar_t>(), N, num_groups);
  }));

  // Random number generator
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);
  }
  TORCH_CHECK(stochastic);

  // Call pack kernels
  int64_t logical_block_y_dim = (N + work_per_thread - 1) / work_per_thread;
  for (int64_t block_idx_y_base = 0; block_idx_y_base < logical_block_y_dim; block_idx_y_base += BLOCK_Y_DIM_MAX) {
    dim3 block_dim(num_groups, std::min(logical_block_y_dim - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
    dim3 thread_dim(group_size, 1, 1);
  
    if (N % work_per_thread == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
        pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<scalar_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          packed.data_ptr<int8_t>(),
          rng_engine_inputs,
          N, num_groups, group_size, block_idx_y_base);
      }));
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
        pack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<scalar_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          packed.data_ptr<int8_t>(),
          rng_engine_inputs,
          N, num_groups, group_size, block_idx_y_base);
      }));
    }
  }

  return std::make_pair(packed, scale);
}

// Unpack int32 bit stream to float16/32 data
template<typename scalar_t, bool boundary_check>
__global__ void unpack_single_precision_kernel(int32_t bits,
                                               const int8_t* __restrict__ data,
                                               const scalar_t* __restrict__ scale,
                                               const scalar_t* __restrict__ min,
                                               scalar_t* __restrict__ unpacked,
                                               int64_t N,
                                               int64_t num_groups,
                                               int64_t group_size,
                                               int64_t block_idx_y_base) {
  const int64_t no = blockIdx.y + block_idx_y_base;
  const int group_id = blockIdx.x;
  const int d = threadIdx.x;
  const int64_t global_thread_id = (no * num_groups + group_id) * group_size + d;

  int work_per_thread = 8 / bits;

  uint8_t local_packed = data[global_thread_id];
  int mask = ((1 << bits) - 1);
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni;

    if (boundary_check && n >= N) { break; }

    const int val = (local_packed >> (ni * bits)) & mask;
    const int64_t id = (n * num_groups + group_id) * group_size + d;
    unpacked[id] = ((scalar_t)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
  }
}

// Unpack int32 bit stream to float16/32 data
Tensor unpack_single_precision_cuda(Tensor data,
                                    int bits,
                                    Tensor scale,
                                    Tensor min,
                                    int64_t N,
                                    int64_t num_groups,
                                    int64_t group_size) {
  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);

  // Call unpack kernels
  int64_t logical_block_y_dim = (N + work_per_thread - 1) / work_per_thread;
  for (int64_t block_idx_y_base = 0; block_idx_y_base < logical_block_y_dim; block_idx_y_base += BLOCK_Y_DIM_MAX) {
    dim3 block_dim(num_groups, std::min(logical_block_y_dim - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
    dim3 thread_dim(group_size, 1, 1);
 
    if (N % work_per_thread == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
        unpack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          unpacked.data_ptr<scalar_t>(),
          N, num_groups, group_size, block_idx_y_base);
      }));
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
        unpack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim>>>(
          bits,
          data.data_ptr<int8_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          unpacked.data_ptr<scalar_t>(),
          N, num_groups, group_size, block_idx_y_base);
      }));
    }
  }

  return unpacked;
}

