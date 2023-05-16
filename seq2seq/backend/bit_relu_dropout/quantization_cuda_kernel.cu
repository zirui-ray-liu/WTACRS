// our code is based-off the official codes of ActNN and BLPA.
// ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training.
// Backprop with Approximate Activations for Memory-efficient Network Training.

// Here we summarize the common parts and the difference.
// 1. ActNN and BLPA hack the byte format in a same way to maximize the memory saving. 
//    We just follow their design principles.

// 2. To maximize the memory saving, we must provide a quantized version for 
//    all of the non-linear operations, e.g., ReLU and LeakyReLU. 
//    Please see Appendix H.2 for a detailed explanation.
//    This part is usually got ignored in BLPA and many previous works. We note that although
//    theorectically the mask matrix in ReLU only take one bit per element, however,
//    Pytorch actually cannot go below one byte due to some engineering trade-off.
//    Please see https://github.com/pytorch/pytorch/issues/41571 for a detailed explanation.
//    To maximize the memory saving, we need to hack the byte format for the mask matrix to achieve
//    a space complexity of one-bit per element.
//    We found that ActNN's quantized ReLU is near-optimal. 
//    Here we just utilize their implementation,
//    and extend it for LeakyReLU and ELU, which is commonly used in GAT.

// 3. For the dropout function, similar to the ReLU and LeakyReLU function, we need to provide
//    a quantized version for the Dropout function. We provide our implementation. We note that
//    ActNN also has one quantized dropout function, however, it runs too slow compared to Pytorch's
//    official dropout function and ours. We note that the run time speed of our quantized dropout 
//    is near-optimal.

// 4. We write some glue codes in C++ such that we can insert the quantization & random projection 
//    in the forward pass & backward pass of common OPs. See backward_func.cc and backward_func_cuda_kernel.cu

#include <stdio.h>
#include <torch/extension.h>
// #include <ATen/CUDAGeneratorImpl.h>

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


#define BLOCK_Y_DIM_MAX ((1l << 16) - 1)

using torch::IntArrayRef;
using torch::Tensor;



/****************************************/
/********** Act Quantized ReLU **********/
/****************************************/
#define ACT_QUANTIZED_RELU_NUM_THREADS 512
// Unpack int32 bit stream to float16/32 data
template <typename scalar_t>
__global__ void act_quantized_relu_forward_kernel(const scalar_t* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  scalar_t* __restrict__ output,
                                                  int64_t N,
                                                  int64_t mask_len) {
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);
  __shared__ int mask_shared[ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8)];

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask_shared)[threadIdx.x] = make_int2(0, 0);
  }

  if (id < N) {
    bool bit = data[id] > 0;
    if (bit) {
      output[id] = data[id];
    } else {
      output[id] = 0.0;
    }

    __syncthreads();
    atomicOr(mask_shared + threadIdx.x % shared_len, bit << (threadIdx.x / shared_len));
    __syncthreads();
  }

  if (threadIdx.x * 2 < shared_len) {
    reinterpret_cast<int2*>(mask)[global_offset / 2 + threadIdx.x] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x];
  }
}

std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int64_t mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "act_quantized_relu_forward", ([&] {
    act_quantized_relu_forward_kernel<scalar_t><<<blocks, threads>>>(
      data.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
      n_elements, mask_len);
  }));

  return std::make_pair(output, mask);
}

template <typename scalar_t>
__global__ void act_quantized_relu_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ grad_input,
                                                   int N) {
  int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t global_offset = (int64_t)blockIdx.x * blockDim.x / (sizeof(int32_t) * 8);
  const int shared_len = ACT_QUANTIZED_RELU_NUM_THREADS / (sizeof(int32_t) * 8);

  if (id < N) {
    bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
    if (bit) {
      grad_input[id] = grad_output[id];
    } else {
      grad_input[id] = 0.0;
    }
  }
}


Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  int threads = ACT_QUANTIZED_RELU_NUM_THREADS;
  int blocks = (n_elements + threads - 1) / threads;

  Tensor grad_input = torch::empty_like(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_relu_backward", ([&] {
      act_quantized_relu_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<scalar_t>(),
        n_elements);
  }));

  return grad_input;
}


/****************************************/
/********** Act Quantized Dropout *******/
/****************************************/
#define ACT_QUANTIZED_DROPOUT_NUM_THREADS 512
#define UNROLL 4
template <typename scalar_t, int ADims, int BDims=ADims>
__global__ void act_quantized_dropout_forward_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> a,
                                                  int32_t* __restrict__ mask,
                                                  at::cuda::detail::TensorInfo<scalar_t, int64_t> b,
                                                  std::pair<uint64_t, uint64_t> seeds,
                                                  int64_t N,
                                                  int64_t mask_len,
                                                  float p) {
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int shared_len = ACT_QUANTIZED_DROPOUT_NUM_THREADS / (sizeof(int32_t) * 8);
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, id, seeds.second, &state);
  const int64_t rounded_size = ((N - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  float pinv = 1. / (1. - p);
  __shared__ int mask_shared[shared_len*UNROLL];
  for (int64_t linearIndex = id;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
       const int64_t global_offset = (int64_t)(blockIdx.x * blockDim.x + linearIndex - id) / (sizeof(int32_t) * 8);
       int64_t local_offset = (int64_t)blockDim.x * gridDim.x / (sizeof(int32_t) * 8);
      //curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for halfs, so generate float for everything
       float4 rand = curand_uniform4(&state);
       scalar_t src[UNROLL];
       bool inrange[UNROLL] = {0};
       rand.x = rand.x > p;
       rand.y = rand.y > p;
       rand.z = rand.z > p;
       rand.w = rand.w > p;
       if (threadIdx.x * 2 < shared_len) {
        for (int ii = 0; ii < UNROLL; ii++) {
          reinterpret_cast<int2*>(mask_shared)[threadIdx.x+ii*shared_len/2] = make_int2(0, 0);}
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           int64_t li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < N) {
              // Convert `linearIndex` into an offset of `a`
               int64_t aOffset =
                   at::cuda::detail::IndexToOffset<scalar_t, int64_t, ADims>::get(li, a);
               src[ii] = a.data[aOffset];
               inrange[ii] = 1;
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           int64_t li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < N) {
              // Convert `linearIndex` into an offset of `b`
               const int64_t bOffset =
                   at::cuda::detail::IndexToOffset<scalar_t, int64_t, BDims>::get(li, b);
               b.data[bOffset] = src[ii]*(&rand.x)[ii]*pinv;
           }
       }
       __syncthreads();
      for (int ii = 0; ii < UNROLL; ii++) {
        bool bit = (&rand.x)[ii];
        if (inrange[ii]){
          atomicOr(mask_shared+ii*shared_len+threadIdx.x%shared_len, bit << (threadIdx.x/shared_len));}
      }
       __syncthreads();
      
      if (threadIdx.x * 2 < shared_len) {
        for (int ii = 0; ii < UNROLL; ii++){
          if (inrange[ii]){
            reinterpret_cast<int2*>(mask)[threadIdx.x+global_offset/2+ii*local_offset/2] = reinterpret_cast<int2*>(mask_shared)[threadIdx.x+ii*shared_len/2];}
          }
      }
  }
}

std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float p) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int64_t mask_len = (n_elements + sizeof(int32_t) * 8 - 1) / (sizeof(int32_t) * 8);
  Tensor mask = torch::empty({mask_len}, options);
  Tensor output = torch::empty_like(data);

  int64_t block_size = ACT_QUANTIZED_DROPOUT_NUM_THREADS;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((n_elements + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  int64_t counter_offset = ((n_elements - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }
                            
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "act_quantized_dropout_forward", ([&] {
    auto data_info =
      at::cuda::detail::getTensorInfo<scalar_t, int64_t>(data);
    auto output_info =
      at::cuda::detail::getTensorInfo<scalar_t, int64_t>(output);
    data_info.collapseDims();
    output_info.collapseDims();
    act_quantized_dropout_forward_kernel<scalar_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
      data_info, mask.data_ptr<int32_t>(), output_info, rng_engine_inputs,
      n_elements, mask_len, p);
  }));

  return std::make_pair(output, mask);
}


template <typename scalar_t>
__global__ void act_quantized_dropout_backward_kernel(const scalar_t* __restrict__ grad_output,
                                                   int32_t* __restrict__ mask,
                                                   scalar_t* __restrict__ grad_input,
                                                   int N,
                                                   float p1m) {
  int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int shared_len = ACT_QUANTIZED_DROPOUT_NUM_THREADS / (sizeof(int32_t) * 8);

   for (int64_t linearIndex = id;
       linearIndex < N;
       linearIndex += gridDim.x * blockDim.x) {
         const int64_t global_offset = (int64_t)(blockIdx.x * blockDim.x+linearIndex-id) / (sizeof(int32_t) * 8);
         bool bit =  (mask[global_offset + threadIdx.x % shared_len] >> (threadIdx.x / shared_len)) & 1;
         if (bit){
           grad_input[linearIndex] = grad_output[linearIndex] / p1m;
         }else{
           grad_input[linearIndex] = 0.0;
         }
  }
}


Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float p1m) {
  int64_t n_elements = 1;
  for (size_t i = 0; i < grad_output.dim(); ++i) {
    n_elements *= grad_output.size(i);
  }

  Tensor grad_input = torch::empty_like(grad_output);
  int64_t block_size = ACT_QUANTIZED_DROPOUT_NUM_THREADS;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((n_elements + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "act_quantized_dropout_backward", ([&] {
      act_quantized_dropout_backward_kernel<scalar_t><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), grad_input.data_ptr<scalar_t>(),
        n_elements, p1m);
  }));
  

  return grad_input;
}