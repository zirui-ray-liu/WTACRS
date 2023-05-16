/*
 * Cuda operators for quantization and packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

#include "ext_common.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;


// ActQuantizedReLU
std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data);
Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask);

// ActQuantizedDropout
std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float p);
Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float p1m);


// Activation quantized relu: use compressed bit stream to store activation
class ActQuantizedReLU : public Function<ActQuantizedReLU> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {act_quantized_relu_backward_cuda(grad_outputs[0], saved[0])};
  }
};

Tensor act_quantized_relu(Tensor input) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedReLU::apply(input);
}

// Activiation quantized dropout: use compressed bit stream to store masks
class ActQuantizedDropout : public Function<ActQuantizedDropout> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, float p, bool train) {
    Tensor output, mask;
    if (!train){
      return input;
    }
    std::tie(output, mask) = act_quantized_dropout_forward_cuda(input, p);
    ctx->save_for_backward({mask});
    ctx->saved_data["p1m"] = 1. - p;
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    float p1m = ctx->saved_data["p1m"].toDouble();
    return {act_quantized_dropout_backward_cuda(grad_outputs[0], saved[0], p1m), Tensor(), Tensor()};
  }
};

Tensor act_quantized_dropout(Tensor input, float p, bool train) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedDropout::apply(input, p, train);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("act_quantized_relu", &act_quantized_relu);
  m.def("act_quantized_dropout", &act_quantized_dropout);
}