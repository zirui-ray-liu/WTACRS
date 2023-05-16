import torch
import numbers
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import seq2seq.backend.inplace_layernorm as inplace_layernorm

class InplaceT5LayerNormFunction(autograd.Function):
    '''
    LayerNorm Function using the output to compute gradient
    '''
    @staticmethod
    def forward(ctx, hidden_states, normalized_shape, weight, bias, eps):
        # FIXME: at::native_layer_norm's output rstd is always float
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        hidden_states = hidden_states * rstd
        # convert into float16 if necessary
        if weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        result = weight * hidden_states
        if bias is not None:
            result  = result + bias 
        ctx.save_for_backward(result, rstd, weight, bias)
        ctx.normalized_shape = normalized_shape
        return result

    @staticmethod
    def backward(ctx, grad_output):
        output, rstd, weight, bias = ctx.saved_tensors
        if bias is not None:
            grad_mask = [True, True, True]
        else:
            grad_mask = [True, True, False]

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        dx, dweight, dbias = inplace_layernorm.backward(grad_output, output, ctx.normalized_shape,
                                                        rstd, weight, bias, grad_mask)
    
        return dx, None, dweight, dbias, None



class InplaceT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, adapter_config=None):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bitfit = adapter_config.bitfit if adapter_config is not None else False 
        if self.bitfit:
           self.bias = nn.Parameter(torch.zeros(hidden_size)) 
        if isinstance(hidden_size, numbers.Integral):
            # mypy error: incompatible types in assignment
            hidden_size = (hidden_size,)  # type: ignore[assignment]
        self.normalized_shape = tuple(hidden_size)  # type: ignore[arg-type]


    def forward(self, hidden_states):
        return InplaceT5LayerNormFunction.apply(hidden_states, 
                                                self.normalized_shape,
                                                self.weight, 
                                                self.bias if self.bitfit else None, 
                                                self.variance_epsilon)

class InplaceLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super(InplaceLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return InplaceLayerNormFunction.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)
    
    
class InplaceLayerNormFunction(autograd.Function):
    '''
    LayerNorm Function using the output to compute gradient
    '''
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps=1e-6):
        output, _, rstd = inplace_layernorm.forward(input, normalized_shape, weight, bias, eps)
        # FIXME: at::native_layer_norm's output rstd is always float
        if output.dtype == torch.half and rstd.dtype == torch.float:
            rstd = rstd.half()
        ctx.save_for_backward(output, rstd, weight, bias)
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, rstd, weight, bias = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        grad_mask = [True, True, True]

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        dx, dweight, dbias = inplace_layernorm.backward(grad_output, output, normalized_shape, rstd, weight, bias, grad_mask)

        return dx, None, dweight, dbias, None