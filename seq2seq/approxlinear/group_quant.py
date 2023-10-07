import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.approxlinear.mem_utils import *
import seq2seq.backend.softmax_quant.quantization as ext_quantization
import seq2seq.backend.softmax_quant.minimax as ext_minimax
# from mesa import custom_quant
# from mesa import native
# from mesa import packbit
#
# from .sparse_matrix import sparsify, unsparsify

from pdb import set_trace

MAX_THREAD = 256

@torch.no_grad()
def group_generation(input, group_size):
    N = input.shape[0]
    D = input.shape[1]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]
    num_pixels = num_features // D

    # Compute min, max by groups
    if num_features % group_size != 0:
        # Padding
        new_num_features = (num_features // group_size + 1) * group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, group_size)
    mn, mx = ext_minimax.minimax(input_groups)

    return input_groups.view(N, -1, group_size), mn.view(N, -1, 1), mx.view(N, -1, 1)


@torch.no_grad()
def quantize_and_pack(data, bits, mn, mx):

    # print(pack_func)
    # print(bits)

    output, scale = ext_quantization.pack_single_precision(data, mn, mx, bits, True)

    return output, scale

@torch.no_grad()
def dequantize_and_unpack(data, shape, bits, scale, mn, group_size):

    # Pad to group_size
    N = shape[0]
    num_features = int(shape[1:].numel())

    num_features = (num_features + (group_size - num_features % group_size) % group_size)

    # Unpack bitstream
    if isinstance(bits, int):
        unpack_func = ext_quantization.unpack_single_precision
    else:
        unpack_func = ext_quantization.unpack_mixed_precision

    data = unpack_func(data, bits, scale, mn, N, num_features // group_size, group_size)

    return data


@torch.no_grad()
def compression(x, scheme):

    Batch, Channel, Higth, Width = x.shape

    featuremap_area = Higth * Width # x_hfc.shape[-2:].numel()  # should be n
    if featuremap_area > scheme.group_size:
        group_size = scheme.group_size
        x_groups, q_min, mx = group_generation(x, group_size)

    else:
        group_size = featuremap_area
        x_groups = x.reshape(Batch, -1, group_size)
        q_min = x_groups.min(dim=-1).values.unsqueeze(dim=-1)
        mx = x_groups.max(dim=-1).values.unsqueeze(dim=-1)

    q_input, q_scale = quantize_and_pack(x_groups, scheme.q_bit, q_min, mx)

    return q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)


@torch.no_grad()
def de_compression(feature_pack, q_input_shape, scheme):

        # Batch, Channel, Higth, Width = q_input_shape

        q_input, q_scale, q_min = feature_pack

        # Estimate valid group size
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)

        # Estimate valid group size
        featuremap_area = q_input_shape[-2:].numel()
        group_size = scheme.group_size if featuremap_area > scheme.group_size else featuremap_area
        x_dequant = dequantize_and_unpack(q_input, q_input_shape, scheme.q_bit, q_scale, q_min, group_size)

        # Remove padding
        num_features = q_input_shape[1:].numel()
        x_dequant = x_dequant.view(q_input_shape[0], -1)[:, :num_features]
        x_dequant = x_dequant.view(*q_input_shape).contiguous()

        return x_dequant

