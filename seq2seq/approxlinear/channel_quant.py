import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.approxlinear.mem_utils import *
import seq2seq.backend.softmax_quant.quantization as ext_quantization
# from mesa import custom_quant
# from mesa import native
# from mesa import packbit
#
# from .sparse_matrix import sparsify, unsparsify

from pdb import set_trace

MAX_THREAD = 256

@torch.no_grad()
def quantize_and_pack(data, bits, mn, mx, N):

    # scale_ = (2 ** bits - 1) / (mx - mn)
    mn_ = mn.view(N, 1, 1).repeat(1, data.shape[1], 1)
    mx_ = mx.view(N, 1, 1).repeat(1, data.shape[1], 1)

    # output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
    output, scale = ext_quantization.pack_single_precision(data, mn_, mx_, bits, True)

    # scale_ = scale[:,0,0].clone()
    # print(scale)
    scale_ = scale.mean(dim=1).squeeze(dim=-1)
    # print(scale_)

    return output, scale_

@torch.no_grad()
def dequantize_and_unpack(data, shape, bits, scale, mn):

    # Pad to group_size
    Batch, Channel, Higth, Width = shape
    num_features = int(shape[2:].numel())

    if num_features > MAX_THREAD:
        mn_ = mn.view(Batch * Channel, 1, 1).repeat(1, Higth, 1)
        scale_ = scale.view(Batch * Channel, 1, 1).repeat(1, Higth, 1) # N, num_features // group_size, group_size)
        data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, Higth, Width)

    else:
        mn_ = mn.view(Batch * Channel, 1, 1)
        scale_ = scale.view(Batch * Channel, 1, 1)
        data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, 1, Higth * Width)

    return data


@torch.no_grad()
def compression(x, scheme):

    Batch, Channel, Higth, Width = x.shape

    featuremap_area = Higth * Width # x_hfc.shape[-2:].numel()  # should be n

    if featuremap_area > MAX_THREAD:
        x_groups = x.reshape(Batch * Channel, Higth, Width)
    else:
        x_groups = x.reshape(Batch * Channel, 1, Higth * Width)

    q_min = x_groups.min(dim=-1).values.min(dim=-1).values
    mx = x_groups.max(dim=-1).values.max(dim=-1).values
    q_input, q_scale = quantize_and_pack(x_groups, scheme.q_bit, q_min, mx, Batch * Channel)

    return q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)


@torch.no_grad()
def de_compression(feature_pack, q_input_shape, scheme):

        # Batch, Channel, Higth, Width = q_input_shape

        q_input, q_scale, q_min = feature_pack

        # Estimate valid group size
        q_scale = q_scale.to(torch.float32)
        q_min = q_min.to(torch.float32)

        x_dequant = dequantize_and_unpack(q_input, q_input_shape, scheme.q_bit, q_scale, q_min)
        x_dequant = x_dequant.view(*q_input_shape).contiguous()

        return x_dequant

