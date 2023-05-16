import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.approxlinear.mem_utils import *
# import seq2seq.backend.softmax_quant.quantization as ext_quantization
# from seq2seq.approxlinear.group_quant import compression as compression_group
# from seq2seq.approxlinear.group_quant import de_compression as de_compression_group
# from seq2seq.approxlinear.channel_quant import compression as compression_channel
# from seq2seq.approxlinear.channel_quant import de_compression as de_compression_channel
from seq2seq.backend.softmax_sparsify.sparse_matrix import sparsify, unsparsify
from seq2seq.backend.softmax_sparsify.masker import Masker

from pdb import set_trace

class QSoftmax_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, masker):

        # print("MEM.     Usage: %.2f MB before softmax" % (get_memory_usage() / 2 ** 20))

        y = torch.softmax(x, dim)
        # print("MEM.     Usage: %.2f MB before compression" % (get_memory_usage() / 2 ** 20))

        compression_func = sparsify
        shape_y, mask_y, sparse_y = compression_func(y, masker(y))

        # if config.amp:
        #     sparse_y = sparse_y.half()

        ctx.y_pack = shape_y, mask_y, sparse_y
        ctx.dim = dim

        # print("MEM.     Usage: %.2f MB after compression" % (get_memory_usage() / 2 ** 20))
        # print(y)

        # act_mem_fp = compute_tensor_bytes(y)
        # print("Act FP.     Usage: %.2f MB" % (act_mem_fp / 2 ** 20))
        # act_mem = compute_tensor_bytes(ctx.y_pack)
        # print("Act.     Usage: %.2f MB" % (act_mem / 2 ** 20))

        return y # .detach()

    @staticmethod
    def backward(ctx, grad_in):

        # torch.cuda.empty_cache()

        shape_y, mask_y, sparse_y = ctx.y_pack

        de_compression_func = unsparsify
        y = de_compression_func(shape_y, mask_y, sparse_y)
        # y = ctx.y

        if ctx.dim == -1:
            grad_out = grad_in * y - ((grad_in.unsqueeze(-2) @ y.unsqueeze(-1)) @ y.unsqueeze(-2)).squeeze(-2)

        else:
            raise NotImplementedError

        # if y.is_cuda:
        #     grad_out = native.softmax_backward_cuda(grad_in, y, ctx.dim, y)
        # else:
        #     grad_out = native.softmax_backward_cpu(grad_in, y, ctx.dim, y)

        del y, ctx.y_pack, ctx.dim
        # del ctx.y

        return grad_out, None, None


class ApproxSoftmax(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.func = QSoftmax_func.apply
        # self.scheme = Scheme_softmax(q_bit=config.softmax_qbit)
        self.masker = Masker(prune_ratio=config.softmax_prune_ratio)
        self.tag = 'softmax'

    def forward(self, x, dim):

        if self.training:
            return self.func(x, dim, self.masker)
        else:
            return torch.softmax(x, dim)