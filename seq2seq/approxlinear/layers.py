import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch import device, topk
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from .functional import approxlinear_only_bw, approxmatmul_4D_only_bw
from seq2seq.approxlinear.scheme import Scheme_3D_new, Scheme_4D_new

class ApproxLinear(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=True, config=None, batch_dim_use_same_indices=False):
        super(ApproxLinear, self).__init__(input_features, output_features, bias)
        self.batch_dim_use_same_indices = batch_dim_use_same_indices
        self.func = approxlinear_only_bw.apply

        # The scheme has some problems
        # assert len(config.tasks) == 1 and TASK_TO_NUM_INSTANCE[config.tasks[0]] is not None
        # Scheme.num_samples = TASK_TO_NUM_INSTANCE[config.tasks[0]]

        self.scheme = Scheme_3D_new(config.sampling_ratio, config.deter_ratio, config.deter_adaptive, config.minimal_k,
                                config.sample_replacement, config.mix_replacement, self.batch_dim_use_same_indices, config.random_sampling)

    def forward(self, input):
        if not self.scheme.inited:
            self.scheme.scheme_init(input.shape)
            
        if self.training:
             return self.func(input, self.weight, self.bias, self.scheme)
        else:
            return super(ApproxLinear, self).forward(input)


class Approxmatmul_4D(torch.nn.Module):
    def __init__(self, config=None, batch_dim_use_same_indices=True):
        super(Approxmatmul_4D, self).__init__()
        self.batch_dim_use_same_indices = batch_dim_use_same_indices
        self.func = approxmatmul_4D_only_bw.apply
        # self.func = approxmatmul_4D_fw_and_bw.apply

        # self.scheme = Scheme_4D(config.sampling_ratio, config.deter_ratio, config.deter_adaptive, config.minimal_k,
        #                         config.sample_replacement, config.mix_replacement, self.batch_dim_use_same_indices)
        
        self.scheme = Scheme_4D_new(config.sampling_ratio, config.deter_ratio, config.deter_adaptive, config.minimal_k,
                                config.sample_replacement, config.mix_replacement, self.batch_dim_use_same_indices, config.random_sampling)

    def forward(self, A, B):

        if not self.scheme.inited:
            b, c, m, _ = A.shape
            _, _, _, n = B.shape
            self.scheme.scheme_init((b, c, m, n))

        if self.training:
             return self.func(A, B, self.scheme)
        else:
            return torch.matmul(A, B)

    # def __str__(self):
    #     return "Approxmatmul_4D"