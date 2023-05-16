import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch import device, topk


class Scheme(object):

    num_samples = 1
    batch = None

    _all_grad_A_shape = []
    _all_grad_B_shape = []
    _grad_A_index_offset = 0
    _grad_B_index_offset = 0 
    batch_whole_grad_A = None
    batch_whole_grad_B = None

    _all_grad_shape = []
    _grad_index_offset = 0
    batch_whole_grad = None
    _whole_grad_buffer_cat = None

    all_grad_A_size = 0
    all_grad_B_size = 0
    all_grad_size = 0
    
    scheme_open = False

    def __init__(self, sample_ratio, deter_ratio, deter_adaptive, minimal_k, sample_replacement, mix_replacement, batch_dim_use_same_indices, uniform=False):
        self.epoch = 0
        self.grad_updated = False
        self.inited = False
        self.sample_ratio = sample_ratio  
        self.deter_ratio = deter_ratio
        self.deter_adaptive = deter_adaptive
        self.minimal_k = minimal_k
        self.sample_replacement = sample_replacement
        self.mix_replacement = mix_replacement
        self.batch_dim_use_same_indices = batch_dim_use_same_indices
        self.q_bit = 2
        self.uniform = uniform
        Scheme.scheme_open = True

    def scheme_init(self, mat_shape, device):

        return NotImplementedError

    def get_scale(self):

        return NotImplementedError

    def set_scale(self, grad):

        return NotImplementedError

    @classmethod
    def _init_buffer(cls):
        dtype = torch.bfloat16
        device = 'cpu'
        # import ipdb; ipdb.set_trace()
        cls.all_grad_A_size = np.sum([np.prod(shape) for shape in cls._all_grad_A_shape]).astype(np.int)
        cls.all_grad_B_size = np.sum([np.prod(shape) for shape in cls._all_grad_B_shape]).astype(np.int)
        cls.all_grad_size = np.sum([np.prod(shape) for shape in cls._all_grad_shape]).astype(np.int)
        cls._whole_grad_buffer_cat = torch.zeros((Scheme.num_samples, cls.all_grad_A_size + cls.all_grad_B_size + cls.all_grad_size), dtype=dtype, device=device)

    @classmethod    
    def step(cls):
        cls._whole_grad_buffer_cat[cls.batch] = torch.cat([cls.batch_whole_grad_A,
                                                           cls.batch_whole_grad_B,
                                                           cls.batch_whole_grad], dim=1).cpu()

    @classmethod
    def fetch_data(cls):

        _whole_grad_buffer_cat_cuda = cls._whole_grad_buffer_cat[cls.batch].cuda()
        # grad_A_size = np.sum([np.prod(shape) for shape in cls._all_grad_A_shape])
        # grad_B_size = np.sum([np.prod(shape) for shape in cls._all_grad_B_shape])
        # grad_size = np.sum([np.prod(shape) for shape in cls._all_grad_shape])
        #
        cls.batch_whole_grad_A = _whole_grad_buffer_cat_cuda[:, :cls.all_grad_A_size]
        cls.batch_whole_grad_B = _whole_grad_buffer_cat_cuda[:, cls.all_grad_A_size:cls.all_grad_A_size + cls.all_grad_B_size]
        cls.batch_whole_grad = _whole_grad_buffer_cat_cuda[:, cls.all_grad_A_size + cls.all_grad_B_size:]


class Scheme_3D_new(Scheme):

    def scheme_init(self, mat_shape):
        assert len(mat_shape) == 3
        _, c, _ = mat_shape
        self.grad_size = c
        self.grad_shape = (c,)
        Scheme._all_grad_shape.append(self.grad_shape)
        self.grad_index_offset = copy.deepcopy(Scheme._grad_index_offset)
        Scheme._grad_index_offset += self.grad_size
        self.inited = True

    def get_scale(self):
        if self._whole_grad_buffer_cat is None:  # self.grad_updated: #
            return None
        else:
            bs = Scheme.batch_whole_grad.size(0)
            grad_ = Scheme.batch_whole_grad[:, self.grad_index_offset:self.grad_index_offset + self.grad_size]
            self.grad_updated = False
            return grad_.view(bs, self.grad_size).contiguous()

    def set_scale(self, grad):
        grad_ = grad.view(-1, self.grad_size, grad.shape[-1]).norm(dim=2).bfloat16()
        Scheme.batch_whole_grad[:, self.grad_index_offset:self.grad_index_offset + self.grad_size] = grad_
        self.grad_updated = True



class Scheme_4D_new(Scheme):

    def scheme_init(self, mat_shape):
        assert len(mat_shape) == 4
        b, c, m, n = mat_shape
        self.grad_A_shape = (c, n)
        self.grad_B_shape = (c, m)
        self.grad_A_size = np.prod(self.grad_A_shape)
        self.grad_B_size = np.prod(self.grad_B_shape)
        Scheme._all_grad_A_shape.append(self.grad_A_shape)
        Scheme._all_grad_B_shape.append(self.grad_B_shape)
        self.grad_A_index_offset = copy.deepcopy(Scheme._grad_A_index_offset)
        self.grad_B_index_offset = copy.deepcopy(Scheme._grad_B_index_offset)
        Scheme._grad_A_index_offset += self.grad_A_size
        Scheme._grad_B_index_offset += self.grad_B_size
        self.inited = True


    def get_scale(self):
        if self._whole_grad_buffer_cat is None: 
            return None, None
        else:
            bs = Scheme.batch_whole_grad_A.size(0)
            grad_A = Scheme.batch_whole_grad_A[:, self.grad_A_index_offset:self.grad_A_index_offset + self.grad_A_size]
            grad_B = Scheme.batch_whole_grad_B[:, self.grad_B_index_offset:self.grad_B_index_offset + self.grad_B_size]
            self.grad_updated = False
            return grad_A.view(bs, *self.grad_A_shape), grad_B.view(bs, *self.grad_B_shape)
        

    def set_scale(self, grad):
        assert grad.ndim == 4
        # print("Grad Shape:", grad.shape)
        grad_A = grad.norm(dim=2).bfloat16() # self.grad_A[Scheme.batch] * 0.5 + grad.norm(dim=2) * 0.5
        grad_B = grad.norm(dim=3).bfloat16() # self.grad_B[Scheme.batch] * 0.5 + grad.norm(dim=3) * 0.5
        Scheme.batch_whole_grad_A[:, self.grad_A_index_offset:self.grad_A_index_offset + self.grad_A_size] = grad_A.view(-1, self.grad_A_size)
        Scheme.batch_whole_grad_B[:, self.grad_B_index_offset:self.grad_B_index_offset + self.grad_B_size] = grad_B.view(-1, self.grad_B_size)
        self.grad_updated = True


class Scheme_3D(Scheme):

    def scheme_init(self, mat_shape):
        assert len(mat_shape) == 3
        self.scales = torch.zeros((Scheme.num_samples, mat_shape[1]), dtype=torch.bfloat16, device='cpu')
        # self.scales = torch.zeros((Scheme.num_samples, mat_shape[1]), dtype=torch.bfloat16, device='cuda')
        self.inited = True

    def get_scale(self):
        if True:  # self.grad_updated:
            assert Scheme.batch is not None
            # scale = self.scales[Scheme.batch].clone()
            scale = self.scales[Scheme.batch]
            return scale.cuda().float()
        else:
            return None

    def set_scale(self, grad):
        assert Scheme.batch is not None
        # print("Grad Shape:", grad.shape)
        scale = grad.view(-1, self.scales.shape[1], grad.shape[-1]).norm(dim=-1)
        self.scales[Scheme.batch] = scale.bfloat16().cpu()
        # self.scales[Scheme.batch] = self.scales[Scheme.batch] * 0.5 + scale.bfloat16().cpu() * 0.5
        self.grad_updated = True


class Scheme_4D(Scheme):

    def scheme_init(self, mat_shape):
        assert len(mat_shape) == 4
        b, c, m, n = mat_shape
        self.grad_A = torch.zeros((Scheme.num_samples, c, n), dtype=torch.bfloat16, device='cpu')
        self.grad_B = torch.zeros((Scheme.num_samples, c, m), dtype=torch.bfloat16, device='cpu')
        # self.grad_A = torch.zeros((Scheme.num_samples, c, n), dtype=torch.bfloat16, device='cuda')
        # self.grad_B = torch.zeros((Scheme.num_samples, c, m), dtype=torch.bfloat16, device='cuda')
        self.inited = True

    def get_scale(self):
        if True:  # self.grad_updated: #
            assert Scheme.batch is not None
            grad_A = self.grad_A[Scheme.batch]
            grad_B = self.grad_B[Scheme.batch]
            self.grad_updated = False
            return grad_A.cuda(), grad_B.cuda()
        else:
            return None, None

    def set_scale(self, grad):
        assert grad.ndim == 4 and Scheme.batch is not None
        # print("Grad Shape:", grad.shape)
        self.grad_A[Scheme.batch] = grad.norm(
            dim=2).bfloat16().cpu()  # self.grad_A[Scheme.batch] * 0.5 + grad.norm(dim=2) * 0.5
        self.grad_B[Scheme.batch] = grad.norm(
            dim=3).bfloat16().cpu()  # self.grad_B[Scheme.batch] * 0.5 + grad.norm(dim=3) * 0.5
        # self.grad_A[Scheme.batch] = self.grad_A[Scheme.batch] * 0.5 + grad.norm(dim=2).bfloat16().cpu() * 0.5
        # self.grad_B[Scheme.batch] = self.grad_B[Scheme.batch] * 0.5 + grad.norm(dim=3).bfloat16().cpu() * 0.5
        self.grad_updated = True
