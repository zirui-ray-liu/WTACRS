import math
import pdb
import json
import numpy as np
import torch
from torch.nn import functional as F
# from utils import get_memory_usage
import os

import sys
sys.path.append("../")

from seq2seq.approxlinear import ApproxLinearConfig, QSoftmax
from seq2seq.approxlinear.layers import Scheme


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

approx_config = ApproxLinearConfig()
approx_config.softmax_qbit = 4
softmax_op_channel = QSoftmax(config=approx_config)
softmax_op_channel.scheme.mode = "channel"
softmax_op_group = QSoftmax(config=approx_config)
softmax_op_group.scheme.mode = "group"

B, L, H, W = 2, 2, 32, 32

data_np = np.random.randn(B, L, H, W).astype("float32")
weight = np.random.randn(B, L, H, W).astype("float32")
weight_tensor = torch.tensor(weight).to("cuda")

x1 = torch.tensor(data_np).to("cuda").clone().requires_grad_()
x2 = torch.tensor(data_np).to("cuda").clone().requires_grad_()
x3 = torch.tensor(data_np).to("cuda").clone().requires_grad_()
# x3 = torch.tensor([2., 1., 0., -1.]).requires_grad_()
# print(x3)

y1 = softmax_op_channel(x1, dim=-1)
(y1 * weight_tensor).sum().backward()

y2 = torch.softmax(x2, dim=-1)
(y2 * weight_tensor).sum().backward()

y3 = softmax_op_group(x3, dim=-1)
(y3 * weight_tensor).sum().backward()

# print(x1.grad[0, 1,])
# print(x2.grad[0, 1,])

error_channel = torch.abs(x1.grad - x2.grad)
error_group = torch.abs(x3.grad - x2.grad)

print(x2.grad.abs().mean(), x2.grad.abs().median(), x2.grad.abs().max(), x2.grad.abs().min())
print(error_channel.mean(), error_channel.median(), error_channel.max(), error_channel.min())
print(error_group.mean(), error_group.median(), error_group.max(), error_group.min())

# y3 = torch.softmax(x3, dim=-1)
# gradin = torch.tensor([4., 3., 2., 1.])
# (y3 * gradin).sum().backward()
# print(x3.grad)
#
#
# gradout = (gradin * y3) - ((gradin * y3).unsqueeze(-1) @ y3.unsqueeze(-2)).sum(dim=-2)
# gradout2 = gradin * y3 - (y3.unsqueeze(-1) @ (gradin.unsqueeze(-2)) @ y3.unsqueeze(-1)).squeeze(-1)
#
# # print(gradin.unsqueeze(-1))
# # print(y3.unsqueeze(-1))
# # print(gradin * y3)
# # print(torch.matmul((gradin * y3).unsqueeze(-1), y3.unsqueeze(-2)))
# # print(torch.matmul((gradin * y3).unsqueeze(-1), y3.unsqueeze(-2)).sum(-2))
# print(gradout)
# print(gradout2)