import math
import pdb
from copy import deepcopy
import json
import numpy as np
import torch
from torch.nn import functional as F
from utils import get_memory_usage
from seq2seq.third_party.models.t5.configuration_t5 import T5Config
from seq2seq.third_party.models.t5.modeling_t5 import T5Attention, T5Block

import sys
sys.path.append("../../")

from seq2seq.third_party.models.t5.modeling_t5 import T5LayerNorm
from seq2seq.inplace_layernorm import InplaceT5LayerNorm, InplaceLayerNorm
from torch.nn import LayerNorm
from seq2seq.approxlinear import ApproxLinearConfig
from seq2seq.approxlinear.layers import Scheme
import os


def test_layernorm_correctness():
    print("========== Layernorm Correctness Test ==========")
    N, L, D = 128, 128, 768
    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.rand(N, L, D).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        ori_layernorm = T5LayerNorm(D).cuda()
        inplace_layernorm = InplaceT5LayerNorm(D).cuda()
        output_ref, grad_data_ref =  test_implementation(ori_layernorm)
        output_us, grad_data_us = test_implementation(inplace_layernorm)

        np.testing.assert_allclose(output_ref, output_us, rtol=1e-3)
        np.testing.assert_allclose(ori_layernorm.weight.grad.cpu().numpy(), 
                                   inplace_layernorm.weight.grad.cpu().numpy(), 
                                   rtol=1e-1)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, atol=1e-5)


def test_t5block_correctness():
    config = T5Config()
    with open('./baseline_config.json', 'r') as f:
        bl_config_dict = json.load(f)

    # load the baseline model config for simulation
    config.update(bl_config_dict)
    config.level = 0
    config.dropout_rate = 0.0
    block_level_0 = T5Block(config=config, has_relative_attention_bias=False, 
                             adapter_config=None, lora_config=None, approx_config=None)
    print(block_level_0)
    config_level_1 = deepcopy(config)
    config_level_1.level = 1
    block_level_1 = T5Block(config=config_level_1, has_relative_attention_bias=False, 
                             adapter_config=None, lora_config=None, approx_config=None)
    
    block_level_1.load_state_dict(block_level_0.state_dict())

    print(block_level_1)
    block_level_0 = block_level_0.cuda()
    block_level_0.train()
    block_level_1 = block_level_1.cuda()
    block_level_1.train()
    # pdb.set_trace()
    B, L, D = 4, 4, config.d_model
    print("========== Attention Memory Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(B, L, D).astype(dtype)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            output = func(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output[0]
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]
        

        output_ref, grad_data_ref =  test_implementation(block_level_0)
        output_us, grad_data_us = test_implementation(block_level_1)
        np.testing.assert_allclose(output_ref, output_us, rtol=1e-3)
        np.testing.assert_allclose(grad_data_ref, grad_data_us, rtol=1e-3)


if __name__ == "__main__":
    # test_gelu_correctness()
    # test_gelu_memory()
    # attention_mem_profile()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_layernorm_correctness()
    test_t5block_correctness()