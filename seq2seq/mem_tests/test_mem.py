import math
import pdb
import json
import numpy as np
import torch
from torch.nn import functional as F
from utils import get_memory_usage

import sys
sys.path.append("../../")

from seq2seq.third_party.models.t5.modeling_t5 import T5Attention, T5Block
from seq2seq.third_party.models.t5.configuration_t5 import T5Config
from seq2seq.approxlinear import ApproxLinearConfig
from seq2seq.approxlinear.layers import Scheme
from torch.profiler import profile, record_function, ProfilerActivity

import os


def test_attention_mem():
    config = T5Config()
    with open('./baseline_config.json', 'r') as f:
        bl_config_dict = json.load(f)

    # load the baseline model config for simulation
    config.update(bl_config_dict)
    config.level = 1
    att_module = T5Block(config=config, has_relative_attention_bias=False, 
                             adapter_config=None, lora_config=None, approx_config=None)
    print(att_module)
    approx_config = ApproxLinearConfig()
    approx_config.only_bw = True
    approx_config.sampling_ratio = 0.1
    approx_config.k_sampling = False
    approx_config.q_sampling = False
    approx_config.v_sampling = False
    approx_config.o_sampling = True
    approx_config.wi_0_sampling = True
    approx_config.wi_1_sampling = True
    approx_config.wo_sampling = True
    approx_config.score_sampling = True
    approx_config.attout_sampling = True
    approx_config.deter_ratio = 0.5
    approx_config.sample_replacement = True
    approx_config.tasks = ["sst2"]
    att_module_with_approx = T5Block(config=config, has_relative_attention_bias=False, 
                                         adapter_config=None, lora_config=None, approx_config=approx_config)

    att_module = att_module.cuda()
    att_module_with_approx = att_module_with_approx.cuda()
    # pdb.set_trace()
    B, L, D = 128, 128, config.d_model
    print("========== Attention Memory Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(B, L, D).astype(dtype)
        Scheme.batch = torch.arange(B)

        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            before = get_memory_usage()

            for i in range(12):
                if isinstance(data, tuple):
                    data = data[0]
                data = func(data)

            after = get_memory_usage()
            
            return after - before

        usage_ref = test_implementation(att_module)
        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        usage_us = test_implementation(att_module_with_approx)

        print("Approx. Usage: %.2f MB" % (usage_us / 2 ** 20))


def attention_mem_profile():
    config = T5Config()
    with open('./baseline_config.json', 'r') as f:
        bl_config_dict = json.load(f)

    # load the baseline model config for simulation
    config.update(bl_config_dict)
    att_module = T5Attention(config=config, has_relative_attention_bias=False, 
                             adapter_config=None, lora_config=None, approx_config=None)
    att_module.train()
    approx_config = ApproxLinearConfig()
    approx_config.only_bw = True
    approx_config.sampling_ratio = 0.1
    approx_config.k_sampling = True
    approx_config.q_sampling = True          
    approx_config.v_sampling = True
    approx_config.o_sampling = True
    approx_config.wi_0_sampling = True
    approx_config.wi_1_sampling = True
    approx_config.wo_sampling = True
    approx_config.score_sampling = True   
    approx_config.attout_sampling = True
    att_module_with_approx = T5Attention(config=config, has_relative_attention_bias=False,
                                         adapter_config=None, lora_config=None, approx_config=approx_config)

    att_module_with_approx.train()
    B, L, D = 128, 128, config.d_model
    # pdb.set_trace()
    print("========== Attention Memory Test ==========")

    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(B, L, D).astype(dtype)

        def test_implementation(model):
            data = torch.tensor(data_np).requires_grad_()
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
                with record_function("model_train"):
                    model(data)
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=100))
            prof.export_stacks('./attention_mem_profile.txt', "self_cpu_time_total")


        test_implementation(att_module)
        test_implementation(att_module_with_approx)


if __name__ == "__main__":
    # test_gelu_correctness()
    # test_gelu_memory()
    # attention_mem_profile()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    test_attention_mem()