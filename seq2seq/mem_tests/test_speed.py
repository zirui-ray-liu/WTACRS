import ipdb
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from utils import get_memory_usage

import sys
sys.path.append("../../")

from seq2seq.third_party.models.t5.modeling_t5 import T5Attention, T5Block
from seq2seq.third_party.models.t5.configuration_t5 import T5Config
from seq2seq.approxlinear import ApproxLinearConfig
from seq2seq.approxlinear.scheme import Scheme_4D_new, Scheme_3D_new, Scheme
from torch.profiler import profile, record_function, ProfilerActivity
from seq2seq.approxlinear import ApproxLinear, Approxmatmul_4D
from timeit_v2 import py_benchmark
import torch.autograd.profiler as profiler


def test_matmul_speed():
    exact_matmul_op = torch.matmul

    approx_config = ApproxLinearConfig()
    approx_config.only_bw = True
    approx_config.sampling_ratio = 0.3
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
    approx_config.mix_replacement = True

    approx_config.tasks = ["sst2"]
    approx_matmul = Approxmatmul_4D(config=approx_config, batch_dim_use_same_indices=False)

    B, H, L, D = 128, 12, 128, 64
    Scheme.batch = torch.arange(B)
    A_np = np.random.randn(B, H, L, D).astype(np.float32)
    B_np = np.random.randn(B, H, D, L).astype(np.float32)

    # A_np = np.random.randn(B, H, L, L).astype(np.float32)
    # B_np = np.random.randn(B, H, L, D).astype(np.float32)

    A_tensor = torch.tensor(A_np).to("cuda").requires_grad_()
    B_tensor = torch.tensor(B_np).to("cuda").requires_grad_()

    print("========== Matmul Speed Test ==========")
    def test_implementation(model, A_tensor, B_tensor):
        stmt = "model(A_tensor, B_tensor)"
        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                    setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        output =  model(A_tensor, B_tensor)
        if isinstance(output, tuple):
            output = output[0]
        head = torch.randn_like(output, dtype=torch.float)
        stmt = "output.backward(head, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                    setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
        return t_forward, t_backward
    
    forward_ref, backward_ref = test_implementation(exact_matmul_op, A_tensor, B_tensor)
    forward_us, backward_us = test_implementation(approx_matmul, A_tensor, B_tensor)

    print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))

    print("approx.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
            (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))


def vis_matmul_speed():
    exact_matmul_op = torch.matmul

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
    approx_config.sample_replacement = False
    approx_config.mix_replacement = False
    approx_config.random_sampling = False
    approx_config.tasks = ["sst2"]
    approx_matmul = Approxmatmul_4D(config=approx_config, batch_dim_use_same_indices=False)

    B, H, L, D = 128, 12, 128, 64
    Scheme.batch = torch.arange(B)
    A_np = np.random.randn(B, H, L, D).astype(np.float32)
    B_np = np.random.randn(B, H, D, L).astype(np.float32)

    A_tensor = torch.tensor(A_np).to("cuda").requires_grad_()
    B_tensor = torch.tensor(B_np).to("cuda").requires_grad_()
    device = "cuda"
    print("========== Vis Matmul Speed ==========")

    def test_implementation(func, A_tensor, B_tensor, name):
        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./test_matmul_speed', worker_name=name),
                record_shapes=True,
                with_stack=True)
        prof.start()
        for step in range(100):
            if step >= (1 + 1 + 3) * 2:
                break
            with record_function("forward"):
                loss = func(A_tensor, B_tensor).sum()
            with record_function("backward"):
                loss.backward()
            prof.step()
        prof.stop()

    test_implementation(exact_matmul_op, A_tensor, B_tensor, "exact")
    test_implementation(approx_matmul, A_tensor, B_tensor, "approx")                


def test_io_speed():    
    def init_scheme(cls):
        return cls(sample_ratio=0.1, 
                       deter_ratio=0.5, 
                       deter_adaptive=True,
                       minimal_k=10,
                       sample_replacement=True,
                       mix_replacement=True,
                       batch_dim_use_same_indices=False)
    N, L = 5000, 12
    b, h, l, d = 128, 12, 128, 64
    mat_shape = (b, h, l, l)
    Scheme.num_samples = N
    
    all_schemes, all_schemes_new = [], []
    for i in range(L):
        scheme = init_scheme(Scheme_4D)
        scheme.scheme_init(mat_shape)
        all_schemes.append(scheme)

        scheme_new = init_scheme(Scheme_4D_new)
        scheme_new.scheme_init(mat_shape)
        all_schemes_new.append(scheme_new)

    Scheme._init_buffer()
    assert Scheme._whole_grad_A_buffer is not None and Scheme._whole_grad_B_buffer is not None

    def test_implementation(schemes, name, cat='baseline'):
        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./test_matmul_speed', worker_name=name),
                record_shapes=True,
                with_stack=True)
        prof.start()
        for step in range(100):
            if step >= (1 + 1 + 3) * 2:
                break
            Scheme.batch = torch.randint(low=0,high=N, size=(b,))
            with record_function("data io"):
                if cat == 'baseline':
                    pass
                elif cat == 'new':
                    Scheme.batch_whole_grad_A = Scheme._whole_grad_A_buffer[Scheme.batch].cuda()
                    Scheme.batch_whole_grad_B = Scheme._whole_grad_B_buffer[Scheme.batch].cuda()
                else:
                    raise NotImplementedError
            for scheme in schemes:
                grad = torch.randn((b, h, l, l), device='cuda')
                with record_function("set_scale"):
                    scheme.set_scale(grad)
                with record_function("get_scale"):
                    scheme.get_scale()
            with record_function("step"):
                if cat == 'baseline':
                    pass
                elif cat == 'new':
                    Scheme.step()
                else:
                    raise NotImplementedError
            prof.step()
        prof.stop()

    test_implementation(all_schemes, 'scheme_baseline', cat='baseline')
    test_implementation(all_schemes_new, 'scheme_new', cat='new')


def test_scheme_correctness():
    def init_scheme(cls):
        return cls(sample_ratio=0.1, 
                       deter_ratio=0.5, 
                       deter_adaptive=True,
                       minimal_k=10,
                       sample_replacement=True,
                       mix_replacement=True,
                       batch_dim_use_same_indices=False)
    N, L = 3940, 12
    b, h, l, d = 128, 12, 128, 64
    mat_shape = (b, h, l, l)
    Scheme.num_samples = N
    
    all_schemes, all_schemes_new = [], []
    for i in range(L):
        scheme = init_scheme(Scheme_4D)
        scheme.scheme_init(mat_shape)
        all_schemes.append(scheme)

        scheme_new = init_scheme(Scheme_4D_new)
        scheme_new.scheme_init(mat_shape)
        all_schemes_new.append(scheme_new)

    Scheme._init_buffer()
    assert Scheme._whole_grad_A_buffer is not None and Scheme._whole_grad_B_buffer is not None
    # import ipdb; ipdb.set_trace()
    Scheme.batch = torch.randint(low=0,high=N, size=(b,))

    def test_implementation(scheme, grad, cat='baseline'):
        if cat == 'baseline':
            pass
        elif cat == 'new':
            Scheme.batch_whole_grad_A = Scheme._whole_grad_A_buffer[Scheme.batch].cuda()
            Scheme.batch_whole_grad_B = Scheme._whole_grad_B_buffer[Scheme.batch].cuda()
        else:
            raise NotImplementedError
        scheme.set_scale(grad)
        x1, x2 = scheme.get_scale()
        return [x1.detach(), x2.detach()]
        
    grad = torch.randn((b, h, l, l), device='cuda')
    output_ref, output_ref_ =  test_implementation(all_schemes[0], grad, cat='baseline')
    output_us, output_us_ = test_implementation(all_schemes_new[0], grad, cat='new')
    print(torch.mean(torch.abs(output_ref - output_us)))
    print(torch.mean(torch.abs(output_ref_ - output_us_)))
    # np.testing.assert_allclose(output_ref, output_us, rtol=1e-3)
    # np.testing.assert_allclose(output_ref_, output_us_, rtol=1e-3)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    test_matmul_speed()
    # vis_matmul_speed()
    # test_io_speed()
    # test_scheme_correctness()