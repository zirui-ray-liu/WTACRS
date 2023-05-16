import os
# import ipdb
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import topk
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt


def rp(A, B, r):
    t = time.time()
    torch.cuda.synchronize()
    N = A.shape[0]
    rp_mat = torch.randn(r, N, device='cuda') / math.sqrt(r)
    proj_A = rp_mat.matmul(A)
    proj_B = rp_mat.matmul(B)
    res = proj_A.T.matmul(proj_B)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def prob_crs(A, B, k, replacement=False):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)
    topk_indices = torch.multinomial(prob, k, replacement=replacement)
    topk_indices, _ = torch.sort(topk_indices)
    A_, B_ = A[topk_indices], B[topk_indices]
    res = A_.T.matmul(B_)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def debiased_prob_crs_a(A, B, k):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)

    sum_part_ratio = 0.5
    deter_k = int(k * sum_part_ratio)
    stoc_k = k - deter_k

    # deter sum part
    deter_topk_values, deter_topk_indices = topk(prob, deter_k, largest=True)
    p_c = torch.sum(deter_topk_values)
    deter_topk_indices, _ = torch.sort(deter_topk_indices)
    A_, B_ = A[deter_topk_indices], B[deter_topk_indices]
    res = A_.T.matmul(B_)

    # stoc sample part
    prob[deter_topk_indices] = 0.
    residual_prob = prob / torch.sum(prob)
    stoc_topk_indices = torch.multinomial(residual_prob, stoc_k, replacement=False)
    stoc_topk_indices, _ = torch.sort(stoc_topk_indices)
    prob_be_sampled = residual_prob[stoc_topk_indices]
    A_, B_ = A[stoc_topk_indices], B[stoc_topk_indices]
    res += A_.T.matmul(B_) * (1 - p_c)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def debiased_prob_crs_b(A, B, k):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)

    sum_part_ratio = 0.5
    deter_k = int(k * sum_part_ratio)
    stoc_k = k - deter_k

    # deter sum part
    deter_topk_values, deter_topk_indices = topk(prob, deter_k, largest=True)
    p_c = torch.sum(deter_topk_values)
    deter_topk_indices, _ = torch.sort(deter_topk_indices)
    A_, B_ = A[deter_topk_indices], B[deter_topk_indices]
    res =  A_.T.matmul(B_)

    # stoc sample part
    residual_prob = prob.clone()
    residual_prob[deter_topk_indices] = 0.
    residual_prob = residual_prob / torch.sum(residual_prob)
    stoc_topk_indices = torch.multinomial(residual_prob, stoc_k, replacement=False)
    stoc_topk_indices, _ = torch.sort(stoc_topk_indices)
    A_, B_ = A[stoc_topk_indices] / stoc_k / residual_prob[stoc_topk_indices].reshape(-1, 1), B[stoc_topk_indices]
    res +=  A_.T.matmul(B_) * (1 - p_c)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def ori(A, B):
    t = time.time()
    torch.cuda.synchronize()
    res = A.T.matmul(B)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def single_exp(A, B, N, k):
    original_result, ori_time = ori(A, B)
    original_result = original_result.item()
    re = 1000
    results = {'rp': np.zeros(re), 'prob_crs': np.zeros(re), 'prob_rep_crs': np.zeros(re), 'debiased_prob_crs': np.zeros(re)}
    for i in range(re):
        rp_result, rp_time = rp(A, B, r=int(N * k))
        crs_result, crs_time = prob_crs(A, B, k=int(N * k), replacement=False)
        crs_rep_result, crs_rep_time = prob_crs(A, B, k=int(N * k), replacement=True)
        debiased_crs_result_a, debiased_crs_time_a = debiased_prob_crs_a(A, B, k=int(N * k))
        debiased_crs_result_b, debiased_crs_time_b = debiased_prob_crs_b(A, B, k=int(N * k))
        results['rp'][i] = rp_result.item()
        results['prob_crs'][i] = crs_result.item()
        results['prob_rep_crs'][i] = crs_rep_result.item()
        results['debiased_prob_crs_a'][i] = debiased_crs_result_a.item()
        results['debiased_prob_crs_b'][i] = debiased_crs_result_b.item()

    return results


def run_exp(num_experiments, N, D, k):
    results = {}
    for i in tqdm(range(num_experiments)):
        A, B = torch.randn((N, D), device='cuda'), torch.randn((N, D), device='cuda')
        ori_result = ori(A, B)[0].item()
        results[ori_result] = single_exp(A, B, N, k)
    return results


if __name__ == '__main__':
    N, D = 10000, 1
    k = 0.1
    results = run_exp(1, N, D, k)
    ori_res = list(results.keys())[0]
    approx_dict = results[ori_res]
    rp_res = (approx_dict['rp'] - ori_res) / np.abs(ori_res)
    crs_res = (approx_dict['prob_crs'] - ori_res) / np.abs(ori_res)
    crs_rep_res = (approx_dict['prob_rep_crs'] - ori_res) / np.abs(ori_res)
    debiased_crs_res_a = (approx_dict['debiased_prob_crs_a'] - ori_res) / np.abs(ori_res)
    debiased_crs_res_b = (approx_dict['debiased_prob_crs_b'] - ori_res) / np.abs(ori_res)

    print(rp_res.mean(), rp_res.std())
    print(crs_res.mean(), crs_res.std())
    print(crs_rep_res.mean(), crs_rep_res.std())
    print(debiased_crs_res_a.mean(), debiased_crs_res_a.std())
    print(debiased_crs_res_b.mean(), debiased_crs_res_b.std())

    plt.boxplot([rp_res, crs_res, debiased_crs_res_a, debiased_crs_res_b])
    plt.grid()
    plt.xticks([1, 2, 3], ['rp', 'prob_crs', 'debiased_prob_crs'])
    plt.savefig('./vis_results')
