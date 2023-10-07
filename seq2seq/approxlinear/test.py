import os
# import ipdb
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def crs_wz_replace_f(A, B, k):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)
    topk_indices = torch.multinomial(prob, k, replacement=True)
    topk_indices, _ = torch.sort(topk_indices)
    A_, B_ = A[topk_indices] / k / prob[topk_indices].reshape(-1, 1), B[topk_indices]
    res = A_.T.matmul(B_)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        log1mexp(y)  # Hope for the best
    )


def compute_log_R(prob, num_points=1000, a=10.):
    log_p = torch.log(prob)
    # Computes the (log) ratio P(S\{s}|S \subseteq D\{s}) / P(S),
    # where S is an unordered sample under the Plackett-Luce model
    # Additionally computes the (conditional) second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S\{s}|S \subseteq D\{s})
    # Multiplying (or adding in log space) the results gives
    # The unconditional second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S)

    # Constant for numeric stability
    a = log_p.new_tensor(a)

    # Integrals are computed by the trapezoidal rule,
    # which equates to approximating the integral by
    # dx * sum_{i=1}^N (f(i) + f(i-1)) / 2 = dx / 2 * (f(0) + f(N) + 2 * sum_{i = 1}^{N-1} f(i))
    # Since f(0) and f(N) in our integral will be zero, we can just compute
    # dx * sum_{i = 1}^{N-1} f(i)
    # See https://en.wikipedia.org/wiki/Trapezoidal_rule

    # Create range of integration points, (1 ... N-1)/N (bounds are 0 to 1)
    log_v = (torch.arange(1, num_points, out=log_p.new()) / num_points).log()

    # First dim, numerical integration (N - 1)
    # Second dim, batch dimension (B)
    # Third dim, i in S (|S|)
    _q = gumbel_log_survival(-((log_p + a)[None, :, :] + torch.log(-log_v)[:, None, None]))

    # Compute the integrands (N - 1 x B)
    q = _q.sum(-1) + (torch.expm1(a + log1mexp(torch.logsumexp(log_p, -1)))[None, :] * log_v[:, None])

    q_without_s = q[..., None] - _q

    # Don't subtract same element twice for diagonals
    # skip_diag = 1 - torch.eye(log_p.size(-1), out=log_p.new())[None, None, :, :]

    # To compute the log probabilities, we should add constant a + phi_S, but these cancel out
    sum_S = torch.logsumexp(q, 0)  # e.g. log_P_S = a + phi_S + sum_S
    sum_S_s = torch.logsumexp(q_without_s, 0)
    return sum_S_s - sum_S[..., None]


def debias_crs_wo_replace_f(A, B, k):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)
    topk_indices = torch.multinomial(prob, k, replacement=False)
    topk_indices, _ = torch.sort(topk_indices)
    print(topk_indices.shape)
    log_r = compute_log_R(prob[topk_indices].reshape(1, -1))
    ratio = torch.exp(log_r)
    A_, B_ = A[topk_indices] * ratio.reshape(-1, 1), B[topk_indices]
    res = A_.T.matmul(B_)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def crs_wo_replace_f(A, B, k):
    # k = int(0.5 * k)
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)
    topk_indices = torch.multinomial(prob, k, replacement=False)
    topk_indices, _ = torch.sort(topk_indices)
    A_, B_ = A[topk_indices], B[topk_indices]
    res = A_.T.matmul(B_)
    torch.cuda.synchronize()
    dur = time.time() - t
    return res, dur


def sas_crs_wz_replace_f(A, B, k):
    t = time.time()
    torch.cuda.synchronize()
    A_row_norm, B_col_norm = A.norm(dim=1), B.norm(dim=1)
    norm_mult = A_row_norm * B_col_norm
    prob = norm_mult / torch.sum(norm_mult)

    sorted_prob, _ = torch.sort(prob, descending=True)
    min_val, sum_part_ratio = float('inf'), 0
    for i in range(8):
        cur_ratio = 0.1 * (i + 1)
        sum_part = int(cur_ratio * k)
        bar_q = 1 - torch.sum(sorted_prob[:sum_part])
        cur_val = bar_q / (k - sum_part)
        if cur_val < min_val:
            min_val = cur_val
            sum_part_ratio = cur_ratio
    # print(sum_part_ratio
    # )
    # cumsum_prob = torch.cumsum(sorted_prob, dim=0)
    # ipdb.set_trace()
    # sum_part_ratio = 0.2
    deter_k = int(k * sum_part_ratio)
    stoc_k = k - deter_k
    # deter sum part
    deter_topk_values, deter_topk_indices = topk(prob, deter_k, largest=True, sorted=False)
    p_c = torch.sum(deter_topk_values)
    # print(p_c)
    deter_topk_indices, _ = torch.sort(deter_topk_indices)
    A_, B_ = A[deter_topk_indices], B[deter_topk_indices]
    res = A_.T.matmul(B_)

    # stoc sample part
    residual_prob = prob.clone()
    residual_prob[deter_topk_indices] = 0.
    residual_prob = residual_prob / torch.sum(residual_prob)
    stoc_topk_indices = torch.multinomial(residual_prob, stoc_k, replacement=True)
    stoc_topk_indices, _ = torch.sort(stoc_topk_indices)
    # A_, B_ = A[stoc_topk_indices] / stoc_k / prob[stoc_topk_indices].reshape(-1, 1), B[stoc_topk_indices]
    A_, B_ = A[stoc_topk_indices] / stoc_k / prob[stoc_topk_indices].reshape(-1, 1), B[stoc_topk_indices]

    res += A_.T.matmul(B_) * (1 - p_c)
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
    re = 500
    results = {'crs_wz_replace': np.zeros(re),
               'crs_wo_replace': np.zeros(re),
               'sas_crs_wz_replace': np.zeros(re),
               'debias_crs_wo_replace': np.zeros(re)}
    for i in range(re):
        # rp_result, rp_time = rp(A, B, r=int(N * k))
        prob_crs_wz_replace_result, _ = crs_wz_replace_f(A, B, k=int(N * k))
        prob_crs_wo_replace_result, crs_time = crs_wo_replace_f(A, B, k=int(N * k))
        sas_crs_wz_replace_result, sas_crs_time = sas_crs_wz_replace_f(A, B, k=int(N * k))
        debias_crs_wo_replace_result, debias_crs_wo_replace_time = debias_crs_wo_replace_f(A, B, k=int(N * k))
        results['crs_wz_replace'][i] = prob_crs_wz_replace_result.item()
        results['crs_wo_replace'][i] = prob_crs_wo_replace_result.item()
        results['sas_crs_wz_replace'][i] = sas_crs_wz_replace_result.item()
        results['debias_crs_wo_replace'][i] = debias_crs_wo_replace_result.item()
    return results


def run_exp(num_experiments, N, D, k):
    results = {}
    for i in tqdm(range(num_experiments)):
        A, B = torch.rand((N, D), device='cuda'), torch.randn((N, D), device='cuda')
        ori_result = ori(A, B)[0].item()
        results[ori_result] = single_exp(A, B, N, k)
    return results


if __name__ == '__main__':
    N, D = 12800, 1
    for i in range(1, 2):
        k = 0.1 * i
        num_exp = 2
        results = run_exp(num_exp, N, D, k)
        crs_wz_replace = []
        crs_wo_replace = []
        sas_crs_wz_replace = []
        debias_crs_wo_replace = []
        for ori_res, approx_dict in results.items():
            crs_wz_replace.append((approx_dict['crs_wz_replace'] - ori_res) / np.abs(ori_res))
            crs_wo_replace.append((approx_dict['crs_wo_replace'] - ori_res) / np.abs(ori_res))
            sas_crs_wz_replace.append((approx_dict['sas_crs_wz_replace'] - ori_res) / np.abs(ori_res))
            debias_crs_wo_replace.append((approx_dict['debias_crs_wo_replace'] - ori_res) / np.abs(ori_res))
        crs_wz_replace = np.hstack(crs_wz_replace)
        crs_wo_replace = np.hstack(crs_wo_replace)
        sas_crs_wz_replace = np.hstack(sas_crs_wz_replace)
        debias_crs_wo_replace = np.hstack(debias_crs_wo_replace)
        plt.boxplot([crs_wz_replace, crs_wo_replace, debias_crs_wo_replace, sas_crs_wz_replace], showfliers=False)
        plt.grid()
        plt.xticks([1, 2, 3, 4], ['w/', '(bias) w/o', '(debias) w/o', 'sas w/'])
        plt.savefig(r'./vis_results_k{k}.png')
        plt.clf()