from dataclasses import dataclass


@dataclass
class ApproxLinearConfig(object):
    sampling_ratio = 1.0
    minimal_k = 10
    only_bw = True
    tasks = ["dummy_task"]
    deter_ratio      = 0.5
    deter_adaptive   = True
    sample_replacement = True
    mix_replacement = False
    k_sampling       = True
    q_sampling       = True
    v_sampling       = True
    o_sampling       = False
    wi_0_sampling    = False
    wi_1_sampling    = False
    wo_sampling      = False
    score_sampling   = False
    attout_sampling  = False
