import torch
import pdb
from torch import topk
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd


class approxmatmul_4D_fw_and_bw(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, A, B, ratio, minimal_k, scheme, batch_dim_use_same_indices=True):
        assert len(A.shape) == 4
        assert len(B.shape) == 4
        if scheme is not None:
            grad_A, grad_B = scheme.get_scale()
        else:
            grad_A, grad_B = None, None

        if ratio == 1.0 or grad_A is None or grad_B is None:
            B_top_k = B
            A_top_k = A
            top_k_indices = None
        else:
            A_top_k, B_top_k, top_k_indices = weighted_subsample_4D_AB(A, B, grad_B, ratio, minimal_k,
                                                        batch_dim_use_same_indices=batch_dim_use_same_indices)

        ctx.saved = A_top_k, B_top_k, top_k_indices, A.shape, B.shape
        ctx.scheme = scheme
        res = torch.matmul(A_top_k, B_top_k)
        return res


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A_top_k, B_top_k, top_k_indices, A_shape, B_shape = ctx.saved
        if top_k_indices is None: # B_top_k.shape[3] == B_shape[3]:
            grad_A = torch.einsum('...mn, ...dn-> ...md', grad_output, B_top_k)
            grad_B  = torch.einsum('...md, ...mn-> ...dn', A_top_k, grad_output)
        else:
            grad_A = torch.zeros(A_shape, device=A_top_k.device)
            grad_A_ = torch.einsum('...mn, ...kn-> ...mk', grad_output, B_top_k)
            grad_A = grad_A.scatter_(3, index=top_k_indices.unsqueeze(2).expand(*A_shape[:3], top_k_indices.shape[-1]), src=grad_A_)


            grad_B = torch.zeros(B_shape, device=B_top_k.device)
            grad_B_  = torch.einsum('...mk, ...mn-> ...kn', A_top_k, grad_output)
            grad_B = grad_B.scatter_(2, index=top_k_indices.unsqueeze(-1).expand(*top_k_indices.shape, grad_B.shape[-1]), src=grad_B_)
            # torch.arange(b*h).unsqueeze(1)

        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)
        return grad_A, grad_B, None, None, None, None


class approxmatmul_4D_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, A, B, scheme):
        assert len(A.shape) == 4
        assert len(B.shape) == 4
        if scheme is not None:
            grad_A, grad_B = scheme.get_scale()
        else:
            grad_A, grad_B = None, None

        if scheme is None or scheme.sample_ratio == 1.0 or grad_A is None or grad_B is None:
            ctx.saved = B.transpose(3, 2), None, None, None, True, A, None, None, None, True, A.shape, B.shape
        else:
            B_top_k_trans_deter, top_k_indices_B_deter, B_top_k_trans_stoc, top_k_indices_B_stoc, fp_b \
                = subsample_4D_B_by_norm(grad_A, B.transpose(3, 2), None, scheme)

            A_top_k_deter, top_k_indices_A_deter, A_top_k_stoc, top_k_indices_A_stoc, fp_a \
                = subsample_4D_B_by_norm(grad_B, A, None, scheme)

            ctx.saved = B_top_k_trans_deter, top_k_indices_B_deter, B_top_k_trans_stoc, top_k_indices_B_stoc, fp_b, \
                        A_top_k_deter, top_k_indices_A_deter, A_top_k_stoc, top_k_indices_A_stoc, fp_a, \
                        A.shape, B.shape

        ctx.scheme = scheme
        res = torch.matmul(A, B)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):

        B_top_k_trans_deter, top_k_indices_B_deter, B_top_k_trans_stoc, top_k_indices_B_stoc, fp_b, \
        A_top_k_deter, top_k_indices_A_deter, A_top_k_stoc, top_k_indices_A_stoc, fp_a, \
        A_shape, B_shape = ctx.saved

        # b, h, m, d = A_shape
        # _, _, _, n = B_shape
        # _, _, _, k = B_top_k.shape
        # print("B={}, h={}, m={}, n={}, d={}, k={}".format(b, h, m, n, d, k))
        # print(A_shape, B_shape)
        # print(A_top_k.shape, B_top_k.shape)
        if fp_b: # B_top_k.shape[3] == B_shape[3]:
            grad_A = torch.einsum('...mn, ...nd->...md', grad_output, B_top_k_trans_deter)
        else:
            # grad_output: b x h x m x n
            # top_k_indices_B: bh x k
            # top_k_indices_gather_B: b x h x m x k
            # grad_output_ Shape: b x h x m x k
            # grad_A Shape: b x h x m x d

            grad_A_deter = 0.
            if top_k_indices_B_deter is not None:
                grad_output_deter = torch.gather(grad_output, dim=3, index=top_k_indices_B_deter.unsqueeze(2).expand(*A_shape[:3], top_k_indices_B_deter.shape[-1]))
                grad_A_deter = torch.einsum('...mk, ...kd->...md', grad_output_deter, B_top_k_trans_deter)

            grad_A_stoc = 0.
            if top_k_indices_B_stoc is not None:
                grad_output_stoc = torch.gather(grad_output, dim=3, index=top_k_indices_B_stoc.unsqueeze(2).expand(*A_shape[:3], top_k_indices_B_stoc.shape[-1]))
                grad_A_stoc = torch.einsum('...mk, ...kd->...md', grad_output_stoc, B_top_k_trans_stoc)

            grad_A = grad_A_deter + grad_A_stoc
            # print(grad_A_deter.shape, grad_A_stoc.shape, B_weight_stoc.shape)

        if fp_a: # A_top_k.shape[2] == A_shape[2]:
            grad_B  = torch.einsum('...md, ...mn-> ...dn', A_top_k_deter, grad_output)
        else:
            # grad_output: b x h x m x n
            # top_k_indices_A: bh x k
            # top_k_indices_gather_A: b x h x k x n
            # grad_output_: b x h x k x n
            # grad_B Shape: b x h x d x n

            grad_B_deter = 0
            if top_k_indices_A_deter is not None:
                grad_output_deter = torch.gather(grad_output, dim=2, index=top_k_indices_A_deter.unsqueeze(-1).expand(*top_k_indices_A_deter.shape, grad_output.shape[-1]))
                grad_B_deter = torch.einsum('...kd, ...kn-> ...dn', A_top_k_deter, grad_output_deter)

            grad_B_stoc = 0
            if top_k_indices_A_stoc is not None:
                grad_output_stoc = torch.gather(grad_output, dim=2, index=top_k_indices_A_stoc.unsqueeze(-1).expand(*top_k_indices_A_stoc.shape, grad_output.shape[-1]))
                grad_B_stoc = torch.einsum('...kd, ...kn-> ...dn', A_top_k_stoc, grad_output_stoc)

            grad_B = grad_B_deter + grad_B_stoc
            # print(grad_B_deter.shape, grad_B_stoc.shape, A_weight_stoc.shape)

        if ctx.scheme:
            # print("A Shape:", A_shape, "B Shape:", B_shape)
            ctx.scheme.set_scale(grad_output)

        del B_top_k_trans_deter, top_k_indices_B_deter, B_top_k_trans_stoc, top_k_indices_B_stoc, \
            A_top_k_deter, top_k_indices_A_deter, A_top_k_stoc, top_k_indices_A_stoc, \
            A_shape, B_shape

        return grad_A, grad_B, None


@torch.no_grad()
def subsample_4D_B_by_norm(A, B, weight, scheme):
    # A: b x h x n
    # B: b x h x n x d
    # print(f'A: {A.shape}')
    # print(f'B: {B.shape}')
    b, h, _ = A.shape
    _, _, n, d = B.shape
    k_candidate = int(n * scheme.sample_ratio)
    k = min(max(k_candidate, scheme.minimal_k), n)

    # print(k, deter_k, stoc_k)

    if k == n:
        return B, None, None, None, True

    a_col_norms = A.reshape(-1, n)  # bh x n
    if weight is None:
        b_row_norms = torch.norm(B, dim=-1).view(-1, n)  # bh x n
    else:
        b_row_norms = torch.norm(B * (weight.norm(dim=2)).unsqueeze(2), dim=-1).view(-1, n)  # bh x n

    if a_col_norms.sum() == 0.:
        norm_mult = b_row_norms
    else:
        norm_mult = a_col_norms * b_row_norms

    prob = (norm_mult) / (norm_mult.sum(dim=1).unsqueeze(dim=1))

    #### debiased sampling
    # # top_k_indices Shape: bh x k
    # # B_top_k_dim Shape: b x h x k x d

    if not scheme.sample_replacement:
        B_top_k_stoc, top_k_indices_stoc = sample_wo_replacement_4D(B, prob, k, (b, h, n, d))
        return None, None, B_top_k_stoc, top_k_indices_stoc, False

    deter_k, stoc_k = k_adaptive_4D(prob, k)  ######### Optimize the deterministic ratio

    # if deter_k == 0:
    #     B_top_k_stoc, top_k_indices_stoc = sample_wo_replacement_4D(B, prob, stoc_k, (b, h, n, d))
    #     return None, None, B_top_k_stoc, top_k_indices_stoc, False

    B_top_k_deter, top_k_indices_deter, top_k_prob, residual_prob, mask = topk_selection_4D(B, prob, deter_k, (b, h, n, d))

    if stoc_k == 0:
        return B_top_k_deter, top_k_indices_deter, None, None, False

    B_top_k_stoc, top_k_indices_stoc = sample_with_replacement_4D(B, residual_prob, stoc_k, (b, h, n, d), top_k_prob, a_col_norms.sum(), mask)

    return B_top_k_deter, top_k_indices_deter, B_top_k_stoc, top_k_indices_stoc, False


@torch.no_grad()
def k_adaptive_4D(prob, k):

    top_k_values, _ = topk(prob, k, dim=1, largest=True)
    top_k_values, _ = torch.sort(top_k_values, dim=1, descending=True)

    # print(top_k_values_deter[0], top_k_indices_deter[0])
    top_k_prob_buf = torch.cumsum(top_k_values, dim=1)  # q_c
    score = (1 - top_k_prob_buf) / (k - (torch.arange(k, device=prob.device) + 1))
    score_ = score.mean(dim=0)

    min_score, deter_k_opt = score_[:-1].min(dim=0)
    deter_k = 0 if 1 / k < min_score else deter_k_opt + 1
    stoc_k = k - deter_k

    return deter_k, stoc_k


@torch.no_grad()
def topk_selection_4D(B, prob, deter_k, shape, eps=1e-3):

    (b, h, n, d) = shape

    if deter_k == 0:
        return None, None, torch.zeros((b * h, 1), device=prob.device), prob, torch.ones((b * h, 1), device=prob.device).type(torch.int)

    top_k_values_deter, top_k_indices_deter = topk(prob, deter_k, dim=1, largest=True)
    top_k_indices_deter, _ = torch.sort(top_k_indices_deter, dim=1)
    residual_prob = prob.scatter(dim=1, index=top_k_indices_deter, src=(top_k_values_deter * 0.))
    top_k_indices_deter = top_k_indices_deter.view(b, h, -1)
    B_top_k_deter = torch.gather(B, dim=2, index=top_k_indices_deter.unsqueeze(-1).expand(b, h, deter_k, d))
    top_k_prob = top_k_values_deter.sum(dim=1)
    mask = (top_k_prob <= (1 - eps)).type(torch.int).unsqueeze(dim=1)

    return B_top_k_deter, top_k_indices_deter, top_k_prob, residual_prob, mask


@torch.no_grad()
def sample_wo_replacement_4D(B, prob, stoc_k, shape):

    (b, h, n, d) = shape
    top_k_indices_stoc = torch.multinomial(prob, stoc_k, replacement=False)
    top_k_indices_stoc, _ = torch.sort(top_k_indices_stoc, dim=1)
    top_k_indices_stoc = top_k_indices_stoc.view(b, h, -1)
    B_top_k_stoc = torch.gather(B, dim=2, index=top_k_indices_stoc.unsqueeze(-1).expand(b, h, stoc_k, d))
    return B_top_k_stoc, top_k_indices_stoc


@torch.no_grad()
def sample_with_replacement_4D(B, prob, stoc_k, shape, top_k_prob, a_sum, mask):

    (b, h, n, d) = shape
    # mask = ((prob > 0).sum(dim=1) > 0).type(torch.int8).view(-1,1)
    prob_ = mask * prob + (1 - mask) * torch.ones_like(prob)
    top_k_indices_stoc = torch.multinomial(prob_, stoc_k, replacement=True)
    top_k_indices_stoc, _ = torch.sort(top_k_indices_stoc, dim=1)
    top_k_indices_stoc = top_k_indices_stoc.view(b, h, -1)

    B_top_k_stoc = torch.gather(B, dim=2, index=top_k_indices_stoc.unsqueeze(-1).expand(b, h, stoc_k, d))
    stoc_weight = (1 - top_k_prob).view(b, h, 1, 1)
    mask = mask.view(b, h, 1, 1)

    if a_sum == 0.:
        B_top_k_stoc_unbiased = stoc_weight * (B_top_k_stoc) * mask

    else:
        B_top_k_stoc_unbiased = stoc_weight * (B_top_k_stoc) * mask
        prob_sampled = torch.gather(prob_.view(b, h, n), dim=2, index=top_k_indices_stoc)
        B_top_k_stoc_unbiased = B_top_k_stoc_unbiased / prob_sampled.unsqueeze(dim=3) / stoc_k

    return B_top_k_stoc_unbiased, top_k_indices_stoc



class approxlinear_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias, scheme):

        scale = scheme.get_scale() if scheme is not None else None
        if scheme is None or scheme.sample_ratio == 1.0 or scale is None:
            ctx.saved = input, None, None, None, True, weight, bias

        else:
            subsampled_input_deter, top_k_indices_deter, subsampled_input_stoc, top_k_indices_stoc, fp \
                = subsample_3D_input_by_norm(input, scale, scheme)

            ctx.saved = subsampled_input_deter, top_k_indices_deter, subsampled_input_stoc, top_k_indices_stoc, \
                        fp, weight, bias

        ctx.scheme = scheme
        res = F.linear(input, weight, bias)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        subsampled_input_deter, top_k_indices_deter, subsampled_input_stoc, top_k_indices_stoc, \
        fp, weight, bias = ctx.saved

        grad_input = torch.matmul(grad_output, weight)
        if fp:
            grad_weight = torch.einsum('blo, bli->oi', grad_output, subsampled_input_deter)
        else:
            # weight Shape: #d_output x #d_input
            # subsampled_input: n x L x k
            # grad_output: n x L x #d_output
            # top_k_indices Shape: k
            # grad_weight_: #d_output x k
            grad_output = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], -1)

            grad_weight_deter = 0.
            if top_k_indices_deter is not None:
                grad_output_deter = grad_output[top_k_indices_deter]
                grad_weight_deter = torch.einsum('ko, ki->oi', grad_output_deter, subsampled_input_deter)

            grad_weight_stoc = 0.
            if top_k_indices_stoc is not None:
                grad_output_stoc = grad_output[top_k_indices_stoc]
                grad_weight_stoc = torch.einsum('ko, ki->oi', grad_output_stoc, subsampled_input_stoc)

            grad_weight = grad_weight_deter + grad_weight_stoc


        if bias is not None:
            grad_bias = grad_output.view(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        del subsampled_input_deter, top_k_indices_deter, subsampled_input_stoc, top_k_indices_stoc, weight, bias

        return grad_input, grad_weight, grad_bias, None


@torch.no_grad()
def subsample_3D_input_by_norm(input, grad_norm, scheme):
    """
    The shape of input tensor is BLD, where B is the batch size, L is the sentence length and D is the embedding dim.
    batch_dim_use_same_indices == False means each input sample could have custom top-k indices, thus the indices shape is
    B x k where k is the size after down-sampling (k << D). In contrast, batch_dim_use_same_indices == True
    means we enforce the top-k indices are the same along the B and L dims. And in this case, the shape of top-k indices is [k].
    """
    # A: input: n x L x #d_input
    # B: weight: #d_input x #d_output
    in_features = input.shape[0] * input.shape[1]
    k_candidate = int(in_features * scheme.sample_ratio)
    k = min(max(k_candidate, scheme.minimal_k), in_features)
    # print(k, deter_k, stoc_k)
    ## ===================================
    ## select only according to weight norm
    # norm_mult = b_row_norms
    ## ===================================

    if k == in_features:
        return input, None, None, None, True

    a_col_norms = torch.norm(input.view(-1, input.shape[-1]), dim=1)

    if grad_norm is None or grad_norm.sum() == 0.:
        # print(a_col_norms.shape, b_row_norms.shape, scale.shape)
        norm_mult = a_col_norms
    else:
        norm_mult = a_col_norms * grad_norm.view(-1)

    prob = (norm_mult) / (norm_mult.sum())


    # print(deter_k_opt, k, min_score, 1/k)

    if not scheme.sample_replacement:
        input_top_k_stoc, top_k_indices_stoc = sample_wo_replacement_3D(input, prob, k, in_features)
        return None, None, input_top_k_stoc, top_k_indices_stoc, False

    deter_k, stoc_k = k_adaptive_3D(prob, k)

    # if deter_k == 0:
    #     input_top_k_stoc, top_k_indices_stoc = sample_wo_replacement_3D(input, prob, stoc_k, in_features)
    #     return None, None, input_top_k_stoc, top_k_indices_stoc, False

    input_top_k_deter, top_k_indices_deter, top_k_prob, residual_prob, mask = topk_selection_3D(input, prob, deter_k, in_features)

    if stoc_k == 0:
        return input_top_k_deter, top_k_indices_deter, None, None, False

    input_top_k_stoc, top_k_indices_stoc = sample_with_replacement_3D(input, residual_prob, stoc_k,
                                                    in_features, top_k_prob, None if grad_norm is None else grad_norm.sum(), mask)

    return input_top_k_deter, top_k_indices_deter, input_top_k_stoc, top_k_indices_stoc, False



@torch.no_grad()
def k_adaptive_3D(prob, k):

    top_k_values, _ = topk(prob, k, largest=True)
    top_k_values, _ = torch.sort(top_k_values, descending=True)

    top_k_prob_buf = torch.cumsum(top_k_values, dim=0)  # [:-1]
    score_ = (1 - top_k_prob_buf) / (k - (torch.arange(k, device=prob.device) + 1))

    min_score, deter_k_opt = score_.min(dim=0)
    deter_k = 0 if 1 / k < min_score else deter_k_opt + 1
    stoc_k = k - deter_k

    return deter_k, stoc_k


@torch.no_grad()
def topk_selection_3D(input, prob, deter_k, in_features, eps=1e-3):

    if deter_k == 0:
        return None, None, 0., prob, True

    top_k_values_deter, top_k_indices_deter = topk(prob, deter_k, largest=True)
    top_k_indices_deter, _ = torch.sort(top_k_indices_deter)
    input_top_k_deter = input.view(in_features, -1)[top_k_indices_deter, ...]
    top_k_prob = top_k_values_deter.sum()
    residual_prob = prob.clone()
    residual_prob[top_k_indices_deter] = 0

    return input_top_k_deter, top_k_indices_deter, top_k_prob, residual_prob, (top_k_prob < (1-eps))


@torch.no_grad()
def sample_wo_replacement_3D(input, prob, stoc_k, in_features):

    top_k_indices_stoc = torch.multinomial(prob, stoc_k, replacement=False)
    top_k_indices_stoc, _ = torch.sort(top_k_indices_stoc)
    input_top_k_stoc = input.view(in_features, -1)[top_k_indices_stoc, ...]

    return input_top_k_stoc, top_k_indices_stoc


@torch.no_grad()
def sample_with_replacement_3D(input, prob, stoc_k, in_features, top_k_prob, grad_norm_sum, mask):

    if not mask:
        return None, None

    top_k_indices_stoc = torch.multinomial(prob, stoc_k, replacement=True)
    top_k_indices_stoc, _ = torch.sort(top_k_indices_stoc)
    input_top_k_stoc = input.view(in_features, -1)[top_k_indices_stoc, ...]

    if grad_norm_sum is None or grad_norm_sum == 0.:
        input_top_k_stoc_unbiased = (1 - top_k_prob) * (input_top_k_stoc)

    else:
        input_top_k_stoc_unbiased = (1 - top_k_prob) * (input_top_k_stoc)
        prob_sampled = prob[top_k_indices_stoc].unsqueeze(dim=1)
        input_top_k_stoc_unbiased = input_top_k_stoc_unbiased / prob_sampled / stoc_k

    return input_top_k_stoc_unbiased, top_k_indices_stoc