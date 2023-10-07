

import torch

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

