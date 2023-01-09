import numpy as np
from scipy.stats import norm

def eval_wang_risk(returns, param, eps=1e-8):
    """
    Evaluates the wang risk measure for a list of returns for a given parameter value.
    """

    returns = sorted(returns)
    ps = np.linspace(0, 1, num=len(returns), endpoint=False) + 1 / len(returns) / 2
    weights = np.zeros(len(returns))

    eval = 0.
    for i in range(len(returns)):
        ret = returns[i]
        p = ps[i]
        weights[i] = norm.pdf(norm.ppf(p) + param) / (norm.pdf(norm.ppf(p)) + eps) / len(returns)

    # ensure that weights are normalised
    weights /= np.sum(weights)
    return np.sum(weights * np.array(returns))
