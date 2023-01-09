import numpy as np
from scipy.stats import norm

def wang_cdf(cdf, param):
    """ Takes the discrete CDF for some function, and returns the new CDF that
    has been reweighted according to the Wang transform.
    """
    new_cdf = norm.cdf(norm.ppf(cdf) + param)
    return new_cdf
