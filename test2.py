import numpy as np
import matplotlib.pyplot as plt
import functools
import math
from scipy.stats import norm

avg = 161
std = 32

cdf = functools.partial(norm.cdf, loc=avg, scale=std)

print(cdf(200) - cdf(120))

# top 10%

print(norm.ppf(0.9, loc=avg, scale=std))
