import numpy as np
import matplotlib.pyplot as plt
import functools
import math
from scipy.stats import norm


def normal_dist(x, mu, sigma):
    return (
        1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


def normal_dist_2d(x, y, mu, sigma):
    return (
        1
        / (2 * np.pi * sigma**2)
        * np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma**2))
    )


def normal_dist_cdf(x, mu, sigma):
    return 0.5 * (1 + norm.cdf((x - mu) / (sigma * np.sqrt(2))))


def copula_1(fun1, fun2, x, y):
    x = fun1(x)
    y = fun2(y)
    return 2 * x + 2 * y - 4 * x * y


def copula_2(fun1, fun2, x, y):
    x = fun1(x)
    y = fun2(y)
    return 2 - 2 * x - 2 * y + 4 * x * y


def density(copula, F1, F2, f1, f2, x, y):
    return copula(F1, F2, x, y) * f1(x) * f2(y)


std_normal = functools.partial(normal_dist, mu=0, sigma=1)
std_normal_2d = functools.partial(normal_dist_2d, mu=0, sigma=1)
std_normal_cdf = functools.partial(normal_dist_cdf, mu=0, sigma=1)

normal_1 = functools.partial(normal_dist, mu=0, sigma=1)
normal_2 = functools.partial(normal_dist, mu=0, sigma=2)
normal_1_cdf = functools.partial(normal_dist_cdf, mu=0, sigma=1)
normal_2_cdf = functools.partial(normal_dist_cdf, mu=0, sigma=2)
# X = np.linspace(-5, 5, 100)
# y = std_normal_cdf(X)

# plt.plot(X, y)
# plt.show()

fun1 = functools.partial(
    density, copula_1, normal_1, normal_2, normal_1_cdf, normal_2_cdf
)
fun2 = functools.partial(
    density, copula_2, normal_1, normal_2, normal_1_cdf, normal_2_cdf
)

X = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, X)
Z = fun1(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z)
# plt.show()

X = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, X)
Z = fun2(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z)
plt.show()
