# Standard Library
import math


def poisson(k, n):
    return math.exp(-k) * (k**n) / math.factorial(n)
