import random
import multiprocessing
import numpy as np 
from cmath import exp, pi
import sys

sys.setrecursionlimit(2**20)
Q = multiprocessing.Queue()

def fft(x):
    N = len(x)
    if N == 1: return x
    w = exp(-2*pi*1j / N) #compute roots of unity
    even = fft(x[::2]) 
    odd = fft(x[1::2])
    x = even + odd #putting even and odd back together.

    for k in range(N // 2):
        a = x[k]
        x[k] = a + w**k*x[k + N//2]
        x[k + N//2] = a - w**k*x[k + N//2]
    return x
    
if __name__ == '__main__':

    x = [random.randint(0, 1000) for _ in range(2 ** 8)]

    if len(x) % 2 > 0:
        raise ValueError("Input must be a power of 2")

    f = fft(x)
    print(f)