import random
import multiprocessing
import timeit
import numpy as np 
from cmath import exp, pi
import sys
from numpy.fft import fft as np_fft

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

def parallel_fft(x):
    N = len(x)
    if N == 1: return x

    w = exp(-2*pi*1j / N) #compute roots of unity

    even = x[::2]
    odd = x[1::2]

    #wont work as getting results from process if extremely difficult
    if len(x) > 128:
        p1 = multiprocessing.Process(target=parallel_fft, args=(even))
        p1.start()
        p2 = multiprocessing.Process(target=parallel_fft, args=(odd))
        p2.start()
        p1.join()
        p2.join()
    else: 
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

    #start_time = timeit.default_timer()
    f = fft(x)
    #print(timeit.default_timer() - start_time)
    #start_time = timeit.default_timer()
    #print(timeit.default_timer() - start_time)
    print(f)