import numpy as np 
import sys

def fft(x):
    n = len(x)
    if n == 1: return x
    even = fft(x[::2]) #starting at 0 take steps of size 2
    odd = fft(x[1::2]) #starting at 1 take steps of size 2


    w = np.exp(2 * np.pi * np.i0)
    print('hello world')
    
if __name__ == '__main__':
    x = [1, 2, 3, 4]
    if x % 2 > 0:
        raise ValueError("Input must be a power of 2")
4