"""
Helper functions to be used across zeus21

Authors: Yonatan Sklansky, Emilie Thelie
UT Austin - February 2025

"""

import numpy as np
import time
from pyfftw import empty_aligned as empty
import powerbox as pbox


def tophat_smooth(rr, ks, dk):
    x = ks * rr + 1e-5
    win_k = 3/(x**3) * (np.sin(x) - x*np.cos(x))
    deltakfilt = dk * win_k
    return np.real(np.fft.ifftn(deltakfilt))

def find_nearest_idx(array, values):
    array = np.atleast_1d(array)
    values = np.atleast_1d(values)
    idx = []
    for i in range(len(values)):
        idx.append((np.abs(array - values[i])).argmin())
    return np.unique(idx)

def print_timer(start_time, text=""):
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time//60)
    secs = int(elapsed_time - mins*60)
    print(f"\n{mins}min {secs}s\n")
    print(text)

def v2r(v):
    return (3/4/np.pi * v)**(1/3)
    
def r2v(r):
    return 4/3 * np.pi * r**3