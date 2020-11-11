# -*- coding: utf-8 -*-
# @Time    : 2020-11-09 8:12 p.m.
# @Author  : young wang
# @FileName: convolution_dictionary.py
# @Software: PyCharm
"""
The script solves complex convolutional basis pursuit denosing problem with
ADMM consensus framework.
The script is modified from sporco.cupy.admm

The input signal is 486 points long gaussian pulse with complex noise
The dictionary is 486 points long gaussian pulse

The reconstructed signal should contain less noise than the input signal
"""

from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
from sporco import signal as si
from sporco import plot
from matplotlib import pyplot as plt
from pytictoc import TicToc
from scipy import signal
import GPUtil

from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import comcbpdn
# from sporco.admm import comcbpdn
#
N = 468  # signal length
M = 2 # filter number
K = 4 # test signal number

t = np.linspace(-45, 45, N, endpoint=False)

s_noise = np.zeros((N,K)).astype(np.complex)
s_clean = np.zeros((N,K)).astype(np.complex)

# construct gaussian clean signal and gaussian signal with complex noise
for i in range(K):

    sigma = 0.02  # noise level
    noise = sigma * si.complex_randn(N)
    # delta[np.random.randint(0, 32), i] = 1
    real_0,imag_0 = signal.gausspulse(t, 4.7, retquad=True, retenv=False)
    s_clean[:,i] = real_0*complex(1,0)+imag_0*complex(0,1)
    s_noise[:, i] =s_clean[:,i] + noise

# construct dictionary from gaussian clean signal with M dictionary states
D0 =s_clean[:,0:M]

# Function computing reconstruction error at lmbda
Maxiter = 500
opt_par = comcbpdn.ComplexConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                            'MaxMainIter': Maxiter,'RelStopTol': 5e-5, 'AuxVarObj': True,
                                            'RelaxParam': 1.515,'AutoRho': {'Enabled': True}})

lmbda = 0.1

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

t = TicToc()
t.tic()
b = comcbpdn.ComplexConvBPDN(np2cp(D0), np2cp(s_noise), lmbda, opt_par, dimK=1, dimN = 1)
X = cp2np(b.solve())
rec = cp2np(b.reconstruct().squeeze())
GPUtil.showUtilization()
t.toc()

plt.subplot(131)
plt.plot(s_clean[:,1].real)
plt.title('clean signal')
plt.subplot(132)
plt.plot(s_noise[:,1].real)
plt.title('noisy signal')
plt.subplot(133)
plt.plot(rec[:,1].real)
plt.title('recovered signal')
plt.show()