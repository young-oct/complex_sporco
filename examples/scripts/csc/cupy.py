# -*- coding: utf-8 -*-
# @Time    : 2020-11-06 11:56 a.m.
# @Author  : young wang
# @FileName: cupy.py
# @Software: PyCharm
from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

# from sporco import signal
from sporco import fft
from sporco import metric
from sporco import plot
plot.config_notebook_plotting()
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import comcbpdn
from pytictoc import TicToc
from sporco.cupy import signal as si
from scipy import signal
from sporco.admm import comcbpdn

N = 468  # signal length
M = 2 # filter number
K = 5000 # test signal number

t = np.linspace(-45, 45, N, endpoint=False)

s_noise = np.zeros((N,K)).astype(np.complex)
s_clean = np.zeros((N,K)).astype(np.complex)
Dtemp = np.zeros((N,K)).astype(np.complex)
# delta matrix to shift the Gaussian pulses
delta = np.zeros((N)).astype(np.complex)
delta_1 = np.zeros((N)).astype(np.complex)
delta[-120] = 1
delta_1[-70] = 1
alpha = signal.gaussian(200, 30)
# construct sine/cosine signal
for i in range(K):

    sigma = 0.02  # noise level
    noise = sigma * si.complex_randn(N)
    # delta[np.random.randint(0, 32), i] = 1
    real_0,imag_0 = signal.gausspulse(t, 4.7, retquad=True, retenv=False)

    real_1 = np.convolve(delta,real_0,'same')
    imag_1 = np.convolve(delta, imag_0,'same')

    real_2 = np.convolve(delta_1, real_0, 'same')
    imag_2 = np.convolve(delta_1, imag_0, 'same')

    incus_real = 0.5 * real_1+0.3*real_2
    incus_imag = 0.5 * imag_1+0.3*imag_2

    TM = real_0 + complex(0, 1)*imag_0
    incus = incus_real + complex(0, 1) * incus_imag

    # temp = np.convolve(temp,delta[:,i],'same')

    if 500 < i < 700:
        s_clean[:,i] = TM + incus*alpha[500-i]
        s_noise[:,i] = s_clean[:,i] + noise
    else:
        s_clean[:, i] = TM
        s_noise[:, i] = s_clean[:, i] + noise

    Dtemp[:,i] = TM + noise

Nv = 650 # sample line index

train_index = np.random.choice(s_noise.shape[1],int(0.1*K))
s_train = s_noise[:,train_index]

D0 =Dtemp[:,0:M]

# Function computing reconstruction error at lmbda
Maxiter = 500
opt_par = comcbpdn.ComplexGenericConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                            'MaxMainIter': Maxiter,'RelStopTol': 5e-5, 'AuxVarObj': True,
                                            'RelaxParam': 1.515,'AutoRho': {'Enabled': True}})

s_clean = np.reshape(s_clean, (-1,1,K))
s_noise = np.reshape(s_noise, (-1,1,K))
s_train = np.reshape(s_train, (-1,1,len(train_index)))
s = s_noise.squeeze()
s_test = s_noise[:,:,Nv].squeeze()
s_test = s_test[:,np.newaxis]

D0 = np.reshape(D0,(-1,1,M))
D0_d = D0.squeeze()

t = TicToc()
lmbda  = 0.01

Kr = 0.05
Maxiter = 30

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))


t.tic()
# for i in range(s_noise_d.shape[1]):

b_i_2d = comcbpdn.ComplexConvBPDN(cp2np(D0_d), cp2np(s), lmbda*Kr, opt=opt_par, dimK=1, dimN=1)
# x_i_2d = cp2np( b_i_2d.solve())
t.toc()