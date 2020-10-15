# -*- coding: utf-8 -*-
# @Time    : 2020-10-09 6:01 p.m.
# @Author  : young wang
# @FileName: complex_cbpdn.py
# @Software: PyCharm

# from __future__ import division

"""
The script solves convolutional basis pursuit denosing problem using the
framework proposed by

J. Kang, D. Hong, J. Liu, G. Baier, N. Yokoya and B. Demir,
"Learning Convolutional Sparse Coding on Complex Domain for Interferometric Phase Restoration," in
IEEE Transactions on Neural Networks and Learning Systems,
doi: 10.1109/TNNLS.2020.2979546.

matlab implementation by the authors can be found https://github.com/jiankang1991/ComCSC
input signal: complex sine wave with complex Gaussian noise
input dictionary: complex sine waves

"""

import numpy as np
from sporco.admm import comcbpdn, cbpdn
from scipy import signal
from sporco import signal as si
from tqdm import tqdm
from sporco import fft
from sporco import plot

from matplotlib import pyplot as plt

import random
from numpy.fft import fft, fft2, ifft, ifft2
from sklearn.metrics import r2_score
import sinesignal
import pickle
import copy
from math import ceil

# load measured point spread function
# with open('psf_complex','rb') as f:
#     psf_complex = pickle.load(f)
#     f.close()
# #load training signals: onion
# with open('onion_complex','rb') as f:
#     onion_complex = pickle.load(f)
#     f.close()

np.seterr(divide='ignore')

N = 128  # signal length

A = 5  # amplitude
M = 10  # filter number
fd = np.linspace(5, 20, M)
fs = 128  # sampling frequency must be higher enough to eliminate cutoff
dimN = 1  # spatial dimension 1

# create a complex gaussian noise using sporco.signal.complex_randn,
# same length as input signal
noise = si.complex_randn(N)

D0 = np.zeros((N, M), dtype=complex)
# construct a complex sine dictionary
for i in range(len(fd)):
    D0[:, i] = sinesignal.generate_sine_wave_no(N, A, fd[i], fs, 0) + \
               complex(0, 1) * sinesignal.generate_sine_wave_no(N, A, fd[i], fs, 0)

lmbda_0 = M * abs(noise.max())

# construct a 5Hz sine signal
s_clean = np.zeros(N)

s_clean = sinesignal.generate_sine_wave_no(N, A, 5, fs, 0) \
          + complex(0, 1) * sinesignal.generate_sine_wave_no(N, A, 5, fs, 0)

s_noise = s_clean + noise

s_clean = s_clean[:, np.newaxis]
s_noise = s_noise[:, np.newaxis]

Maxiter = 200
opt_par = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': Maxiter,
                                  'RelStopTol': 1e-4, 'AuxVarObj': False,
                                  'AutoRho': {'Enabled': True}})

# solve with real valued signal with real valued dictionary
b_r = cbpdn.ConvBPDN(D0.real, s_noise.real, lmbda_0, opt=opt_par, dimK=None, dimN=dimN)
x_r = b_r.solve()
fig, ax = plot.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(18, 8))

rec_r = b_r.reconstruct().squeeze()
fig.suptitle('real value solver ' + str(np.count_nonzero(x_r) * 100 / x_r.size) + '% non zero elements', fontsize=14)
plot.plot(s_clean.real, title='clean(real)', fig=fig, ax=ax[0, 0])
plot.plot(s_clean.imag, title='clean(imag)', fig=fig, ax=ax[1, 0])

plot.plot(s_noise.real, title='corrupted(real)', fig=fig, ax=ax[0, 1])
plot.plot(s_noise.imag, title='corrupted(imag)', fig=fig, ax=ax[1, 1])

plot.plot(rec_r.real, title='reconstructed(real)', fig=fig, ax=ax[0, 2])
plot.plot(rec_r.imag, title='reconstructed(imag)', fig=fig, ax=ax[1, 2])

fig.show()
# solve with a complex valued signal with complex valued dictionary
b_c = comcbpdn.ComplexConvBPDN(D0, s_noise, lmbda_0, opt_par, None, dimN)
x_c = b_c.solve()
rec_c = b_c.reconstruct().squeeze()

fig, ax = plot.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(18, 8))
fig.suptitle('complex value solver ' + str(np.count_nonzero(x_c) * 100 / x_c.size) + '% non zero elements', fontsize=14)

plot.plot(s_clean.real, title='clean(real)', fig=fig, ax=ax[0, 0])
plot.plot(s_clean.imag, title='clean(imag)', fig=fig, ax=ax[1, 0])

plot.plot(s_noise.real, title='corrupted(real)', fig=fig, ax=ax[0, 1])
plot.plot(s_noise.imag, title='corrupted(imag)', fig=fig, ax=ax[1, 1])

plot.plot(rec_c.real, title='reconstructed(real)', fig=fig, ax=ax[0, 2])
plot.plot(rec_c.imag, title='reconstructed(imag)', fig=fig, ax=ax[1, 2])
fig.show()

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
axs[0].scatter(s_clean.real,s_clean.imag)
axs[0].set_title('clean signal')
axs[0].set_xlabel('real axis')
axs[0].set_ylabel('imaginary axis')

axs[1].scatter(s_noise.real,s_noise.imag)
axs[1].set_title('corrupted signal')
axs[1].set_xlabel('real axis')
axs[1].set_ylabel('imaginary axis')

axs[2].scatter(rec_c.real,rec_c.imag)
axs[2].set_title('reconstructed signal')
axs[2].set_xlabel('real axis')
axs[2].set_ylabel('imaginary axis')

plt.show()

