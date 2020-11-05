# -*- coding: utf-8 -*-
# @Time    : 2020-11-04 9:19 p.m.
# @Author  : young wang
# @FileName: complex_cupy.py
# @Software: PyCharm

"""
The script solves convolutional dictionary learning problem using
with ADMM consensus framework for the update dictionary step and CBPDN
for the sparse encoding step

The input signal is 468 points long gaussian pulse trains at different locations
1000 independent 468 points long gaussian pulse trains

The initial dictionary is random number. After the dictionary learning
the updated dictionary should be sine waves

The input signal is changed to simulate middle ear structure, a big peak as TM and a small
peak as incus. The purpose of the script is that (1) learn how to correctly display data from
sparse representation (2) identify small peak from noise corrupted signals

TM(tympanic membrane) is represented by an amplitude of 1 Gaussian pulse
Malleus(?) and incus are represented by an amplitude of 0.3 & 0.15 Gaussian pulse.
Only 20 signals contain malleus(?) and incus, and each one of the 20 signals are coupled
with a weight factor (normal distribution). This is to mimic the shape of those structure in
terms of their amplitude.

The OCT noise is assumed to be an additive Gaussian complex white noise
"Sangmin Kim, John S. Oghalai, and Brian E. Applegate, "Noise and sensitivity in
optical coherence tomography based vibrometry," Opt. Express 27, 33333-33350 (2019)
"

This script demonstrates a standard pipeline in processing OCT data with convolutional
basis pursuit denoising technique
(1) load data(omit here)
(2) normalize input data and dictionary
(3) regularization parameter search
(4) solve sparse representation with initial dictionary
(5) update dictionary
(6) solve sparse representation with learned dictionary
(7) linearly shift sparse representation

steps are not included here:
(8) calculate every 20 lines to form 1 line in the B mode image.

To note, both 1D and 2D method gives the same results, yet
To process 1 line, 1D takes 0.210266 seconds, 1000 lines would be 210.266
To process 1000 line, 2D takes 16.710963 seconds.

2D would be 12x faster

"""

from sporco.admm import comcbpdn, cbpdn,comccmod
from scipy import signal
from sporco import signal as si
from sporco import plot,cnvrep,mpiutil
from sporco.dictlrn import dictlrn

from matplotlib import pyplot as plt
import numpy as np
import random
from numpy.fft import fft, fft2, ifft, ifft2,fftn,ifftn
from pytictoc import TicToc

from sporco import mpiutil
from sporco import fft
from sporco import metric
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import cbpdn
import cupy as cp

#
#
# np.seterr(divide='ignore', invalid='ignore')
#
# def reconstruct(D,x,cri):
#     Df = np.fft.fftn(D, cri.Nv, cri.axisN)
#     Xf = np.fft.fftn(x, None, cri.axisN)
#     Sf = np.sum(Df * Xf, axis=cri.axisM)
#
#     return np.fft.ifftn(Sf, cri.Nv, cri.axisN)
#
# def evalerr(prm):
#     dimN=1
#     lmbda = prm[0]
#     b = comcbpdn.ComplexConvBPDN(D0_d, s_test, lmbda, opt=opt_par, dimK=None, dimN=dimN)
#     x_0 = b.solve()
#     x_1 = x_0.squeeze()
#
#     return (np.sum(np.abs(x_1)))
#
# def evalerr_l(prm):
#     dimN=1
#     lmbda_l= prm[0]
#     b_l = comcbpdn.ComplexConvBPDN(D1_d, s_test, lmbda_l, opt=opt_par, dimK=None, dimN=dimN)
#     x_0_l = b_l.solve()
#     x_1_l = x_0_l.squeeze()
#     return (np.sum(np.abs(x_1_l)))

plt.close('all')

N = 468  # signal length
M = 2 # filter number
K = 5000 # test signal number

t = np.linspace(-45, 45, N, endpoint=False)

#the first element is the length of signal, the fourth one is how

s_noise = np.zeros((N,K))
s_clean = np.zeros((N,K))
Dtemp = np.zeros((N,K))
# delta matrix to shift the Gaussian pulses
delta = np.zeros((N))
delta[-120] = 1

delta_1 = np.zeros((N))
delta_1[-70] = 1

alpha = signal.gaussian(200, std=30)
# construct sine/cosine signal
for i in range(K):

    sigma = 0.02  # noise level
    noise = sigma *np.random.randn(N)
    # delta[np.random.randint(0, 32), i] = 1
    real_0,imag_0 = signal.gausspulse(t, 4.7, retquad=True, retenv=False)

    real_1 = np.convolve(delta,real_0,'same')
    # imag_1 = np.convolve(delta, imag_0,'same')

    real_2 = np.convolve(delta_1, real_0, 'same')
    # imag_2 = np.convolve(delta_1, imag_0, 'same')

    incus_real = 0.5 * real_1+0.3*real_2
    # incus_imag = 0.5 * imag_1+0.3*imag_2

    TM = real_0
    incus = incus_real

    # temp = np.convolve(temp,delta[:,i],'same')

    if 500 < i < 700:
        s_clean[:,i] = TM + incus*alpha[500-i]
        s_noise[:,i] = s_clean[:,i] + noise
    else:
        s_clean[:, i] = TM
        s_noise[:, i] = s_clean[:, i] + noise

    Dtemp[:,i] = TM + noise

Nv = 650 # sample line index

# plt.subplot(131)
# plt.plot(s_clean[:,Nv].real)
# plt.title('true structure')
#
# plt.subplot(132)
# plt.plot(s_noise[:,Nv].real)
# plt.title('structure with noisy')
#
# plt.subplot(133)
# plt.plot(Dtemp[:,Nv].real)
# plt.title('initial dictionary')
# plt.show()
# randomly pick 100 signals from 1000 input signals as training set
train_index = np.random.choice(s_noise.shape[1],int(0.1*K))
s_train = s_noise[:,train_index]

D0 =Dtemp[:,0:M]

# Function computing reconstruction error at lmbda
Maxiter = 500
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                            'MaxMainIter': Maxiter,'RelStopTol': 5e-5, 'AuxVarObj': True,
                                            'RelaxParam': 1.515,'AutoRho': {'Enabled': True}})

s_clean = np.reshape(s_clean, (-1,1,K))
s_noise = np.reshape(s_noise, (-1,1,K))
s_train = np.reshape(s_train, (-1,1,len(train_index)))

s_test = s_noise[:,:,Nv].squeeze()
s_test = s_test[:,np.newaxis]

D0 = np.reshape(D0,(-1,1,M))
D0_d = D0.squeeze()

lmbda = 0.01
# calculate sparse representation with initial, learned and ground truth dictionaries as 1D signal
b_i = cbpdn.ConvBPDN(np2cp(D0), np2cp(s_train), lmbda, opt=opt_par, dimK=1, dimN=1)
x_i = cp2np(b_i.solve())
rec_i = b_i.reconstruct().squeeze()
its_i = b_i.getitstat()
# #
# t.tic()
# # for i in range(s_noise_d.shape[1]):
# b_l = comcbpdn.ComplexConvBPDN(D1_d, s_noise_d[:,Nv], lmbda*Kr, opt=opt_par, dimK=None, dimN=1)
# x_l = b_l.solve()
# rec_l = b_l.reconstruct().squeeze()
# its_i = b_l.getitstat()
# t.toc('1D Elapsed time is ',restart=True)
# # 200 lines 9.443324 seconds.
#
# # 2D method
#
# b_i_2d = comcbpdn.ComplexConvBPDN(D0_d, s_noise_d, lmbda*Kr, opt=opt_par, dimK=1, dimN=1)
# x_i_2d = b_i_2d.solve()
# rec_i_2d = b_i_2d.reconstruct().squeeze()
# its_i_2d = b_i_2d.getitstat()
#
# if not cupy_enabled():
#     print('CuPy/GPU device not available: running without GPU acceleration\n')
# else:
#     id = select_device_by_load()
#     info = gpu_info()
#     if info:
#         print('Running on GPU %d (%s)\n' % (id, info[id].name))
#
# # t.tic()
# # D1_2d = np.reshape(D1,(-1,1,M))
# # b_l_2d = comcbpdn.ComplexConvBPDN(D1_d, s_noise_d, lmbda*Kr, opt=opt_par, dimK=1, dimN=1)
# # x_l_2d = b_l_2d.solve()
# # rec_l_2d = b_l_2d.reconstruct().squeeze()
# # its_l_2d = b_l_2d.getitstat()
# # t.toc('2D Elapsed time is ',restart=True)
# # # 0.439353 seconds.
#
# # t.tic()
# # D1_2d_cp = np.reshape(D1,(-1,1,M))
# # b_l_2d_cp = comcbpdn.ComplexConvBPDN(np2cp(D1_d), np2cp(s_noise_d), lmbda*Kr, opt=opt_par, dimK=1, dimN=1)
# # x_l_2d = cp2np(b_l_2d.solve())
# # rec_l_2d = b_l_2d.reconstruct().squeeze()
# # its_l_2d = b_l_2d.getitstat()
# # t.toc('cupy 2D Elapsed time is ',restart=True)
# # # 0.439353 seconds.
#
#
#
#
#
# fig,ax = plt.subplots(nrows= M, ncols=6, figsize=(16,9))
# fig.suptitle('initial & learned dictionaries',fontsize=14)
#
# for i in range(M):
#     ax[i, 0].plot(D0_d[:, i].real)
#     ax[i, 0].set_title('initial dictionary(real): filter '+str(i))
#
#     ax[i, 1].plot(D0_d[:, i].imag)
#     ax[i, 1].set_title('initial dictionary(imag): filter '+str(i))
#
#     ax[i, 2].plot(D1_d[:, i].real)
#     ax[i, 2].set_title('learned dictionary(real): filter ' + str(i))
#
#     ax[i, 3].plot(D1_d[:, i].imag)
#     ax[i, 3].set_title('learned dictionary(imag): filter ' + str(i))
#
#     ax[i, 4].plot(s_clean_d[:, i].real)
#     ax[i, 4].set_title('true dictionary(real): filter ' + str(i))
#
#     ax[i, 5].plot(s_clean_d[:, i].imag)
#     ax[i, 5].set_title('true dictionary(imag): filter ' + str(i))
#
# plt.show()
# #
# # fig,ax = plt.subplots(nrows=2, ncols=6, figsize=(16,9))
# # fig.suptitle('1D versus 2D processing method',fontsize=14)
# #
# # ax[0,0].set_title('signal: real')
# # ax[0,0].plot(s_clean_d[:,Nv].real)
# #
# # ax[1,0].set_title('signal: imag')
# # ax[1,0].plot(s_clean_d[:,Nv].imag)
# #
# # ax[0,1].set_title('signal: real')
# # ax[0,1].plot(s_noise_d[:,Nv].real)
# #
# # ax[1,1].set_title('signal: imag')
# # ax[1,1].plot(s_noise_d[:,Nv].imag)
# #
# # ax[0,2].set_title('initial: 1D real')
# # ax[0,2].plot(rec_i.real)
# #
# # ax[1,2].set_title('initial: 1D imag')
# # ax[1,2].plot(rec_i.imag)
# #
# # ax[0,3].set_title('learned: 1D real')
# # ax[0,3].plot(rec_l.real)
# #
# # ax[1,3].set_title('learned: 1D imag')
# # ax[1,3].plot(rec_l.imag)
# #
# # ax[0,4].set_title('initial: 2D real')
# # ax[0,4].plot(rec_i_2d[:,Nv].real)
# #
# # ax[1,4].set_title('initial: 2D imag')
# # ax[1,4].plot(rec_i_2d[:,Nv].imag)
# #
# # ax[0,5].set_title('learned: 2D real')
# # ax[0,5].plot(rec_l_2d[:,Nv].real)
# #
# # ax[1,5].set_title('learned: 2D imag')
# # ax[1,5].plot(rec_l_2d[:,Nv].imag)
# #
# # plt.show()
#
# x_l_2d = np.sum(x_l_2d, axis=b_l_2d.cri.axisM).squeeze()
# x_l_2d = np.roll(x_l_2d, np.argmax(D1_d[:,0]), axis=0)
#
# x_l = np.sum(x_l, axis=b_l.cri.axisM).squeeze()
# x_l = np.roll(x_l, np.argmax(D1_d[:,0]), axis=0)
#
# fig,ax = plt.subplots(nrows=1, ncols=6, figsize=(16,9))
# ax[0].set_title('true structure')
# ax[0].plot(s_clean_d[:,Nv].real)
#
# ax[1].set_title('structure with noise')
# ax[1].plot(s_noise_d[:,Nv].real)
#
# ax[2].set_title('reconstruction from 2D')
# ax[2].plot(rec_l_2d[:,Nv].real)
#
# ax[3].set_title('sparse representation from 2D')
# ax[3].plot(x_l_2d[:,Nv].real)
#
# ax[4].set_title('sparse representation from 1D')
# ax[4].plot(rec_l.real)
#
# ax[5].set_title('sparse representation from 1D')
# ax[5].plot(x_l.real)
#
# plt.show()
