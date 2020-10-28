# -*- coding: utf-8 -*-
# @Time    : 2020-10-22 8:12 p.m.
# @Author  : young wang
# @FileName: complex_cdl.py
# @Software: PyCharm
"""
The script solves complex convolutional dictionary learning problem using
with ADMM consensus framework for the update dictionary step and complex CBPDN
for the sparse encoding step

The input signal is 128 points long gaussine pulse trians at 500
different centre frequencies: 200 independent 128 points long  gaussine  wave.

The initial dictionary is random number. After the dictionary learning
the updated dictionary should be  gaussine pulse

input signal layout: [128x1x1x100], 200 independent 128 points long  gaussine  pulse.
ditctionary layput: [128x1x1x5], 5 filters with 128 points long

"""


import numpy as np
from sporco.admm import comcbpdn, cbpdn,comccmod
from scipy import signal
from sporco import signal as si
from tqdm import tqdm
from sporco import fft
from sporco import plot
from sporco.dictlrn import dictlrn
from sporco import cnvrep

from matplotlib import pyplot as plt

import random
from numpy.fft import fft, fft2, ifft, ifft2

np.seterr(divide='ignore')

N = 128  # signal length

A = 5  # amplitude
M = 10  # filter number
fd = np.linspace(5, 20, 20*M)
fs = 128  # sampling frequency must be higher enough to eliminate cutoff

# create a complex gaussian noise using sporco.signal.complex_randn,
# same length as input signal
sigma = 0.05  # noise level
noise = sigma*si.complex_randn(N)

t = np.linspace(-1, 1, 128, endpoint=False)

#the first element is the length of signal, the fourth one is how

S = np.zeros((N,1,1,len(fd)),dtype=complex)
SC = np.zeros((N,1,1,len(fd)),dtype=complex)
# construct sine/cosine signal
for i in range(len(fd)):

    real,imag,e = signal.gausspulse(t, 5, retquad=True, retenv=True)
    temp = real + complex(0, 1) * imag
    SC[:,:,:,i] = np.reshape(temp,(-1,1,1))

    temp1 = real+complex(0,1)*imag + noise
    S[:, :, :, i] = np.reshape(temp1,(-1,1,1))

random.seed(10)
D0 = si.complex_randn(128,1,1,5)
#
cri = cnvrep.CDU_ConvRepIndexing(D0.shape, S)
#
lmbda = 0.4
optx = comcbpdn.ComplexConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1,
            'rho': 50.0*lmbda + 0.5,'AuxVarObj': False})
#
optd = comccmod.ComConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
            'rho': 10, 'ZeroMean': True},
            method='cns')
#
#Dictionary support projection and normalisation (cropped).
#Normalise dictionary according to dictionary Y update options.

D0n = cnvrep.Pcn(D0, D0.shape, cri.Nv, dimN=2, dimC=0, crp=True,
                 zm=optd['ZeroMean'])
#
# Update D update options to include initial values for Y and U.
optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
             'U0': np.zeros(cri.shpD + (cri.K,))})
# #
#Create X update object.
xstep = comcbpdn.ComplexConvBPDN(D0n, S, lmbda, optx)

# # the first one is coefficient map
#Create D update object.
dstep = comccmod.ComConvCnstrMOD(None, S, D0.shape, optd, method='cns')
#
opt = dictlrn.DictLearn.Options({'Verbose': True, 'MaxMainIter': 500})
d = dictlrn.DictLearn(xstep, dstep, opt)
D1 = d.solve()
x = d.getcoef()
print("DictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")

itsx = xstep.getitstat()
itsd = dstep.getitstat()

fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(np.abs(itsx.ObjFun), xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((np.abs(itsx.PrimalRsdl), np.abs(itsx.DualRsdl), np.abs(itsd.PrimalRsdl),
          np.abs(itsd.DualRsdl))).T, ptyp='semilogy', xlbl='Iterations',
          ylbl='Residual', lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual'],
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(np.vstack((np.abs(itsx.Rho), np.abs(itsd.Rho))).T,  xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy', lgnd=['Rho', 'Sigma'],
          fig=fig)
fig.show()

Maxiter = 500
#run with bad dictionary
opt_par = comcbpdn.ComplexConvBPDN.Options({'FastSolve': True, 'Verbose': True,
                               'StatusHeader': False, 'MaxMainIter': Maxiter,
            'rho': 50.0*lmbda + 0.5,'AuxVarObj': False})

b_b = comcbpdn.ComplexConvBPDN(D0, S,  lmbda, opt=opt_par)
x_b = b_b.solve()
rec_b = b_b.reconstruct().squeeze()

b_g = comcbpdn.ComplexConvBPDN(D1, S,  lmbda, opt=opt_par)
x_g = b_g.solve()
rec_g = b_g.reconstruct().squeeze()

fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(16,9))
fig.suptitle('dictionary performance comparison',fontsize=14)

ax[0,0].set_title('noisy input: real part')
S_d = S.squeeze()
ax[0,0].plot(S_d[:,5].real)
#
ax[1,0].set_title('noisy input: imag part')
ax[1,0].plot(S_d[:,5].imag)

ax[0,1].set_title('reconstruction: initial dictionary: real')
ax[0,1].plot(rec_b[:,5].real)

ax[1,1].set_title('reconstruction: initial dictionary: imag')
ax[1,1].plot(rec_b[:,5].imag)

ax[0,2].set_title('reconstruction: learned dictionary: real')
ax[0,2].plot(rec_g[:,5].real)

ax[1,2].set_title('reconstruction: learned dictionary: imag')
ax[1,2].plot(rec_g[:,5].imag)

plt.show()

fig,ax = plt.subplots(nrows= 5, ncols=4, figsize=(16,9))
fig.suptitle('initial & learned dictionaries',fontsize=14)

D0_d = D0.squeeze()
D1_d = D1.squeeze()

for i in range(5):
    ax[i, 0].plot(D0_d[:, i].real)
    ax[i, 0].set_title('initial dictionary(real): filter '+str(i))

    ax[i, 1].plot(D0_d[:, i].imag)
    ax[i, 1].set_title('initial dictionary(imag): filter '+str(i))

    ax[i, 2].plot(D1_d[:, i].real)
    ax[i, 2].set_title('learned dictionary(real): filter ' + str(i))

    ax[i, 3].plot(D1_d[:, i].imag)
    ax[i, 3].set_title('learned dictionary(imag): filter ' + str(i))

plt.show()
