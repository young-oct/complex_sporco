# -*- coding: utf-8 -*-
# @Time    : 2020-10-22 8:12 p.m.
# @Author  : young wang
# @FileName: convolution_dictionary.py
# @Software: PyCharm
"""
The script solves convolutional dictionary learning problem using
with ADMM consensus framework for the update dictionary step and CBPDN
for the sparse encoding step

The input signal is 64 points long gaussian pulse trains at different locations
1000 independent 64 points long gaussian pulse trains

The initial dictionary is random number. After the dictionary learning
the updated dictionary should be sine waves
"""

import numpy as np
from sporco.admm import comcbpdn, cbpdn,comccmod
from scipy import signal
from sporco import signal as si
from sporco import plot,cnvrep
from sporco.dictlrn import dictlrn
from sporco import mpiutil

from matplotlib import pyplot as plt

import random
from numpy.fft import fft, fft2, ifft, ifft2,fftn,ifftn



np.seterr(divide='ignore', invalid='ignore')

def reconstruct(D,x,cri):
    Df = np.fft.fftn(D, cri.Nv, cri.axisN)
    Xf = np.fft.fftn(x, None, cri.axisN)
    Sf = np.sum(Df * Xf, axis=cri.axisM)

    return np.fft.ifftn(Sf, cri.Nv, cri.axisN)

def evalerr(prm):
    dimN=1
    lmbda = prm[0]
    b = comcbpdn.ComplexConvBPDN(D0, s_test, lmbda, opt=opt_par, dimK=None, dimN=dimN)
    x_0 = b.solve()
    x_1 = x_0.squeeze()

    return (np.sum(np.abs(x_1)))

def evalerr_l(prm):
    dimN=1
    lmbda_l= prm[0]
    b_l = comcbpdn.ComplexConvBPDN(D1_d, s_test, lmbda_l, opt=opt_par, dimK=None, dimN=dimN)
    x_0_l = b_l.solve()
    x_1_l = x_0_l.squeeze()
    return (np.sum(np.abs(x_1_l)))

N = 64  # signal length
M = 5 # filter number
K = 20 * M # input signal number
t = np.linspace(-2, 2, N, endpoint=False)

#the first element is the length of signal, the fourth one is how

s_noise = np.zeros((N,K), dtype=complex)
s_clean = np.zeros((N,K), dtype=complex)
Dtemp = np.zeros((N,K), dtype=complex)

# delta matrix to shift the Gaussian pulses
delta = np.zeros((N, K), dtype=complex)

# construct sine/cosine signal
for i in range(K):

    sigma = 0.05  # noise level
    noise = sigma * si.complex_randn(N)
    delta[np.random.randint(0, 32), i] = 1

    real,imag,e = signal.gausspulse(t, 5, retquad=True, retenv=True)
    temp = real + complex(0, 1) * imag
    temp = np.convolve(temp,delta[:,i],'same')

    s_clean[:,i] = temp
    s_noise[:,i] = s_clean[:,i] + noise
    Dtemp[:,i] = s_clean[:,i] + 10*noise

D0 =Dtemp[:,0:M]
# Function computing reconstruction error at lmbda
Maxiter = 400
opt_par = comcbpdn.ComplexConvBPDN.Options({'FastSolve': True, 'Verbose': True, 'StatusHeader': False,
                                            'MaxMainIter': Maxiter,'RelStopTol': 5e-5, 'AuxVarObj': True,
                                            'RelaxParam': 1.515,'AutoRho': {'Enabled': True}})

s_clean = np.reshape(s_clean, (-1,1,1,K))
s_noise = np.reshape(s_noise, (-1,1,1,K))

s_test = s_noise[:,:,:,0].squeeze()
s_test = s_test[:,np.newaxis]

D0 = np.reshape(D0,(-1,1,1,M))

# Parallel evaluation of error function on lmbda grid

# lrng = np.logspace(-5,3,1000)
# sprm, sfvl, fvmx, sidx = mpiutil.grid_search(evalerr, (lrng,))
# lmbda = sprm[0]
# print('Minimum ℓ1 error: %5.2f at 𝜆 = %.2e' % (sfvl, lmbda))
# fig, ax = plot.subplots(figsize=(19.5, 8))
# plot.plot(fvmx, x=lrng, ptyp='semilogx',title='original 𝜆 = %.2e' % lmbda, xlbl='$\lambda$',
#           ylbl='Error', fig=fig)
# fig.show()
lmbda = 1.55

cri = cnvrep.CDU_ConvRepIndexing(D0.shape, s_noise)

optx = comcbpdn.ComplexConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1,
            'rho': 8.13e+01,'AuxVarObj': False})

optd = comccmod.ComConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
            'rho': 10, 'ZeroMean': True},
            method='cns')
#
#Dictionary support projection and normalisation (cropped).
#Normalise dictionary according to dictionary Y update options.

D0n = cnvrep.Pcn(D0, D0.shape, cri.Nv, dimN=1, dimC=0, crp=True,
                 zm=optd['ZeroMean'])
#
# Update D update options to include initial values for Y and U.
optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
             'U0': np.zeros(cri.shpD + (cri.K,))})
# # #
#Create X update object.
xstep = comcbpdn.ComplexConvBPDN(D0n, s_noise, lmbda, optx)

# # the first one is coefficient map
#Create D update object.
dstep = comccmod.ComConvCnstrMOD(None, s_noise, D0.shape, optd, method='cns')
#
opt = dictlrn.DictLearn.Options({'Verbose': True, 'MaxMainIter':1000})
d = dictlrn.DictLearn(xstep, dstep, opt)
D1 = d.solve()
x = d.getcoef()

rec = reconstruct(D1,x,cri)
rec = rec.squeeze()
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

# s_test = s_noise[:,:,:,np.random.randint(0,K)].squeeze()

D0_d = D0.squeeze()
D1_d = D1.squeeze()
s_clean_d = s_clean.squeeze()
s_noise_d = s_noise.squeeze()

opt_par['FastSolve'] = False
opt_par['Verbose'] = False

# calculate sparse representation with initial, learned and ground truth dictionaries
b_i = comcbpdn.ComplexConvBPDN(D0_d, s_test, lmbda / 2, opt=opt_par, dimK=None, dimN=1)
x_i = b_i.solve()
rec_i = b_i.reconstruct().squeeze()
its_i = b_i.getitstat()

b_l = comcbpdn.ComplexConvBPDN(D1_d, s_test, lmbda / 2, opt=opt_par, dimK=None, dimN=1)
x_l = b_l.solve()
rec_l = b_l.reconstruct().squeeze()
its_l = b_l.getitstat()

b_t = comcbpdn.ComplexConvBPDN(s_clean_d[:, 0:M], s_test, lmbda / 2, opt=opt_par, dimK=None, dimN=1)
x_t = b_t.solve()
rec_t = b_t.reconstruct().squeeze()
its_t = b_t.getitstat()

fig,ax = plt.subplots(nrows=2, ncols=5, figsize=(16,9))
fig.suptitle('dictionary performance comparison',fontsize=14)

ax[0,0].set_title('ground truth:real')
ax[0,0].plot(s_clean_d[:,0].real)
#
ax[1,0].set_title('ground truth:imag')
ax[1,0].plot(s_clean_d[:,0].imag)

ax[0,1].set_title('noisy input: real part')
ax[0,1].plot(s_test.real)
#
ax[1,1].set_title('noisy input: imag part')
ax[1,1].plot(s_test.imag)

ax[0,2].set_title('reconstruction: initial dictionary: real')
ax[0,2].plot(rec_i.real)

ax[1,2].set_title('reconstruction: initial dictionary: imag')
ax[1,2].plot(rec_i.imag)

ax[0,3].set_title('reconstruction: learned dictionary: real')
ax[0,3].plot(rec_l.real)

ax[1,3].set_title('reconstruction: learned dictionary: imag')
ax[1,3].plot(rec_l.imag)

ax[0,4].set_title('reconstruction: true dictionary: real')
ax[0,4].plot(rec_t.real)

ax[1,4].set_title('reconstruction: true dictionary: imag')
ax[1,4].plot(rec_t.imag)

plt.show()

fig,ax = plt.subplots(nrows= M, ncols=6, figsize=(16,9))
fig.suptitle('initial & learned dictionaries',fontsize=14)


for i in range(M):
    ax[i, 0].plot(D0_d[:, i].real)
    ax[i, 0].set_title('initial dictionary(real): filter '+str(i))

    ax[i, 1].plot(D0_d[:, i].imag)
    ax[i, 1].set_title('initial dictionary(imag): filter '+str(i))

    ax[i, 2].plot(D1_d[:, i].real)
    ax[i, 2].set_title('learned dictionary(real): filter ' + str(i))

    ax[i, 3].plot(D1_d[:, i].imag)
    ax[i, 3].set_title('learned dictionary(imag): filter ' + str(i))

    ax[i, 4].plot(s_clean_d[:, i].real)
    ax[i, 4].set_title('true dictionary(real): filter ' + str(i))

    ax[i, 5].plot(s_clean_d[:, i].imag)
    ax[i, 5].set_title('true dictionary(imag): filter ' + str(i))

plt.show()

plt.plot(rec.real[:,0],label = 'rec')
plt.plot(rec_i.real, label = 'rec_i')
plt.plot(rec_l.real, label = 'rec_l')
plt.plot(rec_t.real, label = 'rec_t')
plt.title('comparison')
plt.legend()
plt.show()
x_b_plot = np.sum(x_i, axis=3).squeeze()

# x_g_plot = np.sum(x_l, axis=3).squeeze()
# x_a_plot = np.sum(x_t, axis=3).squeeze()
#
# plt.plot(x_b_plot.real,label = 'init X')
# plt.plot(x_g_plot.real, label = 'learned X')
# plt.plot(x_a_plot.real, label = 'true X')
# plt.title('sparse representation comparison')
# plt.legend()
# plt.show()

# # Parallel evaluation of error function on lmbda grid
# lrng_l = np.logspace(-5,3,1000)
# sprm_l, sfvl_l, fvmx_l, sidx_l = mpiutil.grid_search(evalerr_l, (lrng_l,))
# lmbda_l = sprm_l[0]
#
# fig, ax = plot.subplots(figsize=(19.5, 8))
# plot.plot(fvmx_l, x=lrng_l, ptyp='semilogx',title='learned 𝜆 = %.2e' % lmbda_l, xlbl='$\lambda$',
#           ylbl='Error', fig=fig)
#
# fig.show()
#
#
