from __future__ import division
from builtins import object

import pickle
import numpy as np

from sporco.admm import comcbpdn
from sporco.linalg import rrs
from sporco.fft import fftn, ifftn, fftconv



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda, dimK=0)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_02(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_03(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_04(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_05(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 0
        assert b.cri.dimK == 1


    def test_06(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.complex
        opt = comcbpdn.ComplexConvBPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'AutoRho': {'Enabled': True},
                                      'DataType': dt})
        lmbda = 1e-1
        b = comcbpdn.ComplexConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt
