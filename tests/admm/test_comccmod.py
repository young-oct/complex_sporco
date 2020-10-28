from builtins import object

import numpy as np

import sporco.cnvrep as cr
from sporco.admm import comccmod
from sporco.linalg import rrs
from sporco.fft import fftn, ifftn



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    # def test_01(self):
    #     N = 32
    #     M = 4
    #     Nd = 5
    #     D0 = cr.normalise(cr.zeromean(
    #         np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
    #     X = np.zeros((N, N, M),dtype=complex)
    #     xr = np.random.randn(N, N, M)+complex(0,1)*np.random.randn(N, N, M)
    #     xp = np.abs(xr) > 3
    #     X[xp] = np.random.randn(X[xp].size)
    #     S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) *
    #                fftn(X, None, (0, 1)), None, (0, 1)), axis=2)
    #     rho = 1e-1
    #     opt = comccmod.ComConvCnstrMOD_IterSM.Options({'Verbose': False,
    #                 'MaxMainIter': 500, 'LinSolveCheck': True,
    #                 'ZeroMean': True, 'RelStopTol': 1e-5, 'rho': rho,
    #                 'AutoRho': {'Enabled': False}})
    #     Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
    #     Sr = S.reshape(S.shape + (1,))
    #     c = comccmod.ComConvCnstrMOD_IterSM(Xr, Sr, D0.shape, opt)
    #     c.solve()
    #     D1 = cr.bcrop(c.Y, D0.shape).squeeze()
    #     assert rrs(D0, D1) < 1e-5
    #     assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5
    # #
    #
    # def test_03(self):
    #     N = 64
    #     M = 4
    #     Nd = 8
    #     D0 = cr.normalise(cr.zeromean(
    #         np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
    #     X = np.zeros((N, N, M))
    #     xr = np.random.randn(N, N, M)
    #     xp = np.abs(xr) > 3
    #     X[xp] = np.random.randn(X[xp].size)
    #     S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) *
    #                fftn(X, None, (0, 1)), None, (0, 1)).real, axis=2)
    #     rho = 1e1
    #     opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                 'MaxMainIter': 500, 'LinSolveCheck': True,
    #                 'ZeroMean': True, 'RelStopTol': 1e-3, 'rho': rho,
    #                 'AutoRho': {'Enabled': False}})
    #     Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
    #     Sr = S.reshape(S.shape + (1,))
    #     c = comccmod.ComConvCnstrMOD_Consensus(Xr, Sr, D0.shape, opt)
    #     c.solve()
    #     D1 = cr.bcrop(c.Y, D0.shape).squeeze()
    #     assert rrs(D0, D1) < 1e-5
    #     assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5
    #
    #
    def test_04(self):
        N = 16
        M = 4
        Nd = 8
        X = complex(1,1)*np.random.randn(N, N, 1, 1, M)
        S = complex(1,1)*np.random.randn(N, N, 1)
        try:
            c = comccmod.ComConvCnstrMOD(X, S, (Nd, Nd, M))
            c.solve()
        except Exception as e:
            print(e)
            assert 0

    #
    # def test_05(self):
    #     N = 16
    #     M = 8
    #     X = np.random.randn(N, N, 1, 1, M)
    #     S = np.random.randn(N, N, 1)
    #     try:
    #         c = comccmod.ComConvCnstrMOD(X, S, ((4, 4, 4), (8, 8, 4)))
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_06(self):
    #     N = 16
    #     M = 4
    #     Nc = 3
    #     Nd = 8
    #     X = np.random.randn(N, N, Nc, 1, M)
    #     S = np.random.randn(N, N, Ncx)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD(X, S, (Nd, Nd, 1, M), opt=opt, dimK=0)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #     assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5
    #
    #
    # def test_07(self):
    #     N = 16
    #     M = 4
    #     Nd = 8
    #     X = np.random.randn(N, N, 1, 1, M)
    #     S = np.random.randn(N, N, 1)
    #     dt = np.complex
    #     opt = ccmod.ConvCnstrMODOptions(
    #         {'Verbose': False, 'MaxMainIter': 20,
    #          'AutoRho': {'Enabled': True},
    #          'DataType': dt})
    #     c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
    #     c.solve()
    #     assert c.X.dtype == dt
    #     assert c.Y.dtype == dt
    #     assert c.U.dtype == dt
    #
    #
    # def test_08(self):
    #     N = 16
    #     M = 4
    #     Nd = 8
    #     X = np.random.randn(N, N, 1, 1, M)
    #     S = np.random.randn(N, N, 1)
    #     dt = np.complex
    #     opt = comccmod.ComConvCnstrMODOptions(
    #         {'Verbose': False, 'MaxMainIter': 20,
    #          'AutoRho': {'Enabled': True},
    #          'DataType': dt})
    #     c = comccmod.ComConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
    #     c.solve()
    #     assert c.X.dtype == dt
    #     assert c.Y.dtype == dt
    #     assert c.U.dtype == dt
    #
    #
    # def test_09(self):
    #     N = 16
    #     M = 4
    #     Nd = 8
    #     X = np.random.randn(N, N, 1, 1, M)
    #     S = np.random.randn(N, N)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
    #                                          opt=opt, dimK=0)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_10(self):
    #     N = 16
    #     K = 3
    #     M = 4
    #     Nd = 8
    #     X = np.random.randn(N, N, 1, K, M)
    #     S = np.random.randn(N, N, K)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M), opt=opt)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_11(self):
    #     N = 16
    #     M = 4
    #     Nc = 3
    #     Nd = 8
    #     X = np.random.randn(N, N, Nc, 1, M)
    #     S = np.random.randn(N, N, Nc)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
    #                                          opt=opt, dimK=0)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_12(self):
    #     N = 16
    #     M = 4
    #     K = 2
    #     Nc = 3
    #     Nd = 8
    #     X = np.random.randn(N, N, Nc, K, M)
    #     S = np.random.randn(N, N, Nc, K)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
    #                                          opt=opt)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_13(self):
    #     N = 16
    #     M = 4
    #     K = 2
    #     Nc = 3
    #     Nd = 8
    #     X = np.random.randn(N, N, Nc, K, M)
    #     S = np.random.randn(N, N, Nc, K)
    #     try:
    #         opt = comccmod.ComConvCnstrMOD_Consensus.Options({'Verbose': False,
    #                         'MaxMainIter': 20, 'LinSolveCheck': True})
    #         c = comccmod.ComConvCnstrMOD_Consensus(X, S, (Nd, Nd, Nc, M),
    #                                          opt=opt)
    #         c.solve()
    #     except Exception as e:
    #         print(e)
    #         assert 0
    #
    #
    # def test_14(self):
    #     opt = comccmod.ComConvCnstrMOD_Consensus.Options({'AuxVarObj': False})
    #     assert opt['fEvalX'] is True and opt['gEvalY'] is False
    #     opt['AuxVarObj'] = True
    #     assert opt['fEvalX'] is False and opt['gEvalY'] is True
    #
    #
    # def test_15(self):
    #     opt = comccmod.ComConvCnstrMOD_Consensus.Options({'AuxVarObj': True})
    #     assert opt['fEvalX'] is False and opt['gEvalY'] is True
    #     opt['AuxVarObj'] = False
    #     assert opt['fEvalX'] is True and opt['gEvalY'] is False
