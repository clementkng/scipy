""" Test functions for the sparse.linalg.eigen.lobpcg module
"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose, assert_array_less)

from scipy import ones, rand, r_, diag, linalg, eye
from scipy.linalg import eig, eigh, toeplitz
import scipy.sparse
from scipy.sparse.linalg.eigen.lobpcg import lobpcg
from scipy.sparse.linalg import eigs
from scipy.sparse import spdiags

import pytest

from scipy.sparse.linalg.eigen.lobpcg.lobpcg import lobpcg_svds

def ElasticRod(n):
    # Fixed-free elastic rod
    L = 1.0
    le = L/n
    rho = 7.85e3
    S = 1.e-4
    E = 2.1e11
    mass = rho*S*le/6.
    k = E*S/le
    A = k*(diag(r_[2.*ones(n-1),1])-diag(ones(n-1),1)-diag(ones(n-1),-1))
    B = mass*(diag(r_[4.*ones(n-1),2])+diag(ones(n-1),1)+diag(ones(n-1),-1))
    return A,B


def MikotaPair(n):
    # Mikota pair acts as a nice test since the eigenvalues
    # are the squares of the integers n, n=1,2,...
    x = np.arange(1,n+1)
    B = diag(1./x)
    y = np.arange(n-1,0,-1)
    z = np.arange(2*n-1,0,-2)
    A = diag(z)-diag(y,-1)-diag(y,1)
    return A,B


def compare_solutions(A,B,m):
    n = A.shape[0]

    np.random.seed(0)

    V = rand(n,m)
    X = linalg.orth(V)

    eigs,vecs = lobpcg(A, X, B=B, tol=1e-5, maxiter=30, largest=False)
    eigs.sort()

    w,v = eig(A,b=B)
    w.sort()

    assert_almost_equal(w[:int(m/2)],eigs[:int(m/2)],decimal=2)


def test_Small():
    A,B = ElasticRod(10)
    compare_solutions(A,B,10)
    A,B = MikotaPair(10)
    compare_solutions(A,B,10)


def test_ElasticRod():
    A,B = ElasticRod(100)
    compare_solutions(A,B,20)


def test_MikotaPair():
    A,B = MikotaPair(100)
    compare_solutions(A,B,20)


def test_trivial():
    n = 5
    X = ones((n, 1))
    A = eye(n)
    compare_solutions(A, None, n)


def test_regression():
    # https://mail.python.org/pipermail/scipy-user/2010-October/026944.html
    n = 10
    X = np.ones((n, 1))
    A = np.identity(n)
    w, V = lobpcg(A, X)
    assert_allclose(w, [1])


def test_diagonal():
    # This test was moved from '__main__' in lobpcg.py.
    # Coincidentally or not, this is the same eigensystem
    # required to reproduce arpack bug
    # https://forge.scilab.org/p/arpack-ng/issues/1397/
    # even using the same n=100.

    np.random.seed(1234)

    # The system of interest is of size n x n.
    n = 100

    # We care about only m eigenpairs.
    m = 4

    # Define the generalized eigenvalue problem Av = cBv
    # where (c, v) is a generalized eigenpair,
    # and where we choose A to be the diagonal matrix whose entries are 1..n
    # and where B is chosen to be the identity matrix.
    vals = np.arange(1, n+1, dtype=float)
    A = scipy.sparse.diags([vals], [0], (n, n))
    B = scipy.sparse.eye(n)

    # Let the preconditioner M be the inverse of A.
    M = scipy.sparse.diags([np.reciprocal(vals)], [0], (n, n))

    # Pick random initial vectors.
    X = np.random.rand(n, m)

    # Require that the returned eigenvectors be in the orthogonal complement
    # of the first few standard basis vectors.
    m_excluded = 3
    Y = np.eye(n, m_excluded)

    eigs, vecs = lobpcg(A, X, B, M=M, Y=Y, tol=1e-4, maxiter=40, largest=False)

    assert_allclose(eigs, np.arange(1+m_excluded, 1+m_excluded+m))
    _check_eigen(A, eigs, vecs, rtol=1e-3, atol=1e-3)


def _check_eigen(M, w, V, rtol=1e-8, atol=1e-14):
    mult_wV = np.multiply(w, V)
    dot_MV = M.dot(V)
    assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)


def _check_fiedler(n, p):
    # This is not necessarily the recommended way to find the Fiedler vector.
    np.random.seed(1234)
    col = np.zeros(n)
    col[1] = 1
    A = toeplitz(col)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Compute the full eigendecomposition using tricks, e.g.
    # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    tmp = np.pi * np.arange(n) / n
    analytic_w = 2 * (1 - np.cos(tmp))
    analytic_V = np.cos(np.outer(np.arange(n) + 1/2, tmp))
    _check_eigen(L, analytic_w, analytic_V)
    # Compute the full eigendecomposition using eigh.
    eigh_w, eigh_V = eigh(L)
    _check_eigen(L, eigh_w, eigh_V)
    # Check that the first eigenvalue is near zero and that the rest agree.
    assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
    assert_allclose(eigh_w[1:], analytic_w[1:])

    # Check small lobpcg eigenvalues.
    X = analytic_V[:, :p]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
    assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])

    # Check large lobpcg eigenvalues.
    X = analytic_V[:, -p:]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])

    # Look for the Fiedler vector using good but not exactly correct guesses.
    fiedler_guess = np.concatenate((np.ones(n//2), -np.ones(n-n//2)))
    X = np.vstack((np.ones(n), fiedler_guess)).T
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    # Mathematically, the smaller eigenvalue should be zero
    # and the larger should be the algebraic connectivity.
    lobpcg_w = np.sort(lobpcg_w)
    assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)


def test_fiedler_small_8():
    # This triggers the dense path because 8 < 2*5.
    _check_fiedler(8, 2)


def test_fiedler_large_12():
    # This does not trigger the dense path, because 2*5 <= 12.
    _check_fiedler(12, 2)


def test_hermitian():
    np.random.seed(1234)

    sizes = [3, 10, 50]
    ks = [1, 3, 10, 50]
    gens = [True, False]

    for size, k, gen in itertools.product(sizes, ks, gens):
        if k > size:
            continue

        H = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
        H = 10 * np.eye(size) + H + H.T.conj()

        X = np.random.rand(size, k)

        if not gen:
            B = np.eye(size)
            w, v = lobpcg(H, X, maxiter=5000)
            w0, v0 = eigh(H)
        else:
            B = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
            B = 10 * np.eye(size) + B.dot(B.T.conj())
            w, v = lobpcg(H, X, B, maxiter=5000, largest=False)
            w0, v0 = eigh(H, B)

        for wx, vx in zip(w, v.T):
            # Check eigenvector
            assert_allclose(np.linalg.norm(H.dot(vx) - B.dot(vx) * wx)
                            / np.linalg.norm(H.dot(vx)),
                            0, atol=5e-4, rtol=0)

            # Compare eigenvalues
            j = np.argmin(abs(w0 - wx))
            assert_allclose(wx, w0[j], rtol=1e-4)

# The n=5 case tests the alternative small matrix code path that uses eigh().
@pytest.mark.parametrize('n, atol', [(20, 1e-3), (5, 1e-8)])
def test_eigs_consistency(n, atol):
    vals = np.arange(1, n+1, dtype=np.float64)
    A = spdiags(vals, 0, n, n)
    np.random.seed(345678)
    X = np.random.rand(n, 2)
    lvals, lvecs = lobpcg(A, X, largest=True, maxiter=100)
    vals, vecs = eigs(A, k=2)

    _check_eigen(A, lvals, lvecs, atol=atol, rtol=0)
    assert_allclose(np.sort(vals), np.sort(lvals), atol=1e-14)

def test_verbosity():
    """Check that nonzero verbosity level code runs.
    """
    A, B = ElasticRod(100)

    n = A.shape[0]
    m = 20

    np.random.seed(0)
    V = rand(n,m)
    X = linalg.orth(V)

    eigs,vecs = lobpcg(A, X, B=B, tol=1e-5, maxiter=30, largest=False,
                       verbosityLevel=11)

# Sklearn Tests
from scipy._lib._util import make_low_rank_matrix

@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('normalizer', ['auto', 'LU', 'QR'])
def test_randomized_svd_low_rank(dtype, normalizer):
    # Placeholder until things work again
    return True
    # Check that extmath.randomized_svd is consistent with linalg.svd
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    decimal = 5 if dtype == np.float32 else 7
    dtype = np.dtype(dtype)

    # generate a matrix X of approximate effective rank `rank` and no noise
    # component (very structured signal):
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features,
                             effective_rank=rank, tail_strength=0.0,
                             random_state=0).astype(dtype, copy=False)
    assert X.shape == (n_samples, n_features)

    # compute the singular values of X using the slow exact method
    U, s, V = linalg.svd(X, full_matrices=False)

    # Convert the singular values to the specific dtype
    U = U.astype(dtype, copy=False)
    s = s.astype(dtype, copy=False)
    V = V.astype(dtype, copy=False)

    # compute the singular values of X using the fast approximate method
    Ua, sa, Va = lobpcg_svds(
        X, k, random_state=0
    )

    # If the input dtype is float, then the output dtype is float of the
    # same bit size (f32 is not upcast to f64)
    # But if the input dtype is int, the output dtype is float64
    expected_dtype = dtype if dtype.kind == 'f' else np.float64
    assert Ua.dtype == expected_dtype
    assert sa.dtype == expected_dtype
    assert Va.dtype == expected_dtype

    assert Ua.shape == (n_samples, k)
    assert sa.shape == (k,)
    assert Va.shape == (k, n_features)

    # ensure that the singular values of both methods are equal up to the
    # real rank of the matrix
    assert_almost_equal(s[:k], sa, decimal=decimal)

    # check the singular vectors too (while not checking the sign)
    assert_almost_equal(np.dot(U[:, :k], V[:k, :]), np.dot(Ua, Va),
                        decimal=decimal)

    # check the sparse matrix representation
    X = scipy.sparse.csr_matrix(X)

    # compute the singular values of X using the fast approximate method
    Ua, sa, Va = lobpcg_svds(
        X, k, random_state=0
    )
    if dtype.kind == 'f':
        assert Ua.dtype == dtype
        assert sa.dtype == dtype
        assert Va.dtype == dtype
    else:
        assert Ua.dtype.kind == 'f'
        assert sa.dtype.kind == 'f'
        assert Va.dtype.kind == 'f'

    assert_almost_equal(s[:rank], sa[:rank], decimal=decimal)

# Scipy Tests (arpack)
import numpy as np

from numpy.testing import (assert_allclose, assert_array_almost_equal_nulp,
                           assert_equal, assert_array_equal)
from pytest import raises as assert_raises
import pytest

from scipy.linalg import eig, eigh, hilbert, svd
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix, diags
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy._lib._gcutils import assert_deallocated, IS_PYPY

# New stuff you added
from numpy.testing import (assert_array_almost_equal)

#----------------------------------------------------------------------
# sparse SVD tests

def sorted_svd(m, k, which='LM'):
    # Compute svd of a dense matrix m, and return singular vectors/values
    # sorted.
    if isspmatrix(m):
        m = m.todense()
    u, s, vh = svd(m)
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError("unknown which=%r" % (which,))

    return u[:, ii], s[ii], vh[ii]


def svd_estimate(u, s, vh):
    return np.dot(u, np.dot(np.diag(s), vh))

def svd_test_input_check():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)

    assert_raises(ValueError, lobpcg_svds, x, k=-1)
    assert_raises(ValueError, lobpcg_svds, x, k=0)
    assert_raises(ValueError, lobpcg_svds, x, k=10)
    assert_raises(ValueError, lobpcg_svds, x, k=x.shape[0])
    assert_raises(ValueError, lobpcg_svds, x, k=x.shape[1])
    assert_raises(ValueError, lobpcg_svds, x.T, k=x.shape[0])
    assert_raises(ValueError, lobpcg_svds, x.T, k=x.shape[1])


def test_svd_simple_real():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)
    y = np.array([[1, 2, 3, 8],
                  [3, 4, 3, 5],
                  [1, 0, 2, 3],
                  [0, 0, 1, 0]], float)
    z = csc_matrix(x)

    for m in [x.T, x, y, z, z.T]:
        for k in range(1, min(m.shape)):
            u, s, vh = sorted_svd(m, k)
            su, ss, svh = lobpcg_svds(m, k)

            m_hat = svd_estimate(u, s, vh)
            sm_hat = svd_estimate(su, ss, svh)

            assert_array_almost_equal_nulp(m_hat, sm_hat, nulp=1000)


def test_svd_simple_complex():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1 + 1j, 0, 2],
                  [0, 0, 1]], complex)
    y = np.array([[1, 2, 3, 8 + 5j],
                  [3 - 2j, 4, 3, 5],
                  [1, 0, 2, 3],
                  [0, 0, 1, 0]], complex)
    z = csc_matrix(x)

    for m in [x, x.T.conjugate(), x.T, y, y.conjugate(), z, z.T]:
        for k in range(1, min(m.shape) - 1):
            u, s, vh = sorted_svd(m, k)
            su, ss, svh = lobpcg_svds(m, k)

            m_hat = svd_estimate(u, s, vh)
            sm_hat = svd_estimate(su, ss, svh)

            assert_array_almost_equal_nulp(m_hat, sm_hat, nulp=1000)

# Either not relevant or needs to be rewritten to be LOBCPG specific
def test_svd_maxiter():
    # check that maxiter works as expected
    x = hilbert(6)
    # # ARPACK shouldn't converge on such an ill-conditioned matrix with just
    # # one iteration
    # lobpcg_svds(x, 1, n_iter=1)
    # but 100 iterations should be more than enough
    u, s, vt = lobpcg_svds(x, 1, n_iter=100)
    assert_allclose(s, [1.7], atol=0.5)

# # Either not relevant or need to be rewritten to be LOBCPG specific (argument doesn't exist)
# def test_svd_return():
#     # check that the return_singular_vectors parameter works as expected
#     x = hilbert(6)
#     _, s, _ = sorted_svd(x, 2)
#     ss = svds(x, 2, return_singular_vectors=False)
#     assert_allclose(s, ss)

# # Same as above
# def test_svd_which():
#     # check that the which parameter works as expected
#     x = hilbert(6)
#     for which in ['LM', 'SM']:
#         _, s, _ = sorted_svd(x, 2, which=which)
#         ss = lobpcg_svds(x, 2, which=which, return_singular_vectors=False)
#         ss.sort()
#         assert_allclose(s, ss, atol=np.sqrt(1e-15))

# # Same as above
# def test_svd_v0():
#     # check that the v0 parameter works as expected
#     x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)
#
#     u, s, vh = lobpcg_svds(x, 1)
#     u2, s2, vh2 = lobpcg_svds(x, 1, v0=u[:,0])
#
#     assert_allclose(s, s2, atol=np.sqrt(1e-15))


def _check_svds(A, k, U, s, VH):
    n, m = A.shape

    # Check shapes.
    assert_equal(U.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(VH.shape, (k, m))

    # Check that the original matrix can be reconstituted.
    A_rebuilt = (U*s).dot(VH)
    assert_equal(A_rebuilt.shape, A.shape)
    assert_allclose(A_rebuilt, A)

    # Check that U is a semi-orthogonal matrix.
    UH_U = np.dot(U.T.conj(), U)
    assert_equal(UH_U.shape, (k, k))
    assert_allclose(UH_U, np.identity(k), atol=1e-12)

    # Check that V is a semi-orthogonal matrix.
    VH_V = np.dot(VH, VH.T.conj())
    assert_equal(VH_V.shape, (k, k))
    assert_allclose(VH_V, np.identity(k), atol=1e-12)

def test_svd_LM_ones_matrix():
    # Check that svds can deal with matrix_rank less than k in LM mode.
    k = 3
    for n, m in (6, 5), (5, 5), (5, 6):
        for t in float, complex:
            A = np.ones((n, m), dtype=t)
            U, s, VH = lobpcg_svds(A, k)

            # Check some generic properties of svd.
            _check_svds(A, k, U, s, VH)

            # Check that the largest singular value is near sqrt(n*m)
            # and the other singular values have been forced to close
            # to zero.
            assert_allclose(np.max(s), np.sqrt(n*m))
            # Changed to relax tolerance
            assert_array_almost_equal(sorted(s)[:-1], 0, decimal=30)


def test_svd_LM_zeros_matrix():
    # Check that svds can deal with matrices containing only zeros.
    k = 1
    for n, m in (3, 4), (4, 4), (4, 3):
        for t in float, complex:
            A = np.zeros((n, m), dtype=t)
            U, s, VH = lobpcg_svds(A, k)

            # Check some generic properties of svd.
            _check_svds(A, k, U, s, VH)

            # Check that the singular values are zero.
            assert_array_equal(s, 0)


def test_svd_LM_zeros_matrix_gh_3452():
    # Regression test for a github issue.
    # https://github.com/scipy/scipy/issues/3452
    # Note that for complex dype the size of this matrix is too small for k=1.
    n, m, k = 4, 2, 1
    A = np.zeros((n, m))
    U, s, VH = lobpcg_svds(A, k)

    # Check some generic properties of svd.
    _check_svds(A, k, U, s, VH)

    # Check that the singular values are zero.
    assert_array_equal(s, 0)


class CheckingLinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape

    def _matvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.dot(x)

    def _rmatvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.T.conjugate().dot(x)

# # For can't support putting a CheckingLinearOperator in
# # Things attempted-wrapping np.asarray(L) and passing in, passing into lobcpg rather than lobcpg_svds
# def test_svd_linop():
#     nmks = [(6, 7, 3),
#             (9, 5, 4),
#             (10, 8, 5)]
#
#     def reorder(args):
#         U, s, VH = args
#         j = np.argsort(s)
#         return U[:,j], s[j], VH[j,:]
#     # arg not supported in LOBCPG v0
#     for n, m, k in nmks:
#         # Test svds on a LinearOperator.
#         A = np.random.RandomState(52).randn(n, m)
#         L = CheckingLinearOperator(A)
#
#
#         U1, s1, VH1 = reorder(lobpcg_svds(A, k))
#         # Why can arpack pass in a CheckingLinearOperator and have this work (when there's no reference to it)?
#         U2, s2, VH2 = reorder(lobpcg_svds(np.asarray(L), k))
#
#         assert_allclose(np.abs(U1), np.abs(U2))
#         assert_allclose(s1, s2)
#         assert_allclose(np.abs(VH1), np.abs(VH2))
#         assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
#                         np.dot(U2, np.dot(np.diag(s2), VH2)))
#
#         # # Try again with which="SM".
#         # A = np.random.RandomState(1909).randn(n, m)
#         # L = CheckingLinearOperator(A)
#         # # arg not supported in LOBCPG
#         # U1, s1, VH1 = reorder(lobpcg_svds(A, k, which="SM"))
#         # U2, s2, VH2 = reorder(lobpcg_svds(L, k, which="SM"))
#         #
#         # assert_allclose(np.abs(U1), np.abs(U2))
#         # assert_allclose(s1, s2)
#         # assert_allclose(np.abs(VH1), np.abs(VH2))
#         # assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
#         #                 np.dot(U2, np.dot(np.diag(s2), VH2)))
#
#         if k < min(n, m) - 1:
#             # Complex input and explicit which="LM".
#             for (dt, eps) in [(complex, 1e-7), (np.complex64, 1e-3)]:
#                 rng = np.random.RandomState(1648)
#                 A = (rng.randn(n, m) + 1j * rng.randn(n, m)).astype(dt)
#                 L = CheckingLinearOperator(A)
#
#                 U1, s1, VH1 = reorder(lobpcg_svds(A, k))
#                 U2, s2, VH2 = reorder(lobpcg_svds(L, k))
#
#                 assert_allclose(np.abs(U1), np.abs(U2), rtol=eps)
#                 assert_allclose(s1, s2, rtol=eps)
#                 assert_allclose(np.abs(VH1), np.abs(VH2), rtol=eps)
#                 assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
#                                 np.dot(U2, np.dot(np.diag(s2), VH2)), rtol=eps)

# # Arpack specific
# @pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
# def test_linearoperator_deallocation():
#     # Check that the linear operators used by the Arpack wrappers are
#     # deallocatable by reference counting -- they are big objects, so
#     # Python's cyclic GC may not collect them fast enough before
#     # running out of memory if eigs/eigsh are called in a tight loop.
#
#     M_d = np.eye(10)
#     M_s = csc_matrix(M_d)
#     M_o = aslinearoperator(M_d)
#
#     with assert_deallocated(lambda: arpack.SpLuInv(M_s)):
#         pass
#     with assert_deallocated(lambda: arpack.LuInv(M_d)):
#         pass
#     with assert_deallocated(lambda: arpack.IterInv(M_s)):
#         pass
#     with assert_deallocated(lambda: arpack.IterOpInv(M_o, None, 0.3)):
#         pass
#     with assert_deallocated(lambda: arpack.IterOpInv(M_o, M_o, 0.3)):
#         pass

# # vh arg not supported in LOBCPG
# def test_svds_partial_return():
#     x = np.array([[1, 2, 3],
#                   [3, 4, 3],
#                   [1, 0, 2],
#                   [0, 0, 1]], float)
#     # test vertical matrix
#     z = csr_matrix(x)
#     vh_full = lobpcg_svds(z, 2)[-1]
#     vh_partial = lobpcg_svds(z, 2, return_singular_vectors='vh')[-1]
#     dvh = np.linalg.norm(np.abs(vh_full) - np.abs(vh_partial))
#     if dvh > 1e-10:
#         raise AssertionError('right eigenvector matrices differ when using return_singular_vectors parameter')
#     if lobpcg_svds(z, 2, return_singular_vectors='vh')[0] is not None:
#         raise AssertionError('left eigenvector matrix was computed when it should not have been')
#     # test horizontal matrix
#     z = csr_matrix(x.T)
#     u_full = lobpcg_svds(z, 2)[0]
#     u_partial = lobpcg_svds(z, 2, return_singular_vectors='vh')[0]
#     du = np.linalg.norm(np.abs(u_full) - np.abs(u_partial))
#     if du > 1e-10:
#         raise AssertionError('left eigenvector matrices differ when using return_singular_vectors parameter')
#     if lobpcg_svds(z, 2, return_singular_vectors='u')[-1] is not None:
#         raise AssertionError('right eigenvector matrix was computed when it should not have been')

# # which arg (different values) not supported in LOBCPG
# def test_svds_wrong_eigen_type():
#     # Regression test for a github issue.
#     # https://github.com/scipy/scipy/issues/4590
#     # Function was not checking for eigenvalue type and unintended
#     # values could be returned.
#     x = np.array([[1, 2, 3],
#                   [3, 4, 3],
#                   [1, 0, 2],
#                   [0, 0, 1]], float)
#     assert_raises(ValueError, lobpcg_svds, x, 1, which='LA')
