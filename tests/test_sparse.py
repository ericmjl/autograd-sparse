import autograd.numpy as np
import autograd_sparse as sp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad, jacobian
import pdb
from autograd.test_util import check_grads

N = 5

# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return sp.eye(N).tocsr()

@pytest.fixture
def sp_rand():
    return sp.sp_rand(N).tocsr()


# ----- tests for array creation ----- #
@pytest.mark.works
@pytest.mark.sparse
def test_sparse_coo_matrix():
    """This just has to not error out."""
    data = np.array([1, 2, 3]).astype('float32')
    rows = np.array([1, 2, 3]).astype('float32')
    cols = np.array([1, 3, 4]).astype('float32')
    sparse = sp.coo_matrix(data, (rows, cols))


# ----- tests for array multiplication ----- #
@pytest.mark.sparse
def test_sparse_dense_multiplication(eye):
    """This just has to not error out."""
    dense = np.random.random(size=(N, N))
    sp.dot(eye, dense)
    sp.dot(dense, eye)



# ----- tests for dot product ----- #


# differentiating with respect to sparse argument (will fail)
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot_0_2(eye):
    dense = np.random.random(size=(N, N))
    sparse = eye
    def fun(x):
        return sp.dot(x, dense)
    check_grads(fun)(sparse)


# differentiating with respect to dense argument (will pass)
# dense.ndim = 1
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot_1_1(eye):
    dense = np.random.random(size=(N, ))
    sparse = eye
    def fun(x):
        return sp.dot(sparse, x)       
    check_grads(fun)(dense)

# differentiating with respect to dense argument (will pass)
# dense.ndim = 2
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot_1_2(sp_rand):
    dense = np.random.random(size=(N, N))
    sparse = sp_rand
    def fun(x):
        return sp.dot(sparse, x)       
    check_grads(fun)(dense)


# ----- tests of spsolve ----- #


# differentiating with respect to sparse argument (will fail)
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_spsolve_0_2(sp_rand):
    dense = np.random.random(size=(N, N))
    sparse = sp_rand
    def fun(x):
        return sp.spsolve(x, dense)   
    check_grads(fun)(sparse)

# differentiating with respect to dense argument (will pass)
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_spsolve_1_2(sp_rand):
    dense = np.random.random(size=(N, N))
    sparse = sp_rand
    def fun(x):
        return sp.spsolve(sparse, x)   
    check_grads(fun)(dense)





