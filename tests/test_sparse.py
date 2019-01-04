import autograd.numpy as np
import autograd_sparse as sp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
from autograd.test_util import check_grads

N = 5

# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return sp.eye(N).tocsr()


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


# ----- vjp0 ----- #

# # case 5
# @pytest.mark.test
# @pytest.mark.sp_sparse
# def test_sparse_dot_0_5(eye):
#     dense = np.random.random(size=(N, N))
#     sparse = eye
#     def fun(x):
#         return sp.dot(x, dense)
#     check_grads(fun)(sparse)

# ----- vjp1 ----- #

# case 3
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot_1_3(eye):
    dense = np.random.random(size=(N, ))
    sparse = eye
    def fun(x):
        return sp.dot(sparse, x)       
    check_grads(fun)(dense)

# case 5
@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot_1_5(eye):
    dense = np.random.random(size=(N, N))
    sparse = eye
    def fun(x):
        return sp.dot(sparse, x)       
    check_grads(fun)(dense)









