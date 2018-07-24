import autograd.numpy as np
import autograd_sparse as sp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
from autograd.test_util import check_grads


# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return sp.eye(5).tocsr()


# ----- tests for array creation ----- #
@pytest.mark.works
@pytest.mark.sparse
def test_sparse_coo_matrix():
    """This just has to not error out."""
    data = np.array([1, 2, 3]).astype('float32')
    rows = np.array([1, 2, 3]).astype('float32')
    cols = np.array([1, 3, 4]).astype('float32')
    sparse = sp.coo_matrix(data, (rows, cols))
    print(sparse.shape)


# ----- tests for array multiplication ----- #
@pytest.mark.sparse
def test_sparse_dense_multiplication(eye):
    """This just has to not error out."""
    dense = np.random.random(size=(5, 5))
    sp.dot(eye, dense)
    sp.dot(dense, eye)



@pytest.mark.test
@pytest.mark.sp_sparse
def test_sparse_dot(eye):
    dense = np.random.random(size=(1, 5))

    def fun(x):
        return sp.dot(dense, x)

    check_grads(fun)(eye)
