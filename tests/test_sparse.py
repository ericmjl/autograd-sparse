import pdb

import pytest

import autograd.numpy as np
import autograd_sparse as sp
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd.test_util import check_grads


# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return sp.eye(5).tocsr()


# ----- tests for array creation ----- #
@pytest.mark.passes
@pytest.mark.sparse
def test_sparse_coo_matrix():
    """This just has to not error out."""
    data = [1, 1, 1, 1]
    coords = [[0, 1, 2, 3],
              [0, 1, 2, 3]]
    sparse = sp.COO(coords, data)
    print(sparse.shape)


# ----- tests for array multiplication ----- #
@pytest.mark.passes
@pytest.mark.sparse
def test_sparse_dense_multiplication(eye):
    """This just has to not error out."""
    dense = np.random.random(size=(5, 5))
    sp.dot(eye, dense)
    sp.dot(dense, eye)



@pytest.mark.test
@pytest.mark.sparse
def test_sparse_dot_grad(eye):
    dense = np.random.random(size=(1, 5))

    def fun(x):
        return sp.dot(dense, x)

    check_grads(fun)(eye)
