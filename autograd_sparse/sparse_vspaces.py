import numpy as np

import sparse
from autograd.extend import VSpace
import scipy.sparse as ssp

class SparseArrayVSpace(VSpace):

    def __init__(self, value):
        self.t = type(value)
        self.value = self.t(value)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self):
        return self.value.size

    @property
    def ndim(self):
        return len(self.shape)

    def randn(self):
        a = sparse.random((self.shape[0], self.shape[1]))
        return a

    def zeros(self):
        return self.t(self.shape)

    def __eq__(self, other):
        issametype = type(self) == type(other)
        issamevals = (self.value != other.value).nnz == 0
        return issametype # and issamevals


VSpace.register(sparse.COO, lambda x: SparseArrayVSpace(x))
VSpace.register(ssp.csr_matrix, lambda x: SparseArrayVSpace(x))
