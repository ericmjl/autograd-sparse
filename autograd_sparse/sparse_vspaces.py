import numpy as np
from autograd.extend import VSpace
import scipy

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
        a = scipy.sparse.random(m=self.shape[0], n=self.shape[1])
        return self.t(a)

    def zeros(self):
        return self.t(self.shape)

    # def __eq__(self, other):
    #     issametype = type(self) == type(other)
        # issamevals = (self.value != other.value).nnz == 0
    #     return issametype # and issamevals


VSpace.register(scipy.sparse.csc_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.csr_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.coo_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.dia_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(np.matrixlib.defmatrix.matrix, lambda x: VSpace(x))
