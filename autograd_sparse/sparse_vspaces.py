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

    def _inner_prod(self, x, y):
        x_vec = x.reshape((-1, 1))
        y_vec = y.reshape((-1, 1))
        dot_prod = np.sum(x_vec.T.dot(y_vec))
        return dot_prod

    def __eq__(self, other):
        isequal = True
        for k in self.__dict__.keys():
            vself = self.__dict__[k]
            vother = other.__dict__[k]
            if k == "data" or k == "value":
                if not (vself == vother).todense().all():
                    isequal = False
            else:
                if not vself == vother:
                    isequal = False

        return type(self) == type(other) and isequal


VSpace.register(scipy.sparse.csc_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.csr_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.coo_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(scipy.sparse.dia_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(np.matrixlib.defmatrix.matrix, lambda x: VSpace(x))
