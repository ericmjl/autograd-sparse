from __future__ import absolute_import
import scipy.sparse as sp
from autograd.extend import Box, primitive
import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
import numpy as onp

Box.__array_priority__ = 90.0


# Define a general box for a sparse array.
class SparseArrayBox(Box):
    __slots__ = []
    __array_priority__ = 110.0

    @primitive
    def __getitem__(A, idx):
        return A[idx]

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim = property(lambda self: self._value.ndim)
    size = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: anp.transpose(self))

    def __len__(self):
        return len(self._value)

    def astype(self, *args, **kwargs):
        return anp._astype(self, *args, **kwargs)

    def __neg__(self):
        return anp.negative(self)

    def __add__(self, other):
        return anp.add(self, other)

    def __sub__(self, other):
        return anp.subtract(self, other)

    def __mul__(self, other):
        return anp.multiply(self, other)

    def __pow__(self, other):
        return anp.power(self, other)

    def __div__(self, other):
        return anp.divide(self, other)

    def __mod__(self, other):
        return anp.mod(self, other)

    def __truediv__(self, other):
        return anp.true_divide(self, other)

    # AttributeError: 'dia_matrix' has no attribute '__matmul__'
    def __matmul__(self, other):
        return anp.matmul(self, other)

    def __radd__(self, other):
        return anp.add(other, self)

    def __rsub__(self, other):
        return anp.subtract(other, self)

    def __rmul__(self, other):
        return anp.multiply(other, self)

    # AttributeError: 'dia_matrix' object has no attribute '__rpow__'
    def __rpow__(self, other):
        return anp.power(other, self)

    def __rdiv__(self, other):
        return anp.divide(other, self)

    # AttributeError: 'dia_matrix' object has no attribute '__rmod__'
    def __rmod__(self, other):
        return anp.mod(other, self)

    def __rtruediv__(self, other):
        return anp.true_divide(other, self)

    # AttributeError: 'dia_matrix' object has no attribute '__rmatmul__'
    def __rmatmul__(self, other):
        return anp.matmul(other, self)

    def __eq__(self, other):
        return anp.equal(self, other)

    def __ne__(self, other):
        return anp.not_equal(self, other)

    def __gt__(self, other):
        return anp.greater(self, other)

    def __ge__(self, other):
        return anp.greater_equal(self, other)

    def __lt__(self, other):
        return anp.less(self, other)

    def __le__(self, other):
        return anp.less_equal(self, other)

    def __abs__(self):
        return anp.abs(self)

    def __hash__(self):
        return id(self)


# Register the types of sparse arrays
SparseArrayBox.register(sp.dia_matrix)
SparseArrayBox.register(sp.csr_matrix)
SparseArrayBox.register(sp.coo_matrix)
SparseArrayBox.register(sp.csc_matrix)

for type_ in [
    float,
    np.float64,
    np.float32,
    np.float16,
    complex,
    np.complex64,
    np.complex128,
]:
    SparseArrayBox.register(type_)

# These numpy.ndarray methods are just refs to an equivalent numpy function
# nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
#                    'argsort', 'nonzero', 'searchsorted', 'round']
# diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
#                 'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
#                 'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
#                 'trace', 'transpose', 'var']
# for method_name in nondiff_methods + diff_methods:
#     setattr(SparseArrayBox, method_name, anp.__dict__[method_name])

# Flatten has no function, only a method.
# setattr(SparseArrayBox, 'flatten', anp.__dict__['ravel'])

# Register matrix.
# ArrayBox.register(onp.matrixlib.defmatrix.matrix)
