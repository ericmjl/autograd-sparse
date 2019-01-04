import scipy.sparse as _sp
import scipy.sparse.linalg as _spl
from autograd.extend import primitive
import numpy as _np

# ----- definition for coo_matrix ----- #
def coo_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args("coo", arg1, *args, **kwargs)


# ----- definition for csr_matrix ----- #
def csr_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args("csr", arg1, *args, **kwargs)


# ----- definition for csc_matrix ----- #
def csc_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args("csc", arg1, *args, **kwargs)


# ----- definition for dia_matrix ----- #
def dia_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args("dia", arg1, *args, **kwargs)


@primitive
def sparse_matrix_from_args(type, arg1, *args, **kwargs):
    if type == "coo":
        return _sp.coo_matrix(arg1, *args, **kwargs)
    elif type == "csr":
        return _sp.csr_matrix(arg1, *args, **kwargs)
    elif type == "csc":
        return _sp.csc_matrix(arg1, *args, **kwargs)
    elif type == "dia":
        return _sp.dia_matrix(arg1, *args, **kwargs)


def isdense(a):
    """
    Utility function for checking whether a matrix is a dense matrix or not.
    """
    return isinstance(a, np.ndarray)


def issparse(a):
    """
    Utility function for checking whether a matrix is a sparse matrix or not.
    """
    return (
        isinstance(a, _sp.coo_matrix)
        or isinstance(a, _sp.csr_matrix)
        or isinstance(a, _sp.csc_matrix)
        or isinstance(a, _sp.dia_matrix)
    )


@primitive
def dot(a, b):
    return a @ b


@primitive
def spsolve(a, b):
    return _spl.spsolve(a, b)


@primitive
def eye(N):
    return _sp.eye(N)


@primitive
def sp_rand(N):
    return _sp.random(N, N, density=0.5) + _sp.eye(N)
