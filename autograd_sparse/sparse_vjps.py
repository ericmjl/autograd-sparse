from autograd.extend import defvjp
from .sparse_wrapper import dot, spsolve
import autograd.numpy as anp


def _dot_vjp_0(ans, sparse, dense):
    if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(sparse) == 0:
        return lambda g: anp.sum(dense * g)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
        return lambda g: g * dense
    if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
        return lambda g: g[:, None] * dense
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
        print("  4th case")
        return lambda g: anp.dot(dense, g)
    return lambda g: dot(dense.T, g)


def _dot_vjp_1(ans, sparse, dense):
    if anp.ndim(sparse) != 2 or anp.ndim(dense) > 2:
        raise NotImplementedError(
            "Current dot vjps only support sparse matrices with ndim == 2."
        )
    return lambda g: dot(sparse.T, g)


defvjp(dot, _dot_vjp_0, _dot_vjp_1)


""" SPSOLVE (solves Ax=b for x where A is sparse)"""


def _spsolve_vjp_0(ans, sparse, dense):
    def vjp(g):
        adjoint = spsolve(sparse.T, g)
        print(adjoint)
        return -(ans * adjoint).T

    return vjp


def _spsolve_vjp_1(ans, sparse, dense):
    return lambda g: spsolve(sparse.T, g)


defvjp(spsolve, _spsolve_vjp_0, _spsolve_vjp_1)
