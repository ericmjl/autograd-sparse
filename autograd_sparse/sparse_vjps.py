import autograd.numpy as anp
from autograd.extend import defvjp

import autograd_sparse.sparse_wrapper as sp


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
      return lambda g: anp.dot(dense, g)
  return lambda g: sp.dot(g, dense.T)

def _dot_vjp_1(ans, sparse, dense):
    if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(dense) == 0:
        return lambda g: anp.sum(sparse * g)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
        return lambda g: g * sparse
    if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
        return lambda g: sp.dot(g, sparse)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
        return lambda g: sparse[:, None] * g
    return lambda g: sp.dot(sparse.T, g)


defvjp(sp.dot, _dot_vjp_0, _dot_vjp_1)
