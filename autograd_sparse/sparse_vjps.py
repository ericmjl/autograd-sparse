from autograd.extend import defvjp
from .sparse_wrapper import dot
import autograd.numpy as anp

def _dot_vjp_0(ans, sparse, dense):
  print('vjp0')
  if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(sparse) == 0:
    print('  1st case')
    return lambda g: anp.sum(dense * g)
  if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
    print('  2nd case')
    return lambda g: g * dense
  if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
    print('  3rd case')
    return lambda g: g[:, None] * dense
  if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
    print('  4th case')
    return lambda g: anp.dot(dense, g)
  print('  5th case')
  return lambda g: sdot(dense.T, g)

def _dot_vjp_1(ans, sparse, dense):
  print('vjp1')
  if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(dense) == 0:
    print('  1st case')
    return lambda g: anp.sum(sparse * g)
  if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
    print('  2nd case')
    return lambda g: g * sparse
  if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
    print('  3rd case')
    dense = dense[:, None]
    return lambda g: dot(sparse.T, g)
  if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
    print('  4th case')
    return lambda g: sparse[:, None] * g
  print('  5th case')
  return lambda g: dot(sparse.T, g)

defvjp(dot, _dot_vjp_0, _dot_vjp_1)
