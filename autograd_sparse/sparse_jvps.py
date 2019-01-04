from autograd.extend import def_linear
from .sparse_wrapper import dot, spsolve

def_linear(dot)
def_linear(spsolve)
