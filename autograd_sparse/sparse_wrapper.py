import types

import numpy as _np

import sparse as _sp
from autograd.extend import notrace_primitive, primitive

notrace_functions = [
    _sp.zeros_like,
    _sp.ones_like,
]


# Wrap all sparse functions.
def wrap_namespace(old, new):
    """
    Wraps namespace of array library.
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {
        # _sp.int,
        # _sp.int8,
        # _sp.int16,
        # _sp.int32,
        # _sp.int64,
        # _sp.integer,
    }
    function_types = {
        # _sp.ufunc,
        types.FunctionType,
        types.BuiltinFunctionType
    }
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = notrace_primitive(obj)

        # Note: type(obj) == _sp.ufunc doesn't work! Should use:
        #
        #     isinstance(obj, _sp.ufunc)
        #
        elif (
            type(obj) in function_types
            # or isinstance(obj, _sp.ufunc)
            # or isinstance(obj, _sp.core.fusion.reduction)
        ):
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj


wrap_namespace(_sp.__dict__, globals())


# # ----- definition for coo_matrix ----- #
# def coo_matrix(arg1, *args, **kwargs):
#     return sparse_matrix_from_args('coo', arg1,  *args, **kwargs)


# # ----- definition for csr_matrix ----- #
# def csr_matrix(arg1, *args, **kwargs):
#     return sparse_matrix_from_args('csr', arg1, *args, **kwargs)

# # ----- definition for csc_matrix ----- #
# def csc_matrix(arg1, *args, **kwargs):
#     return sparse_matrix_from_args('csc', arg1, *args, **kwargs)

# # ----- definition for dia_matrix ----- #
# def dia_matrix(arg1, *args, **kwargs):
#     return sparse_matrix_from_args('dia', arg1, *args, **kwargs)



# @primitive
# def sparse_matrix_from_args(type, arg1, *args, **kwargs):
#     if type == 'coo':
#         return _sp.coo_matrix(arg1, *args, **kwargs)
#     elif type == 'csr':
#         return _sp.csr_matrix(arg1, *args, **kwargs)
#     elif type == 'csc':
#         return _sp.csc_matrix(arg1, *args, **kwargs)
#     elif type == 'dia':
#         return _sp.dia_matrix(arg1, *args, **kwargs)
