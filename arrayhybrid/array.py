from __future__ import annotations
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

# 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view'

class CuArrayFlags:
    __slots__ = (
            '_array',
            'aligned',
            'writebackifcopy'
            )

    def __init__(self, array: CuArray) -> None:
        self._array = array

    @property
    def c_contiguous(self) -> bool:
        return self._array.flags.c_contiguous

    @property
    def f_contiguous(self) -> bool:
        return self._array.flags.f_contiguous

    @property
    def owndata(self) -> bool:
        return self._array.flags.owndata

    @property
    def writeable(self) -> bool:
        return False

    @writeable.setter
    def writeable(self, value: bool) -> None:
        if value is True:
            raise ValueError('Cannot set array to writeable')



class CuArray:
    '''
    A wrapper around a CuPy array, but supporting operations and astyping to numpy arrays
    '''
    __slots__ = (
            '_array',
            'flags',
            )

    def __init__(self, array) -> None:
        if not cp:
            raise RuntimeError('Cannot create a CuArray as no CuPy installation available.')

        self._array = array # cp.array
        self.flags = CuArrayFlags(array)

    @classmethod
    def ndarray(cls,
            shape,
            dtype=float,
            buffer=None,
            offset=0,
            strides=None,
            order=None,
            ) -> None:
        # NOTE: this might better live on the module
        if not cp:
            raise RuntimeError('Cannot create a CuArray as no CuPy installation available.')

        # offset not an arg; strides can be given if memptr is given;
        return cls(cp.ndarray(shape, dtype=dtype, order=order))

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError()

    def astype(self,
            dtype,
            order='K',
            casting='unsafe',
            subok=True,
            copy=True,
            ) -> tp.Union[CuArray, np.ndarray]:

        if dtype.kind in ('i', 'f', 'b'):
            return CuArray(self._array.astype(dtype))

        return self._array.get().astype(dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
                )

    def transpose(self) -> CuArray:
        return CuArray(self._array.transpose())

    @property
    def T(self) -> CuArray:
        return CuArray(self._array.T)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

# 'abs', 'add', 'all', 'amax', 'amin', 'angle', 'any', 'append', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'byte_bounds', 'ceil', 'choose', 'clip',np'column_stack', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'cumprod', 'cumproduct', 'cumsum', 'deg2rad', 'degrees', 'delete', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'dsplit', 'dstack', 'einsum', 'empty', 'empty_like', 'equal', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', ''float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'from_dlpack', 'frombuffer', 'fromfile', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'gcd', 'geomspace', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'interp', 'intersect1d', 'intp', 'invert', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'kaiser', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mask_indices', 'mat', 'matmul', 'max', 'maximum', 'maximum_sctype', 'mean', 'median', 'meshgrid', 'mgrid', 'min', 'min_scalar_type', 'minimum', 'mintypecode', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'numarray', 'number', 'ogrid', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial', 'polysub', 'polyval', 'positive', 'power', 'prod', 'product', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'r_', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 'searchsorted', 'select', 'shape', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'subtract', 'sum', 'swapaxes', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'tile', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typecodes', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'version', 'void', 'vsplit', 'vstack', 'where', 'who', 'zeros', 'zeros_like']

class ArrayHybridModule:

    @staticmethod
    def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None):
        pass

    @staticmethod
    def empty(shape, dtype=float, order='C', *, like=None):
        pass

    @staticmethod
    def full(shape, fill_value, dtype=None, order='C', *, like=None):
        pass

