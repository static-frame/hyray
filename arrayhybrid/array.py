from __future__ import annotations
import types

import numpy as np
import typing as tp

try:
    import cupy as cp
except ImportError:
    cp = None

CuPyArray = tp.Any # TODO: get cupy array type

DTYPE_KIND_CUPY = frozenset(('b', 'i', 'u', 'f', 'c'))

# 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view'

class CuArrayFlags:
    '''
    Wrapper for flags, reading from the underlying CuPy array when possible, otherwise simiulating a writeable flag that is always False.

    NOTE: `aligned` and `writebackifcopy` are not implemented for now.
    '''
    __slots__ = ('_array',)
    _array: CuPyArray

    def __init__(self, array: CuPyArray) -> None:
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
    A wrapper around a CuPy array that inforces immutability. Conversion to NumPy arrays, either by `astype()` or operators, are supported.
    '''

    __slots__ = (
            '_array',
            'flags',
            )

    _array: CuPyArray
    flags: CuArrayFlags

    def __init__(self, array: CuPyArray) -> None:
        if not cp:
            raise RuntimeError('Cannot create a CuArray as no CuPy installation available.')

        self._array = array # cp.array
        self.flags = CuArrayFlags(array)

    #---------------------------------------------------------------------------
    # properties

    __hash__ = None

    @property
    def __array_priority__(self) -> int:
        return self._array.__array_priority__

    @property
    def base(self) -> tp.Optional[CuArray]:
        b = self._array.base
        if b is None:
            return None
        return CuArray(b)

    @property
    def data(self) -> tp.Optional[CuArray]:
        return CuArray(self._array.data)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    # flags is instance attr

    @property
    def flat(self) -> tp.Iterator[tp.Any]:
        yield from (e.item() for e in self._array.flat)

    @property
    def imag(self) -> CuArray:
        return CuArray(self._array.imag)

    @property
    def itemsize(self) -> int:
        return self._array.itemsize

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def real(self) -> CuArray:
        return CuArray(self._array.real)

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self._array.shape

    @property
    def size(self) -> int:
        return self._array.size

    @property
    def strides(self) -> tp.Tuple[int, ...]:
        return self._array.strides

    @property
    def T(self) -> CuArray:
        return CuArray(self._array.T)

    #---------------------------------------------------------------------------
    # magic methods

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError('In-place mutation not permitted.')

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if self._array.ndim == 1:
            # NOTE: CuPy iters 0 dimensional arrays: convert to PyObjects with item()
            yield from (e.item() for e in self._array.__iter__())
        else:
            yield from (CuArray(a) for a in self._array.__iter__())

    def __getitem__(self, *args) -> tp.Any:
        v = self._array.__getitem__(*args)
        if v.ndim == 0:
            return v.item()
        return CuArray(v)

    def __len__(self) -> tp.Tuple[int, ...]:
        return self._array.__len__()

    #---------------------------------------------------------------------------
    # methods

    def astype(self,
            dtype,
            order='K',
            casting=None,
            subok=None,
            copy=True,
            ) -> tp.Union[CuArray, np.ndarray]:
        if casting is not None:
            raise NotImplementedError('`casting` not supported')
        if subok is not None:
            raise NotImplementedError('`subok` not supported')

        dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)

        if dt.kind in DTYPE_KIND_CUPY:
            return CuArray(self._array.astype(dt,
                    order=order,
                    copy=copy,
                    ))

        # Return a NumPy array of the requested type.
        return self._array.get().astype(dt,
                order=order,
                copy=copy,
                )

    def transpose(self) -> CuArray:
        return CuArray(self._array.transpose())

    def tolist(self) -> np.dtype:
        return self._array.tolist()


    def reshape(self, shape, *args, order='C'):
        return CuArray(self._array.reshape(shape, *args, order=order))



# 'abs', 'add', 'all', 'amax', 'amin', 'angle', 'any', 'append', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'byte_bounds', 'ceil', 'choose', 'clip',np'column_stack', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'cumprod', 'cumproduct', 'cumsum', 'deg2rad', 'degrees', 'delete', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'dsplit', 'dstack', 'einsum', 'empty', 'empty_like', 'equal', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', ''float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'from_dlpack', 'frombuffer', 'fromfile', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'gcd', 'geomspace', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'interp', 'intersect1d', 'intp', 'invert', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'kaiser', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mask_indices', 'mat', 'matmul', 'max', 'maximum', 'maximum_sctype', 'mean', 'median', 'meshgrid', 'mgrid', 'min', 'min_scalar_type', 'minimum', 'mintypecode', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'numarray', 'number', 'ogrid', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial', 'polysub', 'polyval', 'positive', 'power', 'prod', 'product', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'r_', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 'searchsorted', 'select', 'shape', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'subtract', 'sum', 'swapaxes', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'tile', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typecodes', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'version', 'void', 'vsplit', 'vstack', 'where', 'who', 'zeros', 'zeros_like']

UnionNpCuPy = tp.Union[np.ndarray, CuArray]

class ArrayHybrid(types.ModuleType):
    '''A module-like interface that uses CuPy when available, otherwise uses NumPy.
    '''
    # https://docs.python.org/3/library/types.html#types.ModuleType
    __name__ = 'hyray'
    __package__ = ''

    @staticmethod
    def ndarray(shape,
            dtype=float,
            buffer=None,
            offset=0,
            strides=None,
            order=None,
            ) -> UnionNpCuPy:
        dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
        if cp and buffer is None and dt.kind in DTYPE_KIND_CUPY:
            # offset not an arg; strides can be given if memptr is given;
            try:
                return CuArray(cp.ndarray(shape, dtype=dtype, order=order))
            except cp.cuda.memory.OutOfMemoryError:
                pass
        return np.ndarray(shape,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order,
                )

    @staticmethod
    def array(value,
            dtype=None,
            *,
            copy=True,
            order='K',
            subok=False,
            ndmin=0,
            like=None,
            ) -> UnionNpCuPy:
        if like is not None:
            raise NotImplementedError('`like` not supported')
        # default dtype of None is necessary for auto-detection of type
        if cp:
            try:
                return CuArray(cp.array(value,
                        dtype=dtype,
                        copy=copy,
                        order=order,
                        subok=subok,
                        ndmin=ndmin,
                        ))
            except (ValueError, cp.cuda.memory.OutOfMemoryError):
                # expect ValueError is Unsupported dtype
                pass
        return np.array(value,
                dtype=dtype,
                copy=copy,
                order=order,
                subok=subok,
                ndmin=ndmin,
                )

    @staticmethod
    def empty(shape, dtype=float, order='C', *, like=None):
        if like is not None:
            raise NotImplementedError('`like` not supported')

        dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
        if cp and dt.kind in DTYPE_KIND_CUPY:
            try:
                return CuArray(cp.empty(shape,
                        dtype=dt,
                        order=order,
                        ))
            except cp.cuda.memory.OutOfMemoryError:
                pass
        return np.empty(shape,
                dtype=dt,
                order=order,
                )


    @staticmethod
    def full(shape, fill_value, dtype=None, order='C', *, like=None):
        pass


    @staticmethod
    def arange(start, stop=None, step=None, dtype=None, *, like=None):
        if like is not None:
            raise NotImplementedError('`like` not supported')

        if cp:
            try:
                return CuArray(cp.arange(start, stop, step, dtype=dtype))
            except cp.cuda.memory.OutOfMemoryError:
                pass
        return np.arange(start, stop, step, dtype=dtype)