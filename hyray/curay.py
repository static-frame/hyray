from __future__ import annotations
import types

import numpy as np
import typing as tp

try:
    import cupy as cp
    CuPyArray = cp.ndarray
except ImportError:
    cp = None
    CuPyArray = tp.Any

_DTYPE_KIND_CUPY = frozenset(('b', 'i', 'u', 'f', 'c'))

# CuPy explicitly states: "Implicit conversion to a NumPy array is not allowed." But here, we want to permit this.

# 'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view'

class CuArrayFlags:
    '''
    Wrapper for flags, reading from the underlying CuPy array when possible, otherwise simiulating a writeable flag that is always False.

    NOTE: `aligned` and `writebackifcopy` are not implemented for now.
    '''
    __slots__ = (
            '_array',
            '_writeable',
            )


    _array: CuPyArray

    def __init__(self,
            array: CuPyArray,
            writeable: bool = True,
            ) -> None:
        self._array = array
        self._writeable = writeable

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
        return self._writeable

    @writeable.setter
    def writeable(self, value: bool) -> None:
        # like NP, we take the truthy interpretation
        self._writeable = bool(value)

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

    def __init__(self,
            array: CuPyArray,
            writeable: bool = True,
            ) -> None:
        if not cp:
            raise RuntimeError('Cannot create a CuArray as no CuPy installation available.')

        self._array = array # cp.array
        self.flags = CuArrayFlags(array, writeable)

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

    def __abs__(self) -> CuArray:
        return CuArray(self._array.__abs__())

    def __add__(self, value, /) -> CuArray:
        return CuArray(self._array.__add__(value))

    def __and__(self, value, /) -> CuArray:
        return CuArray(self._array.__and__(value))

    def __array__(self, dtype=None, /) -> np.ndarray:
        '''
        NOTE: CuPy raises a TypeError for this, stating Implicit conversion to a NumPy array is not allowed. Here, we permit it.
        '''
        order = 'F' if self.flags.f_contiguous else 'C'
        a = self._array.get(order=order)
        if dtype is not None and dtype != a.dtype:
            return a.astype(dtype)
        return a

    def __array_function__(self, func, types, args, kwargs):
        a = self._array.__array_function__(
                func,
                (cp.ndarray,),
                args,
                kwargs,
                )
        if a is NotImplemented:
            return NotImplemented
        if a.__class__ is not cp.ndarray:
            return a
        if a.ndim == 0:
            return a.item()
        return CuArray(a)

    def __array_ufunc__(self,
            ufunc,
            method,
            *inputs,
            **kwargs,
            ):
        # NOTE: not tested
        a = self._array.__array_ufunc__(ufunc,
                method,
                *inputs,
                **kwargs,
                )
        if a.ndim == 0:
            return a.item()
        return CuArray(a)

    def __bool__(self) -> bool:
        # will raise ValueError
        self._array.__bool__()

    def __complex__(self) -> complex:
        return self._array.__complex__()

    def __copy__(self):
        return CuArray(self._array.__copy__())

    def __deepcopy__(self, memo):
        return CuArray(self._array.__deepcopy__(memo))

    def __dir__(self) -> tp.List[str]:
        return self._array.__dir__()

    def __divmod__(self, value, /) -> tp.Tuple[CuArray, CuArray]:
        q, r = self._array.__divmod__(value)
        return CuArray(q), CuArray(r)

    def __dlpack__(self, stream=None):
        return self._array.__dlpack__(stream)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()

    def __eq__(self, value, /):
        return CuArray(self._array.__eq__(value))

    def __float__(self) -> float:
        return self._array.__float__()

    def __floordiv__(self, value, /):
        return CuArray(self._array.__floordiv__(value))

    def __format__(self):
        return self._array.__format__()

    def __ge__(self, value, /):
        return CuArray(self._array.__ge__(value))

    def __getitem__(self, *args) -> tp.Any:
        v = self._array.__getitem__(*args)
        if v.ndim == 0:
            return v.item()
        return CuArray(v)

    def __gt__(self, value, /):
        return CuArray(self._array.__gt__(value))

    def __iadd__(self, value, /):
        return CuArray(self._array.__iadd__(value))

    def __iand__(self, value, /):
        return CuArray(self._array.__iand__(value))

    def __ifloordiv__(self, value, /):
        return CuArray(self._array.__ifloordiv__(value))

    def __ilshift__(self, value, /):
        return CuArray(self._array.__ilshift__(value))

    def __imod__(self, value, /):
        return CuArray(self._array.__imod__(value))

    def __imul__(self, value, /):
        return CuArray(self._array.__imul__(value))

    def __int__(self) -> int:
        return self._array.__int__()

    def __invert__(self):
        return CuArray(self._array.__invert__())

    def __ior__(self, value, /):
        return CuArray(self._array.__ior__(value))

    def __ipow__(self, value, /):
        return CuArray(self._array.__ipow__(value))

    def __irshift__(self, value, /):
        return CuArray(self._array.__irshift__(value))

    def __isub__(self, value, /):
        return CuArray(self._array.__isub__(value))

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if self._array.ndim == 1:
            # NOTE: CuPy iters 0 dimensional arrays: convert to PyObjects with item()
            yield from (e.item() for e in self._array.__iter__())
        else:
            yield from (CuArray(a) for a in self._array.__iter__())

    def __itruediv__(self, value, /):
        return CuArray(self._array.__itruediv__(value))

    def __ixor__(self, value, /):
        return CuArray(self._array.__ixor__(value))

    def __le__(self, value, /):
        return CuArray(self._array.__le__(value))

    def __len__(self) -> int:
        return self._array.__len__()

    def __lshift__(self, value, /):
        return CuArray(self._array.__lshift__(value))

    def __lt__(self, value, /):
        return CuArray(self._array.__lt__(value))

    def __matmul__(self, value, /):
        return CuArray(self._array.__matmul__(value))

    def __mod__(self, value, /):
        return CuArray(self._array.__mod__(value))

    def __mul__(self, value, /):
        return CuArray(self._array.__mul__(value))

    def __ne__(self, value, /):
        return CuArray(self._array.__ne__(value))

    def __neg__(self):
        return CuArray(self._array.__neg__())

    def __or__(self, value, /):
        return CuArray(self._array.__or__(value))

    def __pos__(self):
        return CuArray(self._array.__pos__())

    def __pow__(self, value, mod=None, /):
        return CuArray(self._array.__pow__(value, mod))

    def __radd__(self, value, /):
        return CuArray(self._array.__radd__(value))

    def __rand__(self, value, /):
        return CuArray(self._array.__rand__(value))

    def __rdivmod__(self, value, /):
        q, r = self._array.__rdivmod__(value)
        return CuArray(q), CuArray(r)

    def __repr__(self):
        return self._array.__repr__()

    def __rfloordiv__(self, value, /):
        return CuArray(self._array.__rfloordiv__(value))

    def __rlshift__(self, value, /):
        return CuArray(self._array.__rlshift__(value))

    def __rmatmul__(self, value, /):
        return CuArray(self._array.__rmatmul__(value))

    def __rmod__(self, value, /):
        return CuArray(self._array.__rmod__(value))

    def __rmul__(self, value, /):
        return CuArray(self._array.__rmul__(value))

    def __ror__(self, value, /):
        return CuArray(self._array.__ror__(value))

    def __rpow__(self, value, mod=None, /):
        return CuArray(self._array.__rpow__(value, mod))

    def __rrshift__(self, value, /):
        return CuArray(self._array.__rrshift__(value))

    def __rshift__(self, value, /):
        return CuArray(self._array.__rshift__(value))

    def __rsub__(self, value, /):
        return CuArray(self._array.__rsub__(value))

    def __rtruediv__(self, value, /):
        return CuArray(self._array.__rtruediv__(value))

    def __rxor__(self, value, /):
        return CuArray(self._array.__rxor__(value))

    def __setitem__(self, key, value, /):
        if not self.flags.writeable:
            raise ValueError('assignment destination is read-only')
        self._array.__setitem__(key, value)

    def __sizeof__(self) -> int:
        return self._array.__sizeof__()

    def __str__(self) -> str:
        return self._array.__str__()

    def __sub__(self, value, /):
        return CuArray(self._array.__sub__(value))

    def __truediv__(self, value, /):
        return CuArray(self._array.__truediv__(value))

    def __xor__(self, value, /):
        return CuArray(self._array.__xor__(value))


    #---------------------------------------------------------------------------
    # methods

    def all(self, axis=None, out=None, keepdims=False):
        a = self._array.all(axis, out, keepdims)
        if a.ndim == 0:
            return a.item()
        return CuArray(a)

    def any(self, axis=None, out=None, keepdims=False):
        a = self._array.any(axis, out, keepdims)
        if a.ndim == 0:
            return a.item()
        return CuArray(a)

    def argmax(self, axis=None, out=None, *, keepdims=False):
        return CuArray(self._array.argmax(axis, out, keepdims=keepdims))

    def argmin(self, axis=None, out=None, *, keepdims=False):
        return CuArray(self._array.argmin(axis, out, keepdims=keepdims))

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        return CuArray(self._array.argpartition(kth, axis, kind, order))

    def argsort(self, axis=-1, kind=None, order=None):
        return CuArray(self._array.argsort(axis, kind, order))

    def astype(self,
            dtype,
            order='K',
            copy=True,
            ) -> tp.Union[CuArray, np.ndarray]:
        dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)

        if dt.kind in _DTYPE_KIND_CUPY:
            return CuArray(self._array.astype(dt,
                    order=order,
                    copy=copy,
                    ))

        # Return a NumPy array of the requested type.
        return self._array.get(order=order).astype(dt,
                copy=copy,
                )


    def choose(self, choices, out=None, mode='raise'):
        return CuArray(self._array.choose(choices, out, mode))

    def clip(self, min=None, max=None, out=None, **kwargs):
        return CuArray(self._array.clip(min, max, out, **kwargs))

    def compress(self, condition, axis=None, out=None):
        return CuArray(self._array.compress(condition, axis, out))

    def conj(self):
        return CuArray(self._array.conj())

    def conjugate(self):
        return CuArray(self._array.conjugate())

    def copy(self, order='C'):
        return CuArray(self._array.copy(order))

    def cumprod(self, axis=None, dtype=None, out=None):
        return CuArray(self._array.cumprod(axis, dtype, out))

    def cumsum(self, axis=None, dtype=None, out=None):
        return CuArray(self._array.cumsum(axis, dtype, out))

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return CuArray(self._array.diagonal(offset, axis1, axis2))

    def dump(self, file):
        return self._array.dump(file)

    def dumps(self):
        return self._array.dumps()

    def fill(self, value):
        return CuArray(self._array.fill(value))

    def flatten(self, order='C'):
        return CuArray(self._array.flatten(order))

    def item(self, *args):
        return self._array.item(*args)

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return CuArray(self._array.max(axis, out, keepdims, initial, where))

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return CuArray(self._array.mean(axis, dtype, out, keepdims, where=where))

    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return CuArray(self._array.min(axis, out, keepdims, initial, where))

    def nonzero(self):
        return CuArray(self._array.nonzero())

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        return CuArray(self._array.partition(kth, axis, kind, order))

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True):
        return CuArray(self._array.prod(axis, dtype, out, keepdims, initial, where))

    def ptp(self, axis=None, out=None, keepdims=False):
        return CuArray(self._array.ptp(axis, out, keepdims))

    def put(self, indices, values, mode='raise'):
        return CuArray(self._array.put(indices, values, mode))

    def ravel(self, order):
        return CuArray(self._array.ravel(order))

    def repeat(self, repeats, axis=None):
        return CuArray(self._array.repeat(repeats, axis))

    def reshape(self, shape, *args, order='C'):
        return CuArray(self._array.reshape(shape, *args, order=order))

    def round(self, decimals=0, out=None):
        return CuArray(self._array.round(decimals, out))

    def sort(self, axis=-1, kind=None) -> None:
        # NOTE: kind not accepted in early versions
        if not self.flags.writeable:
            raise ValueError('sort array is read-only')

        self._array.sort(axis)

    def squeeze(self, axis=None):
        return CuArray(self._array.squeeze(axis))

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        return CuArray(self._array.std(axis, dtype, out, ddof, keepdims, where=where))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        a = self._array.sum(axis, dtype, out, keepdims)
        if a.ndim == 0:
            return a.item()
        return CuArray(a)

    def swapaxes(self, axis1, axis2):
        return CuArray(self._array.swapaxes(axis1, axis2))

    def take(self, indices, axis=None, out=None, mode='raise'):
        return CuArray(self._array.take(indices, axis, out, mode))

    def tobytes(self, order='C') -> bytes:
        return self._array.tobytes(order)

    def tofile(self, fid, sep="", format="%s") -> None:
        self._array.tofile(fid, sep, format)

    def tolist(self) -> tp.List[tp.Any]:
        return self._array.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return CuArray(self._array.trace(offset, axis1, axis2, dtype, out))

    def transpose(self, *axes):
        return CuArray(self._array.transpose(*axes))

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        return CuArray(self._array.var(axis, dtype, out, ddof, keepdims, where=where))

    def view(self, dtype, type):
        return CuArray(self._array.view(dtype, type))



# 'abs', 'add', 'all', 'amax', 'amin', 'angle', 'any', 'append', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'byte_bounds', 'ceil', 'choose', 'clip',np'column_stack', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'cumprod', 'cumproduct', 'cumsum', 'deg2rad', 'degrees', 'delete', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'dsplit', 'dstack', 'einsum', 'empty', 'empty_like', 'equal', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', ''float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'from_dlpack', 'frombuffer', 'fromfile', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'gcd', 'geomspace', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'interp', 'intersect1d', 'intp', 'invert', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'kaiser', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mask_indices', 'mat', 'matmul', 'max', 'maximum', 'maximum_sctype', 'mean', 'median', 'meshgrid', 'mgrid', 'min', 'min_scalar_type', 'minimum', 'mintypecode', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'numarray', 'number', 'ogrid', 'ones', 'ones_like', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial', 'polysub', 'polyval', 'positive', 'power', 'prod', 'product', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'r_', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 'searchsorted', 'select', 'shape', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'subtract', 'sum', 'swapaxes', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'tile', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typecodes', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'version', 'void', 'vsplit', 'vstack', 'where', 'who', 'zeros', 'zeros_like']
