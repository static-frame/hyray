import typing as tp

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from hyray.curay import _DTYPE_KIND_CUPY
from hyray.curay import ndcuray

UnionNpCuRay = tp.Union[np.ndarray, ndcuray]

#-------------------------------------------------------------------------------
# constants: CP uses aliases to NP

c_ = np.c_
e = np.e
euler_gamma = np.euler_gamma
index_exp = np.index_exp
Inf = np.Inf
inf = np.inf
Infinity = np.Infinity
infty = np.infty
mgrid = np.mgrid
NAN = np.NAN
NaN = np.NaN
nan = np.nan
newaxis = np.newaxis
NINF = np.NINF
NZERO = np.NZERO
ogrid = np.ogrid
pi = np.pi
PINF = np.PINF
PZERO = np.PZERO
r_ = np.r_
s_ = np.s_

#-------------------------------------------------------------------------------
# types: CP uses alias to NP

bool_ = np.bool_
broadcast = np.broadcast
byte = np.byte
cdouble = np.cdouble
cfloat = np.cfloat
complex128 = np.complex128
complex64 = np.complex64
complex_ = np.complex_
complexfloating = np.complexfloating
csingle = np.csingle
DataSource = np.DataSource
double = np.double
finfo = np.finfo
float16 = np.float16
float32 = np.float32
float64 = np.float64
float_ = np.float_
floating = np.floating
format_parser = np.format_parser
half = np.half
iinfo = np.iinfo
inexact = np.inexact
int16 = np.int16
int32 = np.int32
int64 = np.int64
int8 = np.int8
int_ = np.int_
intc = np.intc
integer = np.integer
intp = np.intp
longfloat = np.longfloat
longlong = np.longlong
ndindex = np.ndindex
number = np.number
short = np.short
signedinteger = np.signedinteger
single = np.single
singlecomplex = np.singlecomplex
ubyte = np.ubyte
uint = np.uint
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
uint8 = np.uint8
uintc = np.uintc
uintp = np.uintp
ulonglong = np.ulonglong
unsignedinteger = np.unsignedinteger
ushort = np.ushort

#-------------------------------------------------------------------------------
# functions


def ndarray(shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        ) -> UnionNpCuRay:
    dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
    if cp and buffer is None and dt.kind in _DTYPE_KIND_CUPY:
        # offset not an arg; strides can be given if memptr is given;
        try:
            return ndcuray(cp.ndarray(shape, dtype=dtype, order=order))
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ndarray(shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
            )

def array(value,
        dtype=None,
        *,
        copy=True,
        order='K',
        subok=False,
        ndmin=0,
        ) -> UnionNpCuRay:
    # default dtype of None is necessary for auto-detection of type
    if cp:
        try:
            return ndcuray(cp.array(value,
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

def empty(shape, dtype=float, order='C'):
    dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
    if cp and dt.kind in _DTYPE_KIND_CUPY:
        try:
            return ndcuray(cp.empty(shape,
                    dtype=dt,
                    order=order,
                    ))
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.empty(shape,
            dtype=dt,
            order=order,
            )


def arange(start, stop=None, step=None, dtype=None):
    if cp:
        try:
            return ndcuray(cp.arange(start, stop, step, dtype=dtype))
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arange(start, stop, step, dtype=dtype)


def flatiter():
    pass

#-------------------------------------------------------------------------------
# functions generated

def abs(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.abs(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.abs(x, out=out, casting=casting, dtype=dtype)

def absolute(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.absolute(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.absolute(x, out=out, casting=casting, dtype=dtype)

def add(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.add(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.add(x1, x2, out=out, casting=casting, dtype=dtype)

def all(a, axis=None, out=None, keepdims=False):
    if cp:
        try:
            v = cp.all(a, axis, out, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.all(a, axis, out, keepdims)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if cp:
        try:
            v = cp.allclose(a, b, rtol, atol, equal_nan)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.allclose(a, b, rtol, atol, equal_nan)

def amax(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.amax(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.amax(a, axis, out, keepdims, initial)

def amin(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.amin(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.amin(a, axis, out, keepdims, initial)

def angle(z, deg=False):
    if cp:
        try:
            v = cp.angle(z, deg)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.angle(z, deg)

def any(a, axis=None, out=None, keepdims=False):
    if cp:
        try:
            v = cp.any(a, axis, out, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.any(a, axis, out, keepdims)

def append(arr, values, axis=None):
    if cp:
        try:
            v = cp.append(arr, values, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.append(arr, values, axis)

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    if cp:
        try:
            v = cp.apply_along_axis(func1d, axis, arr, *args, **kwargs)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

# def arange(start, stop, step, dtype=None):
#     if cp:
#         try:
#             v = cp.arange(start, stop, step, dtype)
#             if v.ndim == 0:
#                 return v.item()
#             return ndcuray(v)
#         except cp.cuda.memory.OutOfMemoryError:
#             pass
#     return np.arange(start, stop, step, dtype)

def arccos(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arccos(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arccos(x, out=out, casting=casting, dtype=dtype)

def arccosh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arccosh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arccosh(x, out=out, casting=casting, dtype=dtype)

def arcsin(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arcsin(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arcsin(x, out=out, casting=casting, dtype=dtype)

def arcsinh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arcsinh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arcsinh(x, out=out, casting=casting, dtype=dtype)

def arctan(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arctan(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arctan(x, out=out, casting=casting, dtype=dtype)

def arctan2(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arctan2(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arctan2(x1, x2, out=out, casting=casting, dtype=dtype)

def arctanh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.arctanh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arctanh(x, out=out, casting=casting, dtype=dtype)

def argmax(a, axis=None, out=None, *, keepdims=False):
    if cp:
        try:
            v = cp.argmax(a, axis, out, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.argmax(a, axis, out, keepdims=keepdims)

def argmin(a, axis=None, out=None, *, keepdims=False):
    if cp:
        try:
            v = cp.argmin(a, axis, out, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.argmin(a, axis, out, keepdims=keepdims)

def argpartition(a, kth, axis=-1, kind='introselect'):
    if cp:
        try:
            v = cp.argpartition(a, kth, axis, kind)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.argpartition(a, kth, axis, kind)

def argsort(a, axis=-1, kind=None):
    if cp:
        try:
            v = cp.argsort(a, axis, kind)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.argsort(a, axis, kind)

def argwhere(a):
    if cp:
        try:
            v = cp.argwhere(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.argwhere(a)

def around(a, decimals=0, out=None):
    if cp:
        try:
            v = cp.around(a, decimals, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.around(a, decimals, out)

# def array(object, dtype=None, *, copy=True, ndmin=0):
#     if cp:
#         try:
#             v = cp.array(object, dtype, copy=copy, ndmin=ndmin)
#             if v.ndim == 0:
#                 return v.item()
#             return ndcuray(v)
#         except cp.cuda.memory.OutOfMemoryError:
#             pass
#     return np.array(object, dtype, copy=copy, ndmin=ndmin)

def array_equal(a1, a2, equal_nan=False):
    if cp:
        try:
            v = cp.array_equal(a1, a2, equal_nan)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.array_equal(a1, a2, equal_nan)

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if cp:
        return cp.array_repr(arr, max_line_width, precision, suppress_small)
    return np.array_repr(arr, max_line_width, precision, suppress_small)

def array_split(ary, indices_or_sections, axis=0):
    if cp:
        try:
            v = cp.array_split(ary, indices_or_sections, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.array_split(ary, indices_or_sections, axis)

def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    if cp:
        return cp.array_str(a, max_line_width, precision, suppress_small)
    return np.array_str(a, max_line_width, precision, suppress_small)

def asanyarray(a, dtype=None):
    if cp:
        try:
            v = cp.asanyarray(a, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.asanyarray(a, dtype)

def asarray(a, dtype=None):
    if cp:
        try:
            v = cp.asarray(a, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.asarray(a, dtype)

def ascontiguousarray(a, dtype=None):
    if cp:
        try:
            v = cp.ascontiguousarray(a, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ascontiguousarray(a, dtype)

def asfortranarray(a, dtype=None):
    if cp:
        try:
            v = cp.asfortranarray(a, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.asfortranarray(a, dtype)

def atleast_1d(*arys):
    if cp:
        try:
            v = cp.atleast_1d(*arys)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.atleast_1d(*arys)

def atleast_2d(*arys):
    if cp:
        try:
            v = cp.atleast_2d(*arys)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.atleast_2d(*arys)

def atleast_3d(*arys):
    if cp:
        try:
            v = cp.atleast_3d(*arys)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.atleast_3d(*arys)

def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    if cp:
        try:
            v = cp.average(a, axis, weights, returned, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.average(a, axis, weights, returned, keepdims=keepdims)

def bartlett(M):
    if cp:
        try:
            v = cp.bartlett(M)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bartlett(M)

def base_repr(number, base=2, padding=0):
    if cp:
        return cp.base_repr(number, base, padding)
    return np.base_repr(number, base, padding)

def binary_repr(num, width=None):
    if cp:
        return cp.binary_repr(num, width)
    return np.binary_repr(num, width)

def bincount(x, /, weights=None, minlength=0):
    if cp:
        try:
            v = cp.bincount(x, weights=weights, minlength=minlength)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bincount(x, weights=weights, minlength=minlength)

def bitwise_and(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.bitwise_and(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bitwise_and(x1, x2, out=out, casting=casting, dtype=dtype)

def bitwise_not(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.bitwise_not(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bitwise_not(x, out=out, casting=casting, dtype=dtype)

def bitwise_or(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.bitwise_or(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bitwise_or(x1, x2, out=out, casting=casting, dtype=dtype)

def bitwise_xor(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.bitwise_xor(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.bitwise_xor(x1, x2, out=out, casting=casting, dtype=dtype)

def blackman(M):
    if cp:
        try:
            v = cp.blackman(M)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.blackman(M)

def broadcast_arrays(*args):
    if cp:
        try:
            v = cp.broadcast_arrays(*args)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.broadcast_arrays(*args)

def broadcast_to(array, shape):
    if cp:
        try:
            v = cp.broadcast_to(array, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.broadcast_to(array, shape)

def can_cast(from_, to, casting='safe'):
    if cp:
        return cp.can_cast(from_, to, casting)
    return np.can_cast(from_, to, casting)

def cbrt(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.cbrt(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cbrt(x, out=out, casting=casting, dtype=dtype)

def ceil(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.ceil(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ceil(x, out=out, casting=casting, dtype=dtype)

def choose(a, choices, out=None, mode='raise'):
    if cp:
        try:
            v = cp.choose(a, choices, out, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.choose(a, choices, out, mode)

def clip(a, a_min, a_max, out=None, **kwargs):
    if cp:
        try:
            v = cp.clip(a, a_min, a_max, out, **kwargs)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.clip(a, a_min, a_max, out, **kwargs)

def column_stack(tup):
    if cp:
        try:
            v = cp.column_stack(tup)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.column_stack(tup)

def common_type(*arrays):
    if cp:
        return cp.common_type(*arrays)
    return np.common_type(*arrays)

def compress(condition, a, axis=None, out=None):
    if cp:
        try:
            v = cp.compress(condition, a, axis, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.compress(condition, a, axis, out)

def concatenate(*args):
    if cp:
        try:
            v = cp.concatenate(*args)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.concatenate(*args)

def conj(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.conj(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.conj(x, out=out, casting=casting, dtype=dtype)

def conjugate(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.conjugate(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.conjugate(x, out=out, casting=casting, dtype=dtype)

def convolve(a, v, mode='full'):
    if cp:
        try:
            v = cp.convolve(a, v, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.convolve(a, v, mode)

def copy(a):
    if cp:
        try:
            v = cp.copy(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.copy(a)

def copysign(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.copysign(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.copysign(x1, x2, out=out, casting=casting, dtype=dtype)

def copyto(dst, src, casting='same_kind'):
    if cp:
        try:
            v = cp.copyto(dst, src, casting)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.copyto(dst, src, casting)

def corrcoef(x, y=None, rowvar=True, bias=None, ddof=None, *, dtype=None):
    if cp:
        try:
            v = cp.corrcoef(x, y, rowvar, bias, ddof, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.corrcoef(x, y, rowvar, bias, ddof, dtype=dtype)

def correlate(a, v, mode='valid'):
    if cp:
        try:
            v = cp.correlate(a, v, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.correlate(a, v, mode)

def cos(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.cos(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cos(x, out=out, casting=casting, dtype=dtype)

def cosh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.cosh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cosh(x, out=out, casting=casting, dtype=dtype)

def count_nonzero(a, axis=None, *, keepdims=False):
    if cp:
        try:
            v = cp.count_nonzero(a, axis, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.count_nonzero(a, axis, keepdims=keepdims)

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    if cp:
        try:
            v = cp.cov(m, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cov(m, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if cp:
        try:
            v = cp.cross(a, b, axisa, axisb, axisc, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cross(a, b, axisa, axisb, axisc, axis)

def cumprod(a, axis=None, dtype=None, out=None):
    if cp:
        try:
            v = cp.cumprod(a, axis, dtype, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cumprod(a, axis, dtype, out)

def cumsum(a, axis=None, dtype=None, out=None):
    if cp:
        try:
            v = cp.cumsum(a, axis, dtype, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.cumsum(a, axis, dtype, out)

def deg2rad(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.deg2rad(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.deg2rad(x, out=out, casting=casting, dtype=dtype)

def degrees(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.degrees(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.degrees(x, out=out, casting=casting, dtype=dtype)

def diag(v, k=0):
    if cp:
        try:
            v = cp.diag(v, k)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diag(v, k)

def diag_indices(n, ndim=2):
    if cp:
        try:
            v = cp.diag_indices(n, ndim)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diag_indices(n, ndim)

def diag_indices_from(arr):
    if cp:
        try:
            v = cp.diag_indices_from(arr)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diag_indices_from(arr)

def diagflat(v, k=0):
    if cp:
        try:
            v = cp.diagflat(v, k)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diagflat(v, k)

def diagonal(a, offset=0, axis1=0, axis2=1):
    if cp:
        try:
            v = cp.diagonal(a, offset, axis1, axis2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diagonal(a, offset, axis1, axis2)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    if cp:
        try:
            v = cp.diff(a, n, axis, prepend, append)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.diff(a, n, axis, prepend, append)

def digitize(x, bins, right=False):
    if cp:
        try:
            v = cp.digitize(x, bins, right)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.digitize(x, bins, right)

def divide(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.divide(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.divide(x1, x2, out=out, casting=casting, dtype=dtype)

def divmod(x1, x2, out1, out2, /, out=(None, None)):
    if cp:
        try:
            v = cp.divmod(x1, x2, out1, out2, out=out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.divmod(x1, x2, out1, out2, out=out)

def dot(a, b, out=None):
    if cp:
        try:
            v = cp.dot(a, b, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.dot(a, b, out)

def dsplit(ary, indices_or_sections):
    if cp:
        try:
            v = cp.dsplit(ary, indices_or_sections)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.dsplit(ary, indices_or_sections)

def dstack(tup):
    if cp:
        try:
            v = cp.dstack(tup)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.dstack(tup)

def dtype(dtype, align=False, copy=False):
    if cp:
        try:
            v = cp.dtype(dtype, align, copy)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.dtype(dtype, align, copy)

def einsum(*operands, out=None, optimize=False, **kwargs):
    if cp:
        try:
            v = cp.einsum(*operands, out, optimize, **kwargs)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.einsum(*operands, out, optimize, **kwargs)

# def empty(shape, dtype=float):
#     if cp:
#         try:
#             v = cp.empty(shape, dtype)
#             if v.ndim == 0:
#                 return v.item()
#             return ndcuray(v)
#         except cp.cuda.memory.OutOfMemoryError:
#             pass
#     return np.empty(shape, dtype)

def empty_like(prototype, dtype=None, shape=None):
    if cp:
        try:
            v = cp.empty_like(prototype, dtype, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.empty_like(prototype, dtype, shape)

def equal(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.equal(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.equal(x1, x2, out=out, casting=casting, dtype=dtype)

def exp(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.exp(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.exp(x, out=out, casting=casting, dtype=dtype)

def exp2(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.exp2(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.exp2(x, out=out, casting=casting, dtype=dtype)

def expand_dims(a, axis):
    if cp:
        try:
            v = cp.expand_dims(a, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.expand_dims(a, axis)

def expm1(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.expm1(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.expm1(x, out=out, casting=casting, dtype=dtype)

def extract(condition, arr):
    if cp:
        try:
            v = cp.extract(condition, arr)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.extract(condition, arr)

def eye(N, M=None, k=0, dtype=float):
    if cp:
        try:
            v = cp.eye(N, M, k, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.eye(N, M, k, dtype)

def fill_diagonal(a, val, wrap=False):
    if cp:
        try:
            v = cp.fill_diagonal(a, val, wrap)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fill_diagonal(a, val, wrap)

def find_common_type(array_types, scalar_types):
    if cp:
        try:
            v = cp.find_common_type(array_types, scalar_types)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.find_common_type(array_types, scalar_types)

def fix(x, out=None):
    if cp:
        try:
            v = cp.fix(x, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fix(x, out)

def flatnonzero(a):
    if cp:
        try:
            v = cp.flatnonzero(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.flatnonzero(a)

def flip(m, axis=None):
    if cp:
        try:
            v = cp.flip(m, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.flip(m, axis)

def fliplr(m):
    if cp:
        try:
            v = cp.fliplr(m)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fliplr(m)

def flipud(m):
    if cp:
        try:
            v = cp.flipud(m)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.flipud(m)

def floor(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.floor(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.floor(x, out=out, casting=casting, dtype=dtype)

def floor_divide(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.floor_divide(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.floor_divide(x1, x2, out=out, casting=casting, dtype=dtype)

def fmax(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.fmax(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fmax(x1, x2, out=out, casting=casting, dtype=dtype)

def fmin(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.fmin(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fmin(x1, x2, out=out, casting=casting, dtype=dtype)

def fmod(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.fmod(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fmod(x1, x2, out=out, casting=casting, dtype=dtype)

def frexp(x, out1, out2, /, out=(None, None)):
    if cp:
        try:
            v = cp.frexp(x, out1, out2, out=out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.frexp(x, out1, out2, out=out)

def fromfile(file, dtype=float, count=-1, sep='', offset=0):
    if cp:
        try:
            v = cp.fromfile(file, dtype, count, sep, offset)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.fromfile(file, dtype, count, sep, offset)

def full(shape, fill_value, dtype=None):
    if cp:
        try:
            v = cp.full(shape, fill_value, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.full(shape, fill_value, dtype)

def full_like(a, fill_value, dtype=None, shape=None):
    if cp:
        try:
            v = cp.full_like(a, fill_value, dtype, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.full_like(a, fill_value, dtype, shape)

def gcd(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.gcd(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.gcd(x1, x2, out=out, casting=casting, dtype=dtype)

def gradient(f, *varargs, axis=None, edge_order=1):
    if cp:
        try:
            v = cp.gradient(f, *varargs, axis, edge_order)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.gradient(f, *varargs, axis, edge_order)

def greater(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.greater(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.greater(x1, x2, out=out, casting=casting, dtype=dtype)

def greater_equal(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.greater_equal(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.greater_equal(x1, x2, out=out, casting=casting, dtype=dtype)

def hamming(M):
    if cp:
        try:
            v = cp.hamming(M)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.hamming(M)

def hanning(M):
    if cp:
        try:
            v = cp.hanning(M)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.hanning(M)

def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
    if cp:
        try:
            v = cp.histogram(a, bins, range, normed, weights, density)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.histogram(a, bins, range, normed, weights, density)

def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    if cp:
        try:
            v = cp.histogram2d(x, y, bins, range, normed, weights, density)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.histogram2d(x, y, bins, range, normed, weights, density)

def histogramdd(sample, bins=10, range=None, normed=None, weights=None, density=None):
    if cp:
        try:
            v = cp.histogramdd(sample, bins, range, normed, weights, density)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.histogramdd(sample, bins, range, normed, weights, density)

def hsplit(ary, indices_or_sections):
    if cp:
        try:
            v = cp.hsplit(ary, indices_or_sections)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.hsplit(ary, indices_or_sections)

def hstack(tup):
    if cp:
        try:
            v = cp.hstack(tup)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.hstack(tup)

def hypot(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.hypot(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.hypot(x1, x2, out=out, casting=casting, dtype=dtype)

def i0(x):
    if cp:
        try:
            v = cp.i0(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.i0(x)

def identity(n, dtype=None):
    if cp:
        try:
            v = cp.identity(n, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.identity(n, dtype)

def imag(val):
    if cp:
        try:
            v = cp.imag(val)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.imag(val)

def in1d(ar1, ar2, assume_unique=False, invert=False):
    if cp:
        try:
            v = cp.in1d(ar1, ar2, assume_unique, invert)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.in1d(ar1, ar2, assume_unique, invert)

def indices(dimensions, dtype=int, sparse=False):
    if cp:
        try:
            v = cp.indices(dimensions, dtype, sparse)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.indices(dimensions, dtype, sparse)

def inner(a, b, /):
    if cp:
        try:
            v = cp.inner(a, b)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.inner(a, b)

def interp(x, xp, fp, left=None, right=None, period=None):
    if cp:
        try:
            v = cp.interp(x, xp, fp, left, right, period)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.interp(x, xp, fp, left, right, period)

def invert(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.invert(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.invert(x, out=out, casting=casting, dtype=dtype)

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if cp:
        try:
            v = cp.isclose(a, b, rtol, atol, equal_nan)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isclose(a, b, rtol, atol, equal_nan)

def iscomplex(x):
    if cp:
        try:
            v = cp.iscomplex(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.iscomplex(x)

def iscomplexobj(x):
    if cp:
        try:
            v = cp.iscomplexobj(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.iscomplexobj(x)

def isfinite(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.isfinite(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isfinite(x, out=out, casting=casting, dtype=dtype)

def isfortran(a):
    if cp:
        try:
            v = cp.isfortran(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isfortran(a)

def isin(element, test_elements, assume_unique=False, invert=False):
    if cp:
        try:
            v = cp.isin(element, test_elements, assume_unique, invert)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isin(element, test_elements, assume_unique, invert)

def isinf(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.isinf(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isinf(x, out=out, casting=casting, dtype=dtype)

def isnan(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.isnan(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isnan(x, out=out, casting=casting, dtype=dtype)

def isreal(x):
    if cp:
        try:
            v = cp.isreal(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isreal(x)

def isrealobj(x):
    if cp:
        try:
            v = cp.isrealobj(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isrealobj(x)

def isscalar(element):
    if cp:
        try:
            v = cp.isscalar(element)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.isscalar(element)

def issctype(rep):
    if cp:
        try:
            v = cp.issctype(rep)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.issctype(rep)

def issubclass_(arg1, arg2):
    if cp:
        try:
            v = cp.issubclass_(arg1, arg2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.issubclass_(arg1, arg2)

def issubdtype(arg1, arg2):
    if cp:
        try:
            v = cp.issubdtype(arg1, arg2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.issubdtype(arg1, arg2)

def issubsctype(arg1, arg2):
    if cp:
        try:
            v = cp.issubsctype(arg1, arg2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.issubsctype(arg1, arg2)

def ix_(*args):
    if cp:
        try:
            v = cp.ix_(*args)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ix_(*args)

def kaiser(M, beta):
    if cp:
        try:
            v = cp.kaiser(M, beta)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.kaiser(M, beta)

def kron(a, b):
    if cp:
        try:
            v = cp.kron(a, b)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.kron(a, b)

def lcm(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.lcm(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.lcm(x1, x2, out=out, casting=casting, dtype=dtype)

def ldexp(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.ldexp(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ldexp(x1, x2, out=out, casting=casting, dtype=dtype)

def left_shift(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.left_shift(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.left_shift(x1, x2, out=out, casting=casting, dtype=dtype)

def less(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.less(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.less(x1, x2, out=out, casting=casting, dtype=dtype)

def less_equal(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.less_equal(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.less_equal(x1, x2, out=out, casting=casting, dtype=dtype)

def lexsort(keys, axis=-1):
    if cp:
        try:
            v = cp.lexsort(keys, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.lexsort(keys, axis)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    if cp:
        try:
            v = cp.linspace(start, stop, num, endpoint, retstep, dtype, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.linspace(start, stop, num, endpoint, retstep, dtype, axis)

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000):
    if cp:
        try:
            v = cp.load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size=max_header_size)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size=max_header_size)

def log(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.log(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.log(x, out=out, casting=casting, dtype=dtype)

def log10(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.log10(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.log10(x, out=out, casting=casting, dtype=dtype)

def log1p(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.log1p(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.log1p(x, out=out, casting=casting, dtype=dtype)

def log2(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.log2(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.log2(x, out=out, casting=casting, dtype=dtype)

def logaddexp(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logaddexp(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logaddexp(x1, x2, out=out, casting=casting, dtype=dtype)

def logaddexp2(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logaddexp2(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logaddexp2(x1, x2, out=out, casting=casting, dtype=dtype)

def logical_and(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logical_and(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logical_and(x1, x2, out=out, casting=casting, dtype=dtype)

def logical_not(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logical_not(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logical_not(x, out=out, casting=casting, dtype=dtype)

def logical_or(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logical_or(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logical_or(x1, x2, out=out, casting=casting, dtype=dtype)

def logical_xor(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.logical_xor(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logical_xor(x1, x2, out=out, casting=casting, dtype=dtype)

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if cp:
        try:
            v = cp.logspace(start, stop, num, endpoint, base, dtype, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.logspace(start, stop, num, endpoint, base, dtype, axis)

def matmul(x1, x2, /, out=None, *, casting='same_kind', dtype=None, axes, axis):
    if cp:
        try:
            v = cp.matmul(x1, x2, out=out, casting=casting, dtype=dtype, axes=axes, axis=axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.matmul(x1, x2, out=out, casting=casting, dtype=dtype, axes=axes, axis=axis)

def max(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.max(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.max(a, axis, out, keepdims, initial)

def maximum(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.maximum(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.maximum(x1, x2, out=out, casting=casting, dtype=dtype)

def may_share_memory(a, b, /, max_work=None):
    if cp:
        return cp.may_share_memory(a, b, max_work=max_work)
    return np.may_share_memory(a, b, max_work=max_work)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if cp:
        try:
            v = cp.mean(a, axis, dtype, out, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.mean(a, axis, dtype, out, keepdims)

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if cp:
        try:
            v = cp.median(a, axis, out, overwrite_input, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.median(a, axis, out, overwrite_input, keepdims)

def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    if cp:
        try:
            v = cp.meshgrid(*xi, copy, sparse, indexing)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.meshgrid(*xi, copy, sparse, indexing)

def min(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.min(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.min(a, axis, out, keepdims, initial)

def min_scalar_type(a, /):
    if cp:
        try:
            v = cp.min_scalar_type(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.min_scalar_type(a)

def minimum(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.minimum(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.minimum(x1, x2, out=out, casting=casting, dtype=dtype)

def mintypecode(typechars, typeset='GDFgdf', default='d'):
    if cp:
        return cp.mintypecode(typechars, typeset, default)
    return np.mintypecode(typechars, typeset, default)

def mod(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.mod(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.mod(x1, x2, out=out, casting=casting, dtype=dtype)

def modf(x, out1, out2, /, out=(None, None)):
    if cp:
        try:
            v = cp.modf(x, out1, out2, out=out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.modf(x, out1, out2, out=out)

def moveaxis(a, source, destination):
    if cp:
        try:
            v = cp.moveaxis(a, source, destination)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.moveaxis(a, source, destination)

def msort(a):
    if cp:
        try:
            v = cp.msort(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.msort(a)

def multiply(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.multiply(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.multiply(x1, x2, out=out, casting=casting, dtype=dtype)

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if cp:
        try:
            v = cp.nan_to_num(x, copy, nan, posinf, neginf)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nan_to_num(x, copy, nan, posinf, neginf)

def nanargmax(a, axis=None, out=None, *, keepdims=False):
    if cp:
        try:
            v = cp.nanargmax(a, axis, out, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanargmax(a, axis, out, keepdims=keepdims)

def nanargmin(a, axis=None, out=None, *, keepdims=False):
    if cp:
        try:
            v = cp.nanargmin(a, axis, out, keepdims=keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanargmin(a, axis, out, keepdims=keepdims)

def nancumprod(a, axis=None, dtype=None, out=None):
    if cp:
        try:
            v = cp.nancumprod(a, axis, dtype, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nancumprod(a, axis, dtype, out)

def nancumsum(a, axis=None, dtype=None, out=None):
    if cp:
        try:
            v = cp.nancumsum(a, axis, dtype, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nancumsum(a, axis, dtype, out)

def nanmax(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.nanmax(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanmax(a, axis, out, keepdims, initial)

def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    if cp:
        try:
            v = cp.nanmean(a, axis, dtype, out, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanmean(a, axis, dtype, out, keepdims)

def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if cp:
        try:
            v = cp.nanmedian(a, axis, out, overwrite_input, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanmedian(a, axis, out, overwrite_input, keepdims)

def nanmin(a, axis=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.nanmin(a, axis, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanmin(a, axis, out, keepdims, initial)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.nanprod(a, axis, dtype, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanprod(a, axis, dtype, out, keepdims, initial)

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if cp:
        try:
            v = cp.nanstd(a, axis, dtype, out, ddof, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanstd(a, axis, dtype, out, ddof, keepdims)

def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.nansum(a, axis, dtype, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nansum(a, axis, dtype, out, keepdims, initial)

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if cp:
        try:
            v = cp.nanvar(a, axis, dtype, out, ddof, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nanvar(a, axis, dtype, out, ddof, keepdims)

def ndim(a):
    if cp:
        return cp.ndim(a)
    return np.ndim(a)

def negative(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.negative(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.negative(x, out=out, casting=casting, dtype=dtype)

def nextafter(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.nextafter(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nextafter(x1, x2, out=out, casting=casting, dtype=dtype)

def nonzero(a):
    if cp:
        try:
            v = cp.nonzero(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.nonzero(a)

def not_equal(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.not_equal(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.not_equal(x1, x2, out=out, casting=casting, dtype=dtype)

def obj2sctype(rep, default=None):
    if cp:
        return cp.obj2sctype(rep, default)
    return np.obj2sctype(rep, default)

def ones(shape, dtype=None):
    if cp:
        try:
            v = cp.ones(shape, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ones(shape, dtype)

def ones_like(a, dtype=None, shape=None):
    if cp:
        try:
            v = cp.ones_like(a, dtype, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ones_like(a, dtype, shape)

def outer(a, b, out=None):
    if cp:
        try:
            v = cp.outer(a, b, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.outer(a, b, out)

def packbits(a, /, axis=None, bitorder='big'):
    if cp:
        try:
            v = cp.packbits(a, axis=axis, bitorder=bitorder)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.packbits(a, axis=axis, bitorder=bitorder)

def pad(array, pad_width, mode='constant', **kwargs):
    if cp:
        try:
            v = cp.pad(array, pad_width, mode, **kwargs)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.pad(array, pad_width, mode, **kwargs)

def partition(a, kth, axis=-1, kind='introselect'):
    if cp:
        try:
            v = cp.partition(a, kth, axis, kind)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.partition(a, kth, axis, kind)

def percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    if cp:
        try:
            v = cp.percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation=interpolation)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation=interpolation)

def piecewise(x, condlist, funclist, *args, **kw):
    if cp:
        try:
            v = cp.piecewise(x, condlist, funclist, *args, **kw)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.piecewise(x, condlist, funclist, *args, **kw)

def place(arr, mask, vals):
    if cp:
        try:
            v = cp.place(arr, mask, vals)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.place(arr, mask, vals)

def polyadd(a1, a2):
    if cp:
        try:
            v = cp.polyadd(a1, a2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.polyadd(a1, a2)

def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    if cp:
        try:
            v = cp.polyfit(x, y, deg, rcond, full, w, cov)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.polyfit(x, y, deg, rcond, full, w, cov)

def polymul(a1, a2):
    if cp:
        try:
            v = cp.polymul(a1, a2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.polymul(a1, a2)

def polysub(a1, a2):
    if cp:
        try:
            v = cp.polysub(a1, a2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.polysub(a1, a2)

def polyval(p, x):
    if cp:
        try:
            v = cp.polyval(p, x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.polyval(p, x)

def power(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.power(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.power(x1, x2, out=out, casting=casting, dtype=dtype)

def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.prod(a, axis, dtype, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.prod(a, axis, dtype, out, keepdims, initial)

def promote_types(type1, type2):
    if cp:
        return cp.promote_types(type1, type2)
    return np.promote_types(type1, type2)

def ptp(a, axis=None, out=None, keepdims=False):
    if cp:
        try:
            v = cp.ptp(a, axis, out, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ptp(a, axis, out, keepdims)

def put(a, ind, v, mode='raise'):
    if cp:
        try:
            v = cp.put(a, ind, v, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.put(a, ind, v, mode)

def putmask(a, mask, values):
    if cp:
        try:
            v = cp.putmask(a, mask, values)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.putmask(a, mask, values)

def quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None):
    if cp:
        try:
            v = cp.quantile(a, q, axis, out, overwrite_input, method, keepdims, interpolation=interpolation)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.quantile(a, q, axis, out, overwrite_input, method, keepdims, interpolation=interpolation)

def rad2deg(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.rad2deg(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.rad2deg(x, out=out, casting=casting, dtype=dtype)

def radians(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.radians(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.radians(x, out=out, casting=casting, dtype=dtype)

def ravel(a):
    if cp:
        try:
            v = cp.ravel(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ravel(a)

def ravel_multi_index(multi_index, dims, mode='raise'):
    if cp:
        try:
            v = cp.ravel_multi_index(multi_index, dims, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.ravel_multi_index(multi_index, dims, mode)

def real(val):
    if cp:
        try:
            v = cp.real(val)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.real(val)

def reciprocal(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.reciprocal(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.reciprocal(x, out=out, casting=casting, dtype=dtype)

def remainder(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.remainder(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.remainder(x1, x2, out=out, casting=casting, dtype=dtype)

def repeat(a, repeats, axis=None):
    if cp:
        try:
            v = cp.repeat(a, repeats, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.repeat(a, repeats, axis)

def require(a, dtype=None, requirements=None):
    if cp:
        try:
            v = cp.require(a, dtype, requirements)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.require(a, dtype, requirements)

def reshape(a, newshape):
    if cp:
        try:
            v = cp.reshape(a, newshape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.reshape(a, newshape)

def resize(a, new_shape):
    if cp:
        try:
            v = cp.resize(a, new_shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.resize(a, new_shape)

def result_type(*arrays_and_dtypes):
    if cp:
        try:
            v = cp.result_type(*arrays_and_dtypes)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.result_type(*arrays_and_dtypes)

def right_shift(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.right_shift(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.right_shift(x1, x2, out=out, casting=casting, dtype=dtype)

def rint(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.rint(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.rint(x, out=out, casting=casting, dtype=dtype)

def roll(a, shift, axis=None):
    if cp:
        try:
            v = cp.roll(a, shift, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.roll(a, shift, axis)

def rollaxis(a, axis, start=0):
    if cp:
        try:
            v = cp.rollaxis(a, axis, start)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.rollaxis(a, axis, start)

def roots(p):
    if cp:
        try:
            v = cp.roots(p)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.roots(p)

def rot90(m, k=1, axes=(0, 1)):
    if cp:
        try:
            v = cp.rot90(m, k, axes, 1)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.rot90(m, k, axes, 1)

def round(a, decimals=0, out=None):
    if cp:
        try:
            v = cp.round(a, decimals, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.round(a, decimals, out)

def round_(a, decimals=0, out=None):
    if cp:
        try:
            v = cp.round_(a, decimals, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.round_(a, decimals, out)

def save(file, arr, allow_pickle=True, fix_imports=True):
    if cp:
        return cp.save(file, arr, allow_pickle, fix_imports)
    return np.save(file, arr, allow_pickle, fix_imports)

def savez(file, *args, **kwds):
    if cp:
        return cp.savez(file, *args, **kwds)
    return np.savez(file, *args, **kwds)

def savez_compressed(file, *args, **kwds):
    if cp:
        return cp.savez_compressed(file, *args, **kwds)
    return np.savez_compressed(file, *args, **kwds)

def sctype2char(sctype):
    if cp:
        return cp.sctype2char(sctype)
    return np.sctype2char(sctype)

def searchsorted(a, v, side='left', sorter=None):
    if cp:
        try:
            v = cp.searchsorted(a, v, side, sorter)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.searchsorted(a, v, side, sorter)

def select(condlist, choicelist, default=0):
    if cp:
        try:
            v = cp.select(condlist, choicelist, default)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.select(condlist, choicelist, default)

def shape(a):
    if cp:
        return cp.shape(a)
    return np.shape(a)

def shares_memory(a, b, /, max_work=None):
    if cp:
        return cp.shares_memory(a, b, max_work=max_work)
    return np.shares_memory(a, b, max_work=max_work)

def show_config():
    if cp:
        return cp.show_config()
    return np.show_config()

def sign(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.sign(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sign(x, out=out, casting=casting, dtype=dtype)

def signbit(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.signbit(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.signbit(x, out=out, casting=casting, dtype=dtype)

def sin(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.sin(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sin(x, out=out, casting=casting, dtype=dtype)

def sinc(x):
    if cp:
        try:
            v = cp.sinc(x)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sinc(x)

def sinh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.sinh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sinh(x, out=out, casting=casting, dtype=dtype)

def size(a, axis=None):
    if cp:
        return cp.size(a, axis)
    return np.size(a, axis)

def sort(a, axis=-1, kind=None):
    if cp:
        try:
            v = cp.sort(a, axis, kind)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sort(a, axis, kind)

def sort_complex(a):
    if cp:
        try:
            v = cp.sort_complex(a)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sort_complex(a)

def split(ary, indices_or_sections, axis=0):
    if cp:
        try:
            v = cp.split(ary, indices_or_sections, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.split(ary, indices_or_sections, axis)

def sqrt(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.sqrt(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sqrt(x, out=out, casting=casting, dtype=dtype)

def square(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.square(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.square(x, out=out, casting=casting, dtype=dtype)

def squeeze(a, axis=None):
    if cp:
        try:
            v = cp.squeeze(a, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.squeeze(a, axis)

def stack(arrays, axis=0, out=None):
    if cp:
        try:
            v = cp.stack(arrays, axis, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.stack(arrays, axis, out)

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if cp:
        try:
            v = cp.std(a, axis, dtype, out, ddof, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.std(a, axis, dtype, out, ddof, keepdims)

def subtract(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.subtract(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.subtract(x1, x2, out=out, casting=casting, dtype=dtype)

def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
    if cp:
        try:
            v = cp.sum(a, axis, dtype, out, keepdims, initial)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.sum(a, axis, dtype, out, keepdims, initial)

def swapaxes(a, axis1, axis2):
    if cp:
        try:
            v = cp.swapaxes(a, axis1, axis2)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.swapaxes(a, axis1, axis2)

def take(a, indices, axis=None, out=None, mode='raise'):
    if cp:
        try:
            v = cp.take(a, indices, axis, out, mode)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.take(a, indices, axis, out, mode)

def take_along_axis(arr, indices, axis):
    if cp:
        try:
            v = cp.take_along_axis(arr, indices, axis)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.take_along_axis(arr, indices, axis)

def tan(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.tan(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tan(x, out=out, casting=casting, dtype=dtype)

def tanh(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.tanh(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tanh(x, out=out, casting=casting, dtype=dtype)

def tensordot(a, b, axes=2):
    if cp:
        try:
            v = cp.tensordot(a, b, axes)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tensordot(a, b, axes)

def tile(A, reps):
    if cp:
        try:
            v = cp.tile(A, reps)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tile(A, reps)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if cp:
        try:
            v = cp.trace(a, offset, axis1, axis2, dtype, out)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.trace(a, offset, axis1, axis2, dtype, out)

def transpose(a, axes=None):
    if cp:
        try:
            v = cp.transpose(a, axes)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.transpose(a, axes)

def tri(N, M=None, k=0, dtype=float):
    if cp:
        try:
            v = cp.tri(N, M, k, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tri(N, M, k, dtype)

def tril(m, k=0):
    if cp:
        try:
            v = cp.tril(m, k)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.tril(m, k)

def trim_zeros(filt, trim='fb'):
    if cp:
        try:
            v = cp.trim_zeros(filt, trim)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.trim_zeros(filt, trim)

def triu(m, k=0):
    if cp:
        try:
            v = cp.triu(m, k)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.triu(m, k)

def true_divide(x1, x2, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.true_divide(x1, x2, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.true_divide(x1, x2, out=out, casting=casting, dtype=dtype)

def trunc(x, /, out=None, *, casting='same_kind', dtype=None):
    if cp:
        try:
            v = cp.trunc(x, out=out, casting=casting, dtype=dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.trunc(x, out=out, casting=casting, dtype=dtype)

def typename(char):
    if cp:
        try:
            v = cp.typename(char)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.typename(char)

def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True):
    if cp:
        try:
            v = cp.unique(ar, return_index, return_inverse, return_counts, axis, equal_nan=equal_nan)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.unique(ar, return_index, return_inverse, return_counts, axis, equal_nan=equal_nan)

def unpackbits(a, /, axis=None, count=None, bitorder='big'):
    if cp:
        try:
            v = cp.unpackbits(a, axis=axis, count=count, bitorder=bitorder)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.unpackbits(a, axis=axis, count=count, bitorder=bitorder)

def unravel_index(indices, shape):
    if cp:
        try:
            v = cp.unravel_index(indices, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.unravel_index(indices, shape)

def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):
    if cp:
        try:
            v = cp.unwrap(p, discont, axis, period=period)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.unwrap(p, discont, axis, period=period)

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if cp:
        try:
            v = cp.var(a, axis, dtype, out, ddof, keepdims)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.var(a, axis, dtype, out, ddof, keepdims)

def vdot(a, b, /):
    if cp:
        try:
            v = cp.vdot(a, b)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.vdot(a, b)

def vsplit(ary, indices_or_sections):
    if cp:
        try:
            v = cp.vsplit(ary, indices_or_sections)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.vsplit(ary, indices_or_sections)

def vstack(tup):
    if cp:
        try:
            v = cp.vstack(tup)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.vstack(tup)

def where(condition, x, y, /):
    if cp:
        try:
            v = cp.where(condition, x, y)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.where(condition, x, y)

def who(vardict=None):
    if cp:
        return cp.who(vardict)
    return np.who(vardict)

def zeros(shape, dtype=float):
    if cp:
        try:
            v = cp.zeros(shape, dtype)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.zeros(shape, dtype)

def zeros_like(a, dtype=None, shape=None):
    if cp:
        try:
            v = cp.zeros_like(a, dtype, shape)
            if v.ndim == 0:
                return v.item()
            return ndcuray(v)
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.zeros_like(a, dtype, shape)


