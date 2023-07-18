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
        like=None,
        ) -> UnionNpCuRay:
    if like is not None:
        raise NotImplementedError('`like` not supported')
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

def empty(shape, dtype=float, order='C', *, like=None):
    if like is not None:
        raise NotImplementedError('`like` not supported')

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


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    pass


def arange(start, stop=None, step=None, dtype=None, *, like=None):
    if like is not None:
        raise NotImplementedError('`like` not supported')

    if cp:
        try:
            return ndcuray(cp.arange(start, stop, step, dtype=dtype))
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arange(start, stop, step, dtype=dtype)


def flatiter():
    pass