import typing as tp

import numpy as np

try:
    import cupy as cp
    CuPyArray = cp.ndarray
except ImportError:
    cp = None
    CuPyArray = tp.Any


from hyray.curay import _DTYPE_KIND_CUPY
from hyray.curay import CuArray

UnionNpCuPy = tp.Union[np.ndarray, CuArray]



def ndarray(shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        ) -> UnionNpCuPy:
    dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
    if cp and buffer is None and dt.kind in _DTYPE_KIND_CUPY:
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

def empty(shape, dtype=float, order='C', *, like=None):
    if like is not None:
        raise NotImplementedError('`like` not supported')

    dt = dtype if hasattr(dtype, 'kind') else np.dtype(dtype)
    if cp and dt.kind in _DTYPE_KIND_CUPY:
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


def full(shape, fill_value, dtype=None, order='C', *, like=None):
    pass


def arange(start, stop=None, step=None, dtype=None, *, like=None):
    if like is not None:
        raise NotImplementedError('`like` not supported')

    if cp:
        try:
            return CuArray(cp.arange(start, stop, step, dtype=dtype))
        except cp.cuda.memory.OutOfMemoryError:
            pass
    return np.arange(start, stop, step, dtype=dtype)