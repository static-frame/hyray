import pytest
import numpy as np

from arrayhybrid.array import CuArray
from arrayhybrid.array import ArrayHybrid as ah

# pytest -W ignore::DeprecationWarning test/test_unit.py

#-------------------------------------------------------------------------------
# test CuArrayFlags object

def test_ca_flags_a():
    a1 = ah.ndarray((2, 4), dtype=bool)
    assert a1.flags.writeable == True
    assert a1.flags.c_contiguous == True
    assert a1.flags.f_contiguous == False

    a2 = a1.transpose()
    assert a2.flags.c_contiguous == False
    assert a2.flags.f_contiguous == True

    a3 = a1.T
    assert a3.flags.c_contiguous == False
    assert a3.flags.f_contiguous == True


def test_ca_flags_b():
    a1 = ah.ndarray((2, 4), dtype=bool)
    assert a1.flags.writeable == True
    a1.flags.writeable = False
    assert a1.flags.writeable == False

#-------------------------------------------------------------------------------
# test CuArray object

def test_ca_setitem_a():
    a1 = ah.array([8, 3, 5])
    a1[0] = 20
    assert a1.tolist() == [20, 3, 5]
    a1.flags.writeable = False
    with pytest.raises(ValueError):
        a1[0] = 2
    assert a1.tolist() == [20, 3, 5]

#-------------------------------------------------------------------------------

def test_ca_iter_a():
    a1 = ah.array([8, 3, 5])
    assert list(a1) == [8, 3, 5]
    assert list(a1)[0].__class__ == int

def test_ca_iter_b():
    a1 = ah.array([8, 3, 5])
    assert list(a1) == [8, 3, 5]
    assert list(a1)[0].__class__ == int

def test_ca_iter_c():
    a1 = ah.arange(6).reshape((2, 3))
    post = list(a1)
    assert [p.__class__ for p in post] == [CuArray, CuArray]
    assert [p.shape for p in post] == [(3,), (3,)]

#-------------------------------------------------------------------------------

def test_ca_len_a():
    assert len(ah.array([8, 3, 5])) == 3

def test_ca_ndim_a():
    assert ah.array([8, 3, 5]).ndim == 1

def test_ca_ndim_b():
    assert ah.arange(4).reshape((2, 2)).ndim == 2

#-------------------------------------------------------------------------------

def test_ca_flat_a():
    a1 = ah.arange(6).reshape(3, 2)
    assert list(a1.flat) == [0, 1, 2, 3, 4, 5]

#-------------------------------------------------------------------------------

def test_ca_imag_a():
    a1 = ah.arange(3)
    assert list(a1.imag) == [0, 0, 0]

#-------------------------------------------------------------------------------

def test_ca_itemsize_a():
    a1 = ah.arange(3, dtype=np.int8)
    assert a1.itemsize == 1

#-------------------------------------------------------------------------------

def test_ca_nbytes_a():
    a1 = ah.arange(3, dtype=np.int8)
    assert a1.nbytes == 3

#-------------------------------------------------------------------------------

def test_ca_real_a():
    a1 = ah.arange(3)
    assert list(a1.real) == [0, 1, 2]

#-------------------------------------------------------------------------------

def test_ca_size_a():
    a1 = ah.arange(3, dtype=np.int8)
    assert a1.size == 3

#-------------------------------------------------------------------------------

def test_ca_strides_a():
    a1 = ah.arange(20, dtype=np.int8).reshape(2, 10)
    assert a1.strides == (10, 1)

#-------------------------------------------------------------------------------

def test_ca_reshape_a():
    assert ah.arange(4).reshape((2, 2)).ndim == 2
    assert ah.arange(6).reshape(2, 3).shape == (2, 3)
    assert ah.arange(12).reshape(2, 3, 2).shape == (2, 3, 2)

#-------------------------------------------------------------------------------

def test_ca_abs_a():
    assert abs(ah.array([-1, 4, -3])).tolist() == [1, 4, 3]

#-------------------------------------------------------------------------------

def test_ca_add_a():
    assert (ah.array([-1, 4, -3]) + 10).tolist() == [9, 14, 7]

#-------------------------------------------------------------------------------

def test_ca_and_a():
    assert (ah.array([True, False]) & False).tolist() == [False, False]

#-------------------------------------------------------------------------------

def test_ca_array_a():
    a1 = ah.array([True, False]).__array__()
    assert a1.__class__ is np.ndarray

def test_ca_array_b():
    a1 = ah.array([True, False]).__array__(int)
    assert a1.__class__ is np.ndarray
    assert a1.dtype == int

def test_ca_array_c():
    a1 = ah.arange(12).reshape((2, 6), order='F')
    assert a1.flags.f_contiguous == True
    a2 = a1.__array__()
    assert a2.__class__ is np.ndarray
    assert a2.flags.f_contiguous == True

#-------------------------------------------------------------------------------
# CuArray.__array_function__

def test_ca_array_function_a():
    assert np.sum(ah.arange(4)) == 6

def test_ca_array_function_b():
    a1 = ah.arange(8).reshape(2, 4)
    a2 = np.sum(a1, axis=1)
    assert a2.__class__ is CuArray
    assert a2.shape == (2,)

#-------------------------------------------------------------------------------

def test_ca_astype_a():
    a1 = ah.array([1, 0, 1]).astype(bool)
    assert a1.__class__ is CuArray
    assert a1.dtype == bool
    assert a1.tolist() == [True, False, True]

def test_ca_astype_b():
    a1 = ah.array([1, 0, 1]).astype(np.float32)
    assert a1.__class__ is CuArray
    assert a1.dtype == np.float32
    assert a1.tolist() == [1., 0., 1.]

def test_ca_astype_c():
    a1 = ah.array([1, 0, 1, 1]).reshape(2, 2).astype(bool, order='F')
    assert a1.__class__ is CuArray
    assert a1.dtype == bool
    assert a1.flags.f_contiguous == True
    assert a1.flags.c_contiguous == False

#-------------------------------------------------------------------------------

def test_ca_bool_a():
    a1 = ah.array([True, False])
    with pytest.raises(ValueError):
        bool(a1)

#-------------------------------------------------------------------------------

def test_ca_complex_a():
    assert ah.array([3]).__complex__() == 3+0j

#-------------------------------------------------------------------------------

def test_ca_copy_a():
    a1 = ah.arange(3)
    a2 = a1.__copy__()
    assert id(a1) != id(a2)

#-------------------------------------------------------------------------------

def test_ca_deepcopy_a():
    a1 = ah.arange(3)
    a2 = a1.__deepcopy__({})
    assert id(a1) != id(a2)

#-------------------------------------------------------------------------------

def test_ca_dir_a():
    a1 = ah.arange(3)
    post = a1.__dir__()
    assert '__repr__' in post

#-------------------------------------------------------------------------------

def test_ca_divmod_a():
    a1 = ah.array([6, 3, 8])
    a2, a3 = a1.__divmod__(3)
    assert a2.__class__ is CuArray
    assert a2.tolist() == [2, 1, 2]
    assert a3.__class__ is CuArray
    assert a3.tolist() == [0, 0, 2]

#-------------------------------------------------------------------------------

def test_ca_rdivmod_a():
    a1 = ah.array([6, 3, 8])
    a2, a3 = a1.__rdivmod__(3)
    assert a2.__class__ is CuArray
    assert a2.tolist() == [0, 1, 0]
    assert a3.__class__ is CuArray
    assert a3.tolist() == [3, 0, 3]

#-------------------------------------------------------------------------------

# not available in cupy-cuda101
# def test_ca_dlpack_a():
#     a1 = ah.array([6, 3, 8])
#     assert a1.__dlpack__().__class__.__name__ == 'PyCapsule'

# def test_ca_dlpack_device_a():
#     a1 = ah.array([6, 3, 8])
#     assert a1.__dlpack_device__().__class__ is tuple

#-------------------------------------------------------------------------------

def test_ca_getitem_a():
    a1 = ah.arange(12).reshape((3, 4))
    a2 = a1[2]
    assert a2.__class__ is CuArray
    assert a2.tolist() == [8, 9, 10, 11]

    a3 = a1[:, 3]
    assert a3.__class__ is CuArray
    assert a3.tolist() == [3, 7, 11]

    a4 = a1[2, 3]
    assert a4.__class__ is int
    assert a4 == 11

#-------------------------------------------------------------------------------
# '__invert__',
# __neg__
# '__pos__',

# '__itruediv__'
# __matmul__
# '__rmatmul__',

def test_ca_magic():
    for attr in (
            '__floordiv__',
            '__ge__',
            '__gt__',
            '__iadd__',
            '__iand__',
            '__ifloordiv__',
            '__ilshift__',
            '__imod__',
            '__imul__',
            '__ior__',
            '__ipow__',
            '__irshift__',
            '__isub__',
            '__ixor__',
            '__le__',
            '__lshift__',
            '__lt__',
            '__mod__',
            '__mul__',
            '__ne__',
            '__pow__',
            '__radd__',
            '__rand__',
            '__rfloordiv__',
            '__rmod__',
            '__rmul__',
            '__ror__',
            '__rpow__',
            '__rrshift__',
            '__rshift__',
            '__rsub__',
            '__rtruediv__',
            '__rxor__',
            '__sub__',
            '__truediv__',
            '__xor__',
            ):
        ca = getattr(ah.arange(1, 9), attr)
        na = getattr(np.arange(1, 9), attr)
        assert ca(4).tolist() == na(4).tolist()


#-------------------------------------------------------------------------------
# test of module interface

def test_array_a():
    # with cp
    a1 = ah.array([10, 20])
    assert a1.__class__ is CuArray

    a2 = ah.array(['10', '20'])
    assert a2.tolist() == ['10', '20']
    assert a2.__class__ is np.ndarray

    a3 = ah.array(['2021', '2022'], dtype=np.datetime64)
    assert a3.__class__ is np.ndarray
    assert list(a3) == [np.datetime64('2021'), np.datetime64('2022')]


def test_array_b():
    # with cp
    with pytest.raises(NotImplementedError):
        a1 = ah.array([10, 20], like='foo')

#-------------------------------------------------------------------------------

def test_empty_a():
    with pytest.raises(NotImplementedError):
        a1 = ah.empty(20, like='foo')

def test_empty_b():
    a1 = ah.empty(2)
    assert a1.__class__ is CuArray

    a2 = ah.empty(2, dtype='U2')
    assert a2.__class__ is np.ndarray

#-------------------------------------------------------------------------------

def test_arange_a():
    a1 = ah.arange(6)
    assert a1.__class__ is CuArray
    assert a1.tolist() == np.arange(6).tolist()

def test_arange_b():
    a1 = ah.arange(1, 4)
    assert a1.__class__ is CuArray
    assert a1.tolist() == [1, 2, 3]

def test_arange_c():
    a1 = ah.arange(0, 10, 3)
    assert a1.__class__ is CuArray
    assert a1.tolist() == [0, 3, 6, 9]

