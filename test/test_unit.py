import pytest
import numpy as np

from arrayhybrid.array import CuArray
from arrayhybrid.array import ArrayHybrid as ah

# pytest -W ignore::DeprecationWarning test/test_unit.py

#-------------------------------------------------------------------------------
# test CuArrayFlags object

def test_ca_flags_a():
    a1 = ah.ndarray((2, 4), dtype=bool)
    assert a1.flags.writeable == False
    assert a1.flags.c_contiguous == True
    assert a1.flags.f_contiguous == False

    a2 = a1.transpose()
    assert a2.flags.writeable == False
    assert a2.flags.c_contiguous == False
    assert a2.flags.f_contiguous == True

    a3 = a1.T
    assert a3.flags.writeable == False
    assert a3.flags.c_contiguous == False
    assert a3.flags.f_contiguous == True


def test_ca_flags_b():

    a1 = ah.ndarray((2, 4), dtype=bool)
    assert a1.flags.writeable == False

    a1.flags.writeable = False

    with pytest.raises(ValueError):
        a1.flags.writeable = True

#-------------------------------------------------------------------------------
# test CuArray object

def test_ca_setitem_a():
    a1 = ah.array([8, 3, 5])
    with pytest.raises(NotImplementedError):
        a1[0] = 20

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

def test_ca_reshape_a():
    assert ah.arange(4).reshape((2, 2)).ndim == 2
    assert ah.arange(6).reshape(2, 3).shape == (2, 3)
    assert ah.arange(12).reshape(2, 3, 2).shape == (2, 3, 2)

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

