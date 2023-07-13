import pytest
import numpy as np

from arrayhybrid.array import CuArray
from arrayhybrid.array import ArrayHybrid as ah

# pytest -W ignore::DeprecationWarning test/test_unit.py

def test_flags_a():
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


def test_flags_b():

    a1 = ah.ndarray((2, 4), dtype=bool)
    assert a1.flags.writeable == False

    a1.flags.writeable = False

    with pytest.raises(ValueError):
        a1.flags.writeable = True

#-------------------------------------------------------------------------------

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


