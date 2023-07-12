import pytest
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