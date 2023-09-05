import pytest
import hyray as hr
import numpy as np

# NOTE: this test can be run (and should pass) without CUDA

MODULES = (hr, np)

@pytest.mark.parametrize('mod', MODULES)
def test_abs(mod) -> None:
    a1 = mod.arange(12).reshape(3, 4) - 6
    a2 = hr.abs(a1, dtype=hr.int16)
    assert a2.tolist() == [[6, 5, 4, 3], [2, 1, 0, 1], [2, 3, 4, 5]]
    assert a2.dtype == hr.int16


@pytest.mark.parametrize('mod', MODULES)
def test_absolute(mod) -> None:
    a1 = mod.arange(12).reshape(3, 4) - 6
    a2 = hr.absolute(a1, dtype=hr.int16)
    assert a2.tolist() == [[6, 5, 4, 3], [2, 1, 0, 1], [2, 3, 4, 5]]
    assert a2.dtype == hr.int16


@pytest.mark.parametrize('mod', MODULES)
def test_add(mod) -> None:
    a1 = mod.arange(5, 8)
    a2 = mod.array([10, 0, 10])
    a3 = hr.add(a1, a2)
    assert a3.tolist() == [15, 6, 17]

@pytest.mark.parametrize('mod', MODULES)
def test_all(mod) -> None:
    a1 = mod.arange(12).reshape(3, 4) % 2 == 0
    a2 = hr.all(a1, axis=0)
    assert a2.tolist() == [True, False, True, False]
    assert a2.dtype == hr.bool_

    a3 = hr.all(a1, axis=1)
    assert a3.tolist() == [False, False, False]
    assert a3.dtype == hr.bool_

@pytest.mark.parametrize('mod', MODULES)
def test_allclose(mod) -> None:
    a1 = mod.array(10) / 2
    a2 = mod.array(10) / 2
    assert hr.allclose(a1, a2) == True



@pytest.mark.parametrize('mod', MODULES)
def test_concatenate(mod) -> None:
    a1 = mod.arange(6).reshape(3, 2) % 3
    a2 = mod.arange(6).reshape(3, 2) % 2

    a3 = hr.concatenate((a1, a2), axis=0)
    assert a3.tolist() == [[0, 1], [2, 0], [1, 2], [0, 1], [0, 1], [0, 1]]

    a4 = hr.concatenate((a1, a2), axis=1)
    assert a4.tolist() == [[0, 1, 0, 1], [2, 0, 0, 1], [1, 2, 0, 1]]
