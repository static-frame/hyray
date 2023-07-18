import pytest
import hyray as hr
import numpy as np

MODULES = (hr, np)

@pytest.mark.parametrize('mod', MODULES)
def test_abs(mod) -> None:
    a1 = mod.arange(12).reshape(3, 4) - 6
    a2 = hr.abs(a1, dtype=hr.int16)
    assert a2.tolist() == [[6, 5, 4, 3], [2, 1, 0, 1], [2, 3, 4, 5]]
    assert a2.dtype == hr.int16


