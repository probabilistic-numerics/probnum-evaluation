import pytest

import probnumeval as pneval


def test_work_precision():
    wp = pneval.WorkPrecision(algorithm=None)
    out = wp.evaluate(problem=None)
    assert isinstance(out, dict)
