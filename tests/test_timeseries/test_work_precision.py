"""Tests for work-precision diagram tools."""

import probnumeval.timeseries as ts


def test_work_precision():
    wp = ts.WorkPrecision(algorithm=None)
    out = wp.evaluate(problem=None)
    assert isinstance(out, dict)
