import pytest

from probnumeval.timeseries import intersample_rmse, normaltest


def test_intersample_rmse():
    with pytest.raises(NotImplementedError):
        intersample_rmse(None)


def test_normaltest():
    with pytest.raises(NotImplementedError):
        normaltest(None)
