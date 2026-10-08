import warnings

import pyroads
from pyroads import _backend


def test_backend_diagnostic_reports_a_supported_backend():
    assert pyroads.backend() in {"rust", "python"}
    assert pyroads.__backend__ == pyroads.backend()


def test_fallback_is_announced_once(monkeypatch):
    monkeypatch.setattr(_backend, "_fallback_announced", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            _backend.announce_fallback()
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)