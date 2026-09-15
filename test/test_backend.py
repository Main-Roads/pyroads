import pyroads


def test_backend_diagnostic_reports_a_supported_backend():
    assert pyroads.backend() in {"rust", "python"}
    assert pyroads.__backend__ == pyroads.backend()