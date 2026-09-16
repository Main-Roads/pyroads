from importlib.metadata import PackageNotFoundError, version

from ._backend import __backend__, backend


try:
	__version__ = version("pyroads")
except PackageNotFoundError:  # pragma: no cover - during local editing
	__version__ = "0.6.0"


__all__ = ["__backend__", "__version__", "backend"]
