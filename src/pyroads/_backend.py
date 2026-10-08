import warnings
from typing import Literal


try:
    import pyroads._native as _rust_native
except ImportError:
    _rust_native = None


Backend = Literal["rust", "python"]


def backend() -> Backend:
    """Return the active numeric backend."""
    return "rust" if _rust_native is not None else "python"


__backend__: Backend = backend()


_fallback_announced = False


def announce_fallback() -> None:
    global _fallback_announced
    if _fallback_announced:
        return
    _fallback_announced = True
    warnings.warn(
        "Rust binaries were not found; defaulting to the Numba/Python fallback backend.",
        RuntimeWarning,
        stacklevel=3,
    )