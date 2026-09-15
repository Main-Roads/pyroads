from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"Wheel does not exist: {wheel}")

    with zipfile.ZipFile(wheel) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise SystemExit(f"Corrupt wheel member: {bad_member}")

    environment = Path(tempfile.mkdtemp(prefix="pyroads-wheel-check-"))
    try:
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "numpy"],
            check=True,
        )
        subprocess.run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", str(wheel)],
            check=True,
        )

        check = """
import pathlib
import numpy as np
import pyroads
import pyroads._native as native

assert pyroads.__version__ == expected_version
assert pyroads.backend() == "rust"
assert pyroads.__backend__ == "rust"
assert pathlib.Path(native.__file__).resolve().suffix in {".pyd", ".so"}
result = native.cumulative_p(np.array([1.0, 2.0, 3.0]))
assert result.shape == (2,)
print(f"pyroads {pyroads.__version__}")
print(f"backend: {pyroads.backend()}")
print(f"native: {native.__file__}")
"""
        environment_vars = os.environ.copy()
        environment_vars.pop("PYTHONPATH", None)
        subprocess.run(
            [python, "-c", f"expected_version = {args.expected_version!r}\n{check}"],
            cwd=environment,
            env=environment_vars,
            check=True,
        )
    finally:
        shutil.rmtree(environment, ignore_errors=True)


if __name__ == "__main__":
    main()