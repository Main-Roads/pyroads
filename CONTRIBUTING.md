# Contributing to pyroads

## Before you start

Please open an issue for substantial changes so the proposed API or behavior
can be discussed first. Small fixes and documentation improvements can go
straight to a pull request.

## Development setup

```bash
uv sync
uv run pytest
uv run ruff check src
uv run pyright src
```

The package supports Python 3.10 and newer. Numba is installed as a required
dependency because it is used by the default implementation. Ruff checks
Python style and likely errors; Pyright checks type consistency. Both are
included in the development dependencies.

Pre-commit runs the same checks before a commit. Install its hooks once after
setting up the environment:

```bash
uv run pre-commit install
```

To run them manually across the repository:

```bash
uv run pre-commit run --all-files
```

## Making changes

- Keep public APIs and existing behavior stable unless the change is
  intentional and documented.
- Add or update focused tests with every behavior change.
- Benchmark changes to interval merging or segmentation using the examples in
  `examples/`.
- Keep changes scoped and avoid committing generated caches or local files.
- Update `README.md` or `CHANGELOG.md` when a change affects users.

## Pull requests

Before opening a pull request:

1. Run `uv run pytest`.
2. Run `uv run ruff check src`.
3. Run `uv run pyright src`.
4. Build the package with `uv build`.
5. Check the working tree for unintended files with `git status`.
6. Describe the user-visible change and any performance impact.

Pull requests should explain compatibility considerations and include tests
for regressions. Maintainers may ask for additional benchmark results for
changes to optimized paths.

## License

Contributions are made under the BSD 3-Clause License in `LICENSE`. By
submitting a contribution, you confirm that you have the right to submit it
under that license.