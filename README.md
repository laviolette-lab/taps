# TAPS

TAPS deploys the trained prostate MRI segmentation model as a Python package and command-line application. It accepts a NIfTI image and writes a binary NIfTI mask aligned with the original input image.

## Install

From this directory:

```console
pip install .
```

The installation includes PyTorch, MONAI, NumPy, and NiBabel. Install a CUDA-compatible PyTorch build first when GPU inference is required.

## CLI

```console
TAPS segment input_image.nii.gz prostate_mask.nii.gz \
	--checkpoint ../best_segresnet_model.pth
```

TAPS uses CUDA automatically when it is available. Select a device explicitly with `--device`, for example:

```console
TAPS segment input_image.nii.gz prostate_mask.nii.gz \
	--checkpoint ../best_segresnet_model.pth --device cpu
```

The package applies the validation preprocessing pipeline, performs sliding-window inference, then restores the prediction to the source image's voxel space before saving it.

## Python API

```python
from taps import segment

segment(
		"input_image.nii.gz",
		"prostate_mask.nii.gz",
		checkpoint="best_segresnet_model.pth",
)
```

## Development

```console
hatch run test:test
hatch run lint:check
hatch build
```

[![Build](https://github.com/LavLabInfrastructure/python-template/actions/workflows/build.yml/badge.svg)](https://github.com/LavLabInfrastructure/python-template/actions/workflows/build.yml)
[![Tests](https://github.com/LavLabInfrastructure/python-template/actions/workflows/pytest.yml/badge.svg)](https://github.com/LavLabInfrastructure/python-template/actions/workflows/pytest.yml)
[![Lint](https://github.com/LavLabInfrastructure/python-template/actions/workflows/pylint.yml/badge.svg)](https://github.com/LavLabInfrastructure/python-template/actions/workflows/pylint.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/python-template.svg)](https://pypi.org/project/python-template)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-template.svg)](https://pypi.org/project/python-template)

-----

A consistent, feature-rich template for Python projects. Use this as a starting point for new packages to get batteries-included tooling out of the box.

## What's Included

| Feature | Tool |
|---------|------|
| Build & environments | [Hatch](https://hatch.pypa.io/) with [hatch-pip-compile](https://github.com/juftin/hatch-pip-compile) |
| Linting & formatting | [Ruff](https://docs.astral.sh/ruff/) |
| Testing | [pytest](https://docs.pytest.org/) + [coverage](https://coverage.readthedocs.io/) |
| Type checking | [mypy](https://mypy.readthedocs.io/) |
| Documentation | [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) + [mkdocstrings](https://mkdocstrings.github.io/) |
| Containerization | Multi-stage [Dockerfile](./Dockerfile) (dev / hatch / prod) |
| CI/CD | [GitHub Actions](./.github/workflows/) with Dependabot |
| Dev environment | [Dev Container](./.devcontainer/) for VS Code / Codespaces |
| Git hygiene | [pre-commit](./.pre-commit-config.yaml) hooks |

## Quick Start

### Installation

```console
pip install python-template
```

### CLI

The template ships with an example CLI entry point:

```console
python-template --version
python-template example "hello world"
```

### As a Library

```python
from python_template import example

result = example("hello")
```

## Development Setup

**Prerequisites:** Python 3.9+ and [Hatch](https://hatch.pypa.io/latest/install/).

```console
git clone https://github.com/LavLabInfrastructure/python-template.git
cd python-template
pip install hatch
```

Optionally install pre-commit hooks:

```console
pip install pre-commit
pre-commit install
```

### Common Commands

Run these directly or use the provided [`Makefile`](./Makefile) shortcuts (e.g. `make test`, `make lint`).

| Task | Command |
|------|---------|
| Run tests | `hatch run test:test` |
| Tests + coverage | `hatch run test:cov` |
| Lint | `hatch run lint:check` |
| Format | `hatch run lint:format` |
| Auto-fix lint | `hatch run lint:fix` |
| Format + fix + lint | `hatch run lint:all` |
| Type check | `hatch run types:check` |
| Build docs | `hatch run docs:build-docs` |
| Serve docs | `hatch run docs:serve-docs` |
| Build wheel | `hatch build` |
| Clean artifacts | `make clean` |

### Docker

```console
# Run tests via Docker
docker build --target hatch -t myapp:hatch .
docker run --rm -e HATCH_ENV=test myapp:hatch cov

# Production image (just the installed wheel)
docker build --target prod -t myapp:prod .
```

## Project Structure

```text
python-template/
├── src/
│   └── python_template/        # Package source
│       ├── __init__.py          # Public API & version export
│       ├── __about__.py         # Version string
│       ├── cli.py               # CLI entry point (thin wrapper)
│       ├── example.py           # Example library module
│       └── py.typed             # PEP 561 marker
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   └── test_example.py          # Example tests
├── docs/                        # MkDocs source files
├── requirements/                # Locked deps (auto-generated by hatch-pip-compile)
├── .devcontainer/               # Dev container config
├── .github/
│   ├── workflows/               # CI workflows (build, test, lint)
│   └── dependabot.yml           # Auto-update deps + actions + Docker
├── pyproject.toml               # All project & tool configuration
├── Dockerfile                   # Multi-stage build
├── Makefile                     # Dev shortcuts
├── mkdocs.yml                   # Docs config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .editorconfig                # Editor consistency
└── .gitignore
```

## Design Philosophy

This template follows a **library-first** approach:

1. **All logic** lives in importable modules under `src/python_template/`.
2. **The CLI** (`cli.py`) is a thin `argparse` wrapper that delegates to library functions.
3. **Tests** call library functions directly — never through the CLI.

This keeps your code reusable whether it's called from the command line, another package, a notebook, or an API.

## Using This Template

1. Click **Use this template** on GitHub (or clone and reinitialize).
2. Rename `src/python_template/` to your package name.
3. Find-and-replace `python-template` → your project name and `python_template` → your package name.
4. Update `pyproject.toml` with your metadata (author, description, dependencies).
5. Update `LICENSE.txt` if needed.
6. Delete `example.py` and `test_example.py` once you have real code.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## License

`python-template` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.