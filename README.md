# TAPS

[![Build](https://github.com/laviolette-lab/taps/actions/workflows/build.yml/badge.svg)](https://github.com/laviolette-lab/taps/actions/workflows/build.yml)
[![Tests](https://github.com/laviolette-lab/taps/actions/workflows/pytest.yml/badge.svg)](https://github.com/laviolette-lab/taps/actions/workflows/pytest.yml)
[![Lint](https://github.com/laviolette-lab/taps/actions/workflows/lint.yml/badge.svg)](https://github.com/laviolette-lab/taps/actions/workflows/lint.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/lavlab-taps.svg)](https://pypi.org/project/lavlab-taps)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lavlab-taps.svg)](https://pypi.org/project/lavlab-taps)

TAPS deploys a trained prostate MRI segmentation model as a Python package and command-line application. It accepts a NIfTI image and writes a binary NIfTI mask aligned with the original input image.

See https://github.com/laviolette-lab/TAPS-Training-Code for the original training code used to create this model and write the paper.

## Install

```console
pip install lavlab-taps
```

The installation includes PyTorch, MONAI, NumPy, and NiBabel. Install a CUDA-compatible PyTorch build first when GPU inference is required.

## CLI

```console
taps segment input_image.nii.gz prostate_mask.nii.gz
```

TAPS uses the bundled checkpoint by default and selects CUDA automatically when it is available. Override either with `--checkpoint` and `--device`:

```console
taps segment input_image.nii.gz prostate_mask.nii.gz \
	--checkpoint best_segresnet_model.pth --device cpu
```

The package applies the validation preprocessing pipeline, performs sliding-window inference, then restores the prediction to the source image's voxel space before saving it.

## Python API

```python
from taps import segment

segment(
    "input_image.nii.gz",
    "prostate_mask.nii.gz",
    checkpoint=None,  # use the bundled model
)
```

## Project Structure

```text
taps/
├── src/
│   └── taps/
│       ├── __init__.py          # Public API & version export
│       ├── __about__.py         # Version string
│       ├── cli.py               # CLI entry point (thin wrapper)
│       ├── inference.py         # Preprocessing, model loading & segmentation
│       ├── py.typed             # PEP 561 marker
│       └── resources/
│           └── model_v1.pth     # Bundled checkpoint
├── tests/
├── docs/                        # MkDocs source files
└── pyproject.toml
```

## Development

**Prerequisites:** Python 3.9+ and [Hatch](https://hatch.pypa.io/latest/install/).

```console
git clone https://github.com/laviolette-lab/taps.git
cd taps
pip install hatch
```

| Task | Command |
|------|---------|
| Run tests | `hatch run test:test` |
| Tests + coverage | `hatch run test:cov` |
| Lint | `hatch run lint:check` |
| Format | `hatch run lint:format` |
| Auto-fix lint | `hatch run lint:fix` |
| Type check | `hatch run types:check` |
| Build docs | `hatch run docs:build-docs` |
| Serve docs | `hatch run docs:serve-docs` |
| Build wheel | `hatch build` |

Or via the [`Makefile`](./Makefile): `make test`, `make lint`, `make build`, etc.

### Docker

```console
# Run tests via Docker
docker build --target hatch -t taps:hatch .
docker run --rm -e HATCH_ENV=test taps:hatch cov

# Production image (just the installed wheel)
docker build --target prod -t taps:prod .
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## License

`taps` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.