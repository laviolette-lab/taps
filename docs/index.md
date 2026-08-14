# TAPS

TAPS deploys a trained prostate MRI segmentation model as a Python package and command-line application. It accepts a NIfTI image and writes a binary NIfTI mask aligned with the original input image.

## Install

```console
pip install lavlab-taps
```

## CLI

```console
taps segment input_image.nii.gz prostate_mask.nii.gz
```

## Python API

```python
from taps import segment

segment("input_image.nii.gz", "prostate_mask.nii.gz", checkpoint=None)
```

See the [API Reference](api.md) for full details.
