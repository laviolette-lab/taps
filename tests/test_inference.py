"""Tests for TAPS inference utilities."""

import pytest

from taps.inference import resolve_checkpoint, segment


def test_resolve_checkpoint_returns_explicit_path(tmp_path):
    """An explicit checkpoint path is returned unchanged."""
    checkpoint = tmp_path / "custom.pth"

    assert resolve_checkpoint(checkpoint) == checkpoint


def test_resolve_checkpoint_falls_back_to_bundled_model():
    """With no override, the bundled checkpoint is used."""
    bundled = resolve_checkpoint(None)

    assert bundled.name == "model_v1.pth"
    assert bundled.is_file()


def test_segment_raises_for_missing_image(tmp_path):
    """A missing input image raises FileNotFoundError before any inference."""
    with pytest.raises(FileNotFoundError, match="Input image"):
        segment(tmp_path / "missing.nii.gz", tmp_path / "out.nii.gz", None)


def test_segment_raises_for_missing_checkpoint(tmp_path):
    """A missing checkpoint override raises FileNotFoundError."""
    image = tmp_path / "scan.nii.gz"
    image.touch()

    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        segment(image, tmp_path / "out.nii.gz", tmp_path / "missing.pth")


def test_segment_raises_for_invalid_output_suffix(tmp_path):
    """An output path without a NIfTI suffix raises ValueError."""
    image = tmp_path / "scan.nii.gz"
    image.touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="must end in"):
        segment(image, tmp_path / "out.txt", checkpoint)
