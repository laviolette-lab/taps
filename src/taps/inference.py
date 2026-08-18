"""Inference utilities for the TAPS prostate-segmentation model."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from monai.inferers.utils import sliding_window_inference
from monai.networks.nets.segresnet_ds import SegResNetDS
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.post.dictionary import Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.networks.layers.simplelayers import GaussianFilter
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VOXEL_SPACING = (0.8, 0.8, 3.0)
ROI_SIZE = (128, 128, 32)
DEFAULT_SIGMA = 2.5
DEFAULT_PROBABILITY_THRESHOLD = 0.55
MIN_PROSTATE_VOLUME_CC = 20.0
MAX_PROSTATE_VOLUME_CC = 55.0
MIN_GEOMETRY_EXTENT_MM = 10.0
MAX_GEOMETRY_EXTENT_MM = 150.0

class ONNXPredictor:
    def __init__(self, model_path: Path, device: str):
        options = ort.SessionOptions()
        options.log_severity_level = 3 
        
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is already on the CPU now, so .numpy() is instant
        ort_outs = self.session.run(None, {self.input_name: x.numpy()})
        # Return a CPU tensor for MONAI to stitch
        return torch.from_numpy(ort_outs[0])
    
class BlankMaskError(RuntimeError):
    """Raised after a segmentation produces no foreground voxels."""


def build_model(device: torch.device) -> SegResNetDS:
    """Create the network architecture used by the released TAPS checkpoint."""
    return SegResNetDS(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=1,
        norm=("GROUP", {"num_groups": 8}),
        act=("MISH", {"inplace": True}),
        dsdepth=4,
    ).to(device)


def build_preprocessing() -> Compose:
    """Return the preprocessing pipeline used during TAPS validation."""
    return Compose(
        [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image", dtype=torch.float32),
            Orientationd(keys="image", axcodes="RAS", labels=None),
            Spacingd(keys="image", pixdim=VOXEL_SPACING, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys="image", source_key="image", margin=5),
        ]
    )


def load_model(checkpoint: str | Path, device: torch.device) -> SegResNetDS:
    """Load a TAPS checkpoint, including checkpoints saved from compiled models."""
    model = build_model(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def gaussian_blur_logits(
    logits: torch.Tensor, sigma: float = DEFAULT_SIGMA
) -> torch.Tensor:
    """Apply Gaussian blur natively on the GPU."""
    if sigma <= 0.0 or logits.ndim != 5:
        return logits

    blur_filter = GaussianFilter(spatial_dims=3, sigma=sigma).to(logits.device)
    return blur_filter(logits)


def resolve_checkpoint(checkpoint: str | Path | None) -> Path:
    """Resolve a checkpoint path, falling back to the packaged model when none is provided."""
    if checkpoint is not None:
        return Path(checkpoint)

    bundled_checkpoint = Path(__file__).resolve().parent / "resources" / "model_v1.onnx"
    if bundled_checkpoint.is_file():
        return bundled_checkpoint

    raise FileNotFoundError(
        "Bundled checkpoint not found; pass --checkpoint to provide a model file explicitly."
    )


def _with_suffix(path: Path, suffix: str) -> Path:
    """Insert a suffix before the NIfTI extension."""
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}{suffix}.nii.gz")
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the largest 3D foreground component and its component count."""
    labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3, 3)))
    if count <= 1:
        return mask, count
    sizes = np.bincount(labels.ravel())[1:]
    return (labels == (np.argmax(sizes) + 1)).astype(np.uint8), count


def _clean_slice_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Blank slices containing more than one 2D foreground component."""
    cleaned = mask.copy()
    abnormal_slices = 0
    for slice_index in range(mask.shape[2]):
        _, count = ndimage.label(
            mask[:, :, slice_index] > 0, structure=np.ones((3, 3))
        )
        if count > 1:
            cleaned[:, :, slice_index] = 0
            abnormal_slices += 1
    return cleaned, abnormal_slices


def _warn_geometry(mask: np.ndarray, affine: np.ndarray) -> bool:
    """Warn when mask geometry or physical extents are implausible."""
    if mask.ndim != 3 or affine.shape != (4, 4):
        logger.warning("Warning: segmentation has invalid mask or affine geometry.")
        return True

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    occupied = np.argwhere(mask > 0)
    if (
        not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
        or not np.all(np.isfinite(affine))
        or occupied.size == 0
    ):
        logger.warning("Warning: segmentation has invalid voxel geometry.")
        return True

    extent_mm = (occupied.max(axis=0) - occupied.min(axis=0) + 1) * spacing
    if np.any(
        (extent_mm < MIN_GEOMETRY_EXTENT_MM) | (extent_mm > MAX_GEOMETRY_EXTENT_MM)
    ):
        formatted_extent = ", ".join(f"{extent:.1f}" for extent in extent_mm)
        logger.warning(
            f"Warning: segmentation physical extent ({formatted_extent} mm) "
            "is outside the expected 10-150 mm range."
        )
        return True
    return False


def _warn_border_contact(mask: np.ndarray) -> bool:
    """Warn when the foreground touches any image boundary."""
    touches_border = any(
        np.any(face)
        for axis in range(3)
        for face in (np.take(mask > 0, 0, axis), np.take(mask > 0, -1, axis))
    )
    if touches_border:
        logger.warning("Warning: segmentation touches the image border.")
    return touches_border


def _warn_slice_continuity(mask: np.ndarray) -> bool:
    """Warn when occupied slices contain gaps or isolated single slices."""
    occupied_slices = np.flatnonzero(np.any(mask > 0, axis=(0, 1)))
    if occupied_slices.size < 2:
        return False
    gaps = np.diff(occupied_slices)
    has_gap = bool(np.any(gaps > 1))
    if has_gap:
        logger.warning("Warning: segmentation has gaps between occupied slices.")

    runs = np.split(occupied_slices, np.flatnonzero(gaps > 1) + 1)
    has_isolated_slice = any(run.size == 1 for run in runs)
    if has_isolated_slice:
        logger.warning("Warning: segmentation contains isolated occupied slices.")
    return has_gap or has_isolated_slice


def _warn_holes(mask: np.ndarray) -> bool:
    """Warn when foreground encloses one or more background regions."""
    holes = ndimage.binary_fill_holes(mask > 0) & ~(mask > 0)
    if np.any(holes):
        logger.warning(
            f"Warning: segmentation contains {int(ndimage.label(holes)[1])} "
            "enclosed hole(s)."
        )
        return True
    return False


def _run_qc(
    mask: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    exact: bool,
) -> tuple[np.ndarray, bool, Path]:
    """Run mask QC and return the 3D-cleaned mask and abnormal status."""
    voxel_volume_cc = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    volume_cc = float(np.count_nonzero(mask) * voxel_volume_cc)
    abnormal = False
    abnormal |= _warn_geometry(mask, affine)
    abnormal |= _warn_border_contact(mask)
    abnormal |= _warn_slice_continuity(mask)
    abnormal |= _warn_holes(mask)
    if not MIN_PROSTATE_VOLUME_CC <= volume_cc <= MAX_PROSTATE_VOLUME_CC:
        logger.warning(
            f"Warning: prostate volume {volume_cc:.1f} cc is outside "
            f"the expected {MIN_PROSTATE_VOLUME_CC:.0f}-{MAX_PROSTATE_VOLUME_CC:.0f} cc range."
        )
        abnormal = True

    cleaned_mask, component_count = _largest_component(mask)
    if component_count > 1:
        logger.warning(
            f"Warning: segmentation contains {component_count} disconnected "
            "3D components; keeping the largest component."
        )
        abnormal = True

    slice_cleaned, abnormal_slices = _clean_slice_components(mask)
    if abnormal_slices:
        logger.warning(
            f"Warning: segmentation contains multiple 2D components in "
            f"{abnormal_slices} slice(s); writing a cleaned mask."
        )
        cleaned_path = _with_suffix(output_path, "_cleaned")
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(slice_cleaned & cleaned_mask, affine), cleaned_path)
        logger.info("Saved cleaned segmentation to %s", cleaned_path)
        abnormal = True

    if abnormal and not exact:
        output_path = _with_suffix(output_path, "_abnormal")
    return cleaned_mask, abnormal, output_path


def segment(
    image_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path | None,
    device: str | None = None,
    exact: bool = False,
    *,
    sigma: float = DEFAULT_SIGMA,
    threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> Path:
    """Segment a prostate MRI and write a binary NIfTI mask in native image space.

    :param image_path: Source NIfTI image (.nii or .nii.gz).
    :param output_path: Destination NIfTI segmentation mask.
    :param checkpoint: Path to a TAPS SegResNetDS checkpoint.
    :param device: Torch device override. Defaults to CUDA when available.
    :param exact: Keep the requested output path even when QC finds an abnormality.
    :param sigma: Gaussian blur applied to logits before thresholding.
    :param threshold: Foreground probability threshold used for the final mask.
    :return: The written output path.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    checkpoint = resolve_checkpoint(checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not str(output_path).endswith((".nii", ".nii.gz")):
        raise ValueError("Output path must end in .nii or .nii.gz")
    
    # 1. Determine the best available hardware
    if device is None:
        if torch.cuda.is_available():
            target_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target_device = "mps"
        else:
            target_device = "cpu"
    else:
        target_device = device
        
    torch_device = torch.device(target_device)
    preprocess = build_preprocessing()
    data = cast(dict[Hashable, Any], preprocess({"image": str(image_path)}))
    
    torch_device = torch.device("cpu")
    
    # Initialize ONNX with the actual target hardware (e.g., "mps" or "cuda")
    # You can pass "mps" directly here since you are on a Mac
    model = ONNXPredictor(resolve_checkpoint(checkpoint), device=target_device) 

    # Keep the image on the CPU!
    image = data["image"].unsqueeze(0)
    
    with torch.inference_mode():
        logits = sliding_window_inference(
            image,
            roi_size=ROI_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            mode="gaussian",
        )

    blurred_logits = gaussian_blur_logits(logits, sigma=sigma)
    data["pred"] = (
        (torch.sigmoid(blurred_logits[0]) > threshold).to(torch.float32).cpu()
    )
    inverted = Invertd(
        keys="pred",
        transform=preprocess,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )(data)

    prediction = inverted["pred"].cpu().numpy()[0].astype(np.uint8)
    affine = np.asarray(inverted["pred"].meta["affine"])
    if not np.any(prediction):
        raise BlankMaskError("Inferred segmentation mask is blank")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction, _, output_path = _run_qc(prediction, affine, output_path, exact)
    nib.save(nib.Nifti1Image(prediction, affine), output_path)
    return output_path
