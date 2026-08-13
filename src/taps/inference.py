"""Inference utilities for the TAPS prostate-segmentation model."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from monai.inferers.utils import sliding_window_inference
from monai.networks.nets.segresnet_ds import SegResNetDS
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.post.dictionary import Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped

VOXEL_SPACING = (0.8, 0.8, 3.0)
ROI_SIZE = (128, 128, 32)


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
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=VOXEL_SPACING, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys="image", source_key="image", margin=5),
        ]
    )


def load_model(checkpoint: str | Path, device: torch.device) -> SegResNetDS:
    """Load a TAPS checkpoint, including checkpoints saved from compiled models."""
    model = build_model(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    clean_state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


def resolve_checkpoint(checkpoint: str | Path | None) -> Path:
    """Resolve a checkpoint path, falling back to the packaged model when none is provided."""
    if checkpoint is not None:
        return Path(checkpoint)

    bundled_checkpoint = Path(__file__).resolve().parent / "resources" / "model_v1.pth"
    if bundled_checkpoint.is_file():
        return bundled_checkpoint

    raise FileNotFoundError(
        "Bundled checkpoint not found; pass --checkpoint to provide a model file explicitly."
    )


def segment(
    image_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path | None,
    device: str | None = None,
) -> Path:
    """Segment a prostate MRI and write a binary NIfTI mask in native image space.

    :param image_path: Source NIfTI image (.nii or .nii.gz).
    :param output_path: Destination NIfTI segmentation mask.
    :param checkpoint: Path to a TAPS SegResNetDS checkpoint.
    :param device: Torch device override. Defaults to CUDA when available.
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

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    preprocess = build_preprocessing()
    data = cast(dict[Hashable, Any], preprocess({"image": str(image_path)}))
    model = load_model(checkpoint, torch_device)

    image = data["image"].unsqueeze(0).to(torch_device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=torch_device.type, enabled=torch_device.type == "cuda"
    ):
        logits = sliding_window_inference(
            image,
            roi_size=ROI_SIZE,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

    data["pred"] = (torch.sigmoid(logits[0]) > 0.5).to(torch.float32).cpu()
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, affine), output_path)
    return output_path