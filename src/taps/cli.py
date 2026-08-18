"""Command-line interface for TAPS."""

from __future__ import annotations

import argparse
import logging

from taps.__about__ import __version__
from taps.inference import (
    DEFAULT_PROBABILITY_THRESHOLD,
    DEFAULT_SIGMA,
    BlankMaskError,
    segment,
)

logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    """Build the TAPS command-line parser."""
    parser = argparse.ArgumentParser(
        prog="taps",
        description="Segment prostate MRI scans using the TAPS model.",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment_parser = subparsers.add_parser(
        "segment", help="Create a prostate segmentation mask."
    )
    segment_parser.add_argument(
        "image", help="Input MRI NIfTI image (.nii or .nii.gz)."
    )
    segment_parser.add_argument(
        "output", help="Output segmentation NIfTI path (.nii or .nii.gz)."
    )
    segment_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a TAPS model checkpoint. Defaults to the bundled model.",
    )
    segment_parser.add_argument(
        "--device",
        help="Torch device to use, such as cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    )
    segment_parser.add_argument(
        "--exact",
        action="store_true",
        help="Use the requested output path even when QC finds an abnormality.",
    )
    segment_parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Gaussian blur sigma applied to logits before probability thresholding.",
    )
    segment_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="Probability threshold used to convert blurred logits to a binary mask.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the TAPS command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "segment":
        try:
            output_path = segment(
                args.image,
                args.output,
                args.checkpoint,
                args.device,
                args.exact,
                sigma=args.sigma,
                threshold=args.threshold,
            )
        except BlankMaskError as error:
            logger.error("Error: %s", error)
            return 1
        logger.info("Saved segmentation to %s", output_path)
        return 0
    return 1
