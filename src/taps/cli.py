"""Command-line interface for TAPS."""

from __future__ import annotations

import argparse

from taps.__about__ import __version__
from taps.inference import segment


def build_parser() -> argparse.ArgumentParser:
    """Build the TAPS command-line parser."""
    parser = argparse.ArgumentParser(
        prog="taps",
        description="Segment prostate MRI scans using the TAPS model.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment_parser = subparsers.add_parser("segment", help="Create a prostate segmentation mask.")
    segment_parser.add_argument("image", help="Input MRI NIfTI image (.nii or .nii.gz).")
    segment_parser.add_argument("output", help="Output segmentation NIfTI path (.nii or .nii.gz).")
    segment_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a TAPS model checkpoint. Defaults to the bundled model.",
    )
    segment_parser.add_argument(
        "--device",
        help="Torch device to use, such as cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the TAPS command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "segment":
        output_path = segment(args.image, args.output, args.checkpoint, args.device)
        print(f"Saved segmentation to {output_path}")
        return 0
    return 1
