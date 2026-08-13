"""Tests for the TAPS command-line interface."""

from taps import cli


def test_segment_command_delegates_to_library(monkeypatch, capsys):
    """The CLI passes parsed segment arguments to the inference API."""
    calls = []

    def fake_segment(image, output, checkpoint, device):
        calls.append((image, output, checkpoint, device))
        return output

    monkeypatch.setattr(cli, "segment", fake_segment)

    exit_code = cli.main(
        [
            "segment",
            "scan.nii.gz",
            "mask.nii.gz",
            "--checkpoint",
            "model.pth",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert calls == [("scan.nii.gz", "mask.nii.gz", "model.pth", "cpu")]
    assert "Saved segmentation to mask.nii.gz" in capsys.readouterr().out


def test_segment_command_uses_bundled_checkpoint_by_default(monkeypatch, capsys):
    """The CLI should use the packaged checkpoint when no override is provided."""
    calls = []

    def fake_segment(image, output, checkpoint, device):
        calls.append((image, output, checkpoint, device))
        return output

    monkeypatch.setattr(cli, "segment", fake_segment)

    exit_code = cli.main(["segment", "scan.nii.gz", "mask.nii.gz", "--device", "cpu"])

    assert exit_code == 0
    assert calls == [("scan.nii.gz", "mask.nii.gz", None, "cpu")]
    assert "Saved segmentation to mask.nii.gz" in capsys.readouterr().out


def test_segment_command_allows_bundled_checkpoint_by_default():
    """The CLI should not require an explicit checkpoint path for the packaged model."""
    parser = cli.build_parser()

    args = parser.parse_args(["segment", "scan.nii.gz", "mask.nii.gz"])

    assert args.image == "scan.nii.gz"
    assert args.output == "mask.nii.gz"
    assert args.checkpoint is None