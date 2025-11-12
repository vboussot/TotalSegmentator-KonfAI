import argparse
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk  # noqa: N813
from konfai.utils.utils import get_available_models_on_hf_repo

SUPPORTED_EXTENSIONS = [
    "mha",
    "mhd",  # MetaImage
    "nii",
    "nii.gz",  # NIfTI
    "nrrd",
    "nrrd.gz",  # NRRD
    "gipl",
    "gipl.gz",  # GIPL
]

TOTAL_SEGMENTATOR_KONFAI_REPO = "VBoussot/TotalSegmentator-KonfAI"


def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator-KonfAI", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i",
        "--input",
        metavar="filepath",
        dest="input",
        help="Input image path.",
        type=lambda p: Path(p).absolute(),
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="filepath",
        dest="output",
        help="Output segmentation path.",
        type=lambda p: Path(p).absolute(),
        required=False,
        default=Path("Seg.nii.gz").absolute(),
    )

    parser.add_argument(
        "-ta",
        "--task",
        choices=get_available_models_on_hf_repo(TOTAL_SEGMENTATOR_KONFAI_REPO),
        help="Select which model to use. This determines what is predicted.",
        default="total",
    )

    parser.add_argument(
        "-f", "--fast", action="store_true", help="Run faster lower resolution model (3mm)", default=False
    )

    parser.add_argument("-quiet", action="store_true", help="Suppress console output.")

    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        default=(os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""),
        help="GPU list (e.g. '0' or '0,1'). Leave empty for CPU.",
    )

    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU cores to use when --gpu is empty.")

    parser.add_argument("--version", action="version", version=importlib.metadata.version("TotalSegmentator-KonfAI"))

    args = parser.parse_args()

    # --- Input checks ---
    if not args.input.exists():
        print(f"❌ Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not any(str(args.input).endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        print(f"❌ Unsupported input extension: {args.input.name}", file=sys.stderr)
        print(f"   Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    # --- Output checks ---
    out_parent = args.output.parent
    try:
        out_parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Cannot create output directory {out_parent}: {e}", file=sys.stderr)
        sys.exit(1)

    if not any(str(args.output).endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        print(f"❌ Unsupported output extension: {args.output.name}", file=sys.stderr)
        print(f"   Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dataset_p = tmpdir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)

        # Convert input to expected NIfTI
        vol_out = dataset_p / "Volume.nii.gz"
        try:
            img = sitk.ReadImage(str(args.input))
            sitk.WriteImage(img, str(vol_out))
        except Exception as e:
            print(
                f"❌ Error reading/writing image with SimpleITK:\n   in : {args.input}\n",
                f"out: {vol_out}\n   detail: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        cmd = [
            "konfai",
            "PREDICTION_HF",
            "-y",
            "--config",
            f"{TOTAL_SEGMENTATOR_KONFAI_REPO}:{args.task}",
        ]
        if args.gpu:
            cmd += ["--gpu", args.gpu]
        else:
            cmd += ["--cpu", str(args.cpu)]
        if args.quiet:
            cmd += ["-quiet"]
        try:
            subprocess.run(cmd, cwd=tmpdir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 'konfai PREDICTION' failed with exit code {e.returncode}.", file=sys.stderr)
            sys.exit(e.returncode)
        except FileNotFoundError:
            print("❌ 'konfai' executable not found. Ensure it is installed and on PATH.", file=sys.stderr)
            sys.exit(1)

        pred = tmpdir / "Predictions" / "TotalSegmentator" / "Dataset" / "P001" / "Seg.mha"
        if not pred.exists():
            print(f"❌ Prediction not found at: {pred}\n   Check KonfAI logs for details.", file=sys.stderr)
            sys.exit(1)

        try:
            seg = sitk.ReadImage(str(pred))
            sitk.WriteImage(seg, str(args.output))
        except Exception as e:
            print(
                f"❌ Error saving output segmentation:\n   from: {pred}\n   to  : {args.output}\n   detail: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.quiet:
        print(f"✅ Done. Segmentation saved to: {args.output}")
