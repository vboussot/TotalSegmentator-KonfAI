import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk  # noqa: N813
from huggingface_hub import hf_hub_download

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


def ensure_konfai_available() -> None:
    from shutil import which

    if which("konfai") is None:
        print("❌ 'konfai' CLI not found in PATH. Install/activate KonfAI.", file=sys.stderr)
        sys.exit(1)


def _get_available_models() -> list[str]:
    return ["total", "total_mr"]


def get_models_name(task: str, fast: bool) -> tuple[list[str], str]:
    models_name = []
    inference_file = ""
    if task == "total":
        models_name = ["M291.pt", "M292.pt", "M293.pt", "M294.pt", "M295.pt"] if not fast else ["M297.pt"]
        inference_file = "Prediction_CT.yml" if not fast else "Prediction_CT_Fast.yml"

    if task == "total_mr":
        models_name = ["M850.pt", "M851.pt"] if not fast else ["M852.pt"]
        inference_file = "Prediction_MR.yml" if not fast else "Prediction_MR_Fast.yml"

    return models_name, inference_file


def download_models(models_name: list[str], inference_file: str) -> tuple[list[str], str, str]:
    try:
        models_path = []
        for model_name in models_name:
            models_path.append(
                hf_hub_download(
                    repo_id="VBoussot/TotalSegmentator-KonfAI", filename=model_name, repo_type="model", revision="main"
                )
            )
        model_path = hf_hub_download(
            repo_id="VBoussot/TotalSegmentator-KonfAI", filename="Model.py", repo_type="model", revision="main"
        )

        inference_file_path = hf_hub_download(
            repo_id="VBoussot/TotalSegmentator-KonfAI", filename=inference_file, repo_type="model", revision="main"
        )
        return models_path, inference_file_path, model_path
    except Exception as e:
        print(f"❌ Error downloading models/config from Hugging Face: {e}", file=sys.stderr)
        sys.exit(1)


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
        choices=_get_available_models(),
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

    ensure_konfai_available()
    models_name, inference_file = get_models_name(args.task, args.fast)
    models_path, inference_file_path, model_path = download_models(models_name, inference_file)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dataset_p = tmpdir / "Dataset" / "P001"
        dataset_p.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(model_path, tmpdir / "Model.py")
        except Exception as e:
            print(f"❌ Cannot copy Model.py into temp dir: {e}", file=sys.stderr)
            sys.exit(1)

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
            "PREDICTION",
            "-y",
            "--MODEL",
            ":".join(models_path),
            "--config",
            inference_file_path,
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
