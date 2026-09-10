
from pathlib import Path
import subprocess
import sys
import os
import shutil

docker_dir = r"C:\Users\Matin\AppData\Local\Programs\DockerDesktop\resources\bin"

os.environ["PATH"] = docker_dir + os.pathsep + os.environ["PATH"]

print("Docker found at:", shutil.which("docker"))


# Paths
PDF_PATH = Path(
    r"D:\Edu\Project\bot literature\xiu 2026.pdf"
)

OUTPUT_DIR = Path(
    r"D:\Lit-review-bot\LitReviewBot\extracted_visuals"
)

# Output directory setup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run Marker
# Use the Python executable from the current virtual environment
marker_exe = Path(sys.executable).parent / "marker_single.exe"

if not marker_exe.exists():
    marker_exe = Path(sys.executable).parent / "marker_single"

if not marker_exe.exists():
    raise FileNotFoundError(
        f"Could not find Marker executable in:\n"
        f"{Path(sys.executable).parent}\n\n"
        f"Make sure marker-pdf is installed in this venv."
    )

command = [
    str(marker_exe), 
    str(PDF_PATH), 
    "--output_dir", 
    str(OUTPUT_DIR), 
    "--output_format", 
    "markdown", 
    # CPU-only layout and table reconstruction
    "--mode",
    "fast",
    "--disable_ocr",
    "--disable_multiprocessing",
]

print("=" * 70)
print("Running Marker")
print("=" * 70)
print(f"PDF:    {PDF_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()
print("Command:")
print(" ".join(f'"{x}"' if " " in x else x for x in command))
print()

# Execute
result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env={
        **os.environ,
        "TORCH_DEVICE": "cpu",
        "FAST_DETECTOR_DEVICE": "cpu",
    },
)

# Print Marker output
print(result.stdout)

# Check result
if result.returncode != 0:
    raise RuntimeError(
        f"\nMarker failed with exit code {result.returncode}."
    )

print("\n" + "=" * 70)
print("Marker extraction completed successfully.")
print("=" * 70)
print(f"Results saved to:\n{OUTPUT_DIR}")
