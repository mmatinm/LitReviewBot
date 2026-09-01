import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "paper"


def _marker_command() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    executable = scripts_dir / ("marker_single.exe" if os.name == "nt" else "marker_single")
    if executable.exists():
        return str(executable)
    return shutil.which("marker_single") or "marker_single"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def _find_markdown(output_dir: Path, source_stem: str) -> Path:
    candidates = list(output_dir.rglob("*.md")) + list(output_dir.rglob("*.markdown"))
    if not candidates:
        raise RuntimeError("Marker finished but did not produce a Markdown file.")

    source_stem_low = source_stem.lower()
    matching = [p for p in candidates if source_stem_low in p.stem.lower()]
    candidates = matching or candidates
    return max(candidates, key=lambda p: p.stat().st_size)


def extract_pdf_data_with_marker(upload_files, progress_callback=None, mode: str = "fast"):
    """Parse PDFs with Marker and return the same document mapping as other backends."""
    documents_data = {}
    total_files = len(upload_files)
    os.makedirs("extracted_visuals", exist_ok=True)

    for file_idx, uploaded_file in enumerate(upload_files):
        filename = uploaded_file.name
        source_stem = _safe_stem(filename)
        if progress_callback:
            progress_callback(f"Marker parsing '{filename}' ({file_idx + 1}/{total_files})")

        pdf_bytes = uploaded_file.read()
        with tempfile.TemporaryDirectory(prefix="marker_run_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / f"{source_stem}.pdf"
            output_dir = tmp_path / "marker_output"
            input_path.write_bytes(pdf_bytes)

            cmd = [
                _marker_command(),
                str(input_path),
                "--output_dir", str(output_dir),
                "--output_format", "markdown",
                "--mode", mode,
                "--disable_ocr",
                "--disable_multiprocessing",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "TORCH_DEVICE": "cpu",
                    "FAST_DETECTOR_DEVICE": "cpu",
                },
            )
            if proc.returncode != 0:
                error_text = (proc.stderr or proc.stdout or "").strip()[-1600:]
                raise RuntimeError(
                    "Marker CLI failed. Ensure marker-pdf is installed in the active Python environment. "
                    f"Command: {' '.join(cmd)}\nError: {error_text}"
                )

            markdown_path = _find_markdown(output_dir, source_stem)
            markdown = _read_text(markdown_path).strip()
            if not markdown:
                raise RuntimeError("Marker produced an empty Markdown file.")

            full_paper_text = f"--- START OF PAPER: {filename} ---\n\n{markdown}\n"
            documents_data[filename] = full_paper_text

            debug_output_dir = Path("extracted_visuals") / f"marker_{source_stem}"
            if debug_output_dir.exists():
                shutil.rmtree(debug_output_dir)
            shutil.copytree(output_dir, debug_output_dir)

    return documents_data
