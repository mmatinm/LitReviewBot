import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from api_client import generate_image_caption, get_openrouter_client
from config import IMAGE_MODELS


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


def _image_context(markdown: str, image_path: Path) -> str:
    """Find complete paper paragraphs that reference this image."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", markdown) if block.strip()]
    image_name = image_path.stem
    names = {image_path.name.lower(), image_name.lower()}
    image_index = next(
        (index for index, block in enumerate(blocks)
         if any(name in block.lower() for name in names)),
        None,
    )
    if image_index is None:
        return ""

    image_block = blocks[image_index]
    anchor_match = re.search(
        r'(?:id|href|src)\s*=\s*["\']?(page-\d+-\d+)|#(page-\d+-\d+)',
        image_block,
        re.IGNORECASE,
    )
    anchor = next((value for value in anchor_match.groups() if value), None) if anchor_match else None

    references = []
    if anchor:
        references.append(rf'(?<![\w-])#?{re.escape(anchor)}(?![\w-])')

    # Marker object numbers are page-local, so `_page_2_Figure_1` may be
    # "Fig. 3" in the source paper. The caption next to the image contains
    # the paper's authoritative global number.
    caption_blocks = []
    for index in range(max(0, image_index - 2), min(len(blocks), image_index + 4)):
        if index == image_index:
            continue
        if re.search(r'\b(?:fig(?:ure)?|table|diagram|picture)\b', blocks[index], re.IGNORECASE):
            caption_blocks.append(blocks[index])

    def label_pattern(label: str, number: str) -> str:
        if label.lower().startswith("fig"):
            labels = "figure|fig"
        else:
            labels = re.escape(label.lower())
        return rf'\b(?:{labels})\.?\s*(?:\[\s*|\(\s*)?{number}(?:\s*[\]\)])?\b'

    label_patterns = set()
    for caption in caption_blocks:
        for label, number in re.findall(
            r'\b(fig(?:ure)?|table|diagram|picture)\.?\s*(?:[\[\(]\s*)?(\d+)',
            caption,
            re.IGNORECASE,
        ):
            label_patterns.add(label_pattern(label, number))

    # Retain filename labels as a fallback for papers without a usable caption.
    label_match = re.search(
        r'(?:^|_)((?:Figure|Picture|Diagram|Table))[_ -]?(\d+)$',
        image_name,
        re.IGNORECASE,
    )
    if label_match:
        label, number = label_match.groups()
        label_patterns.add(label_pattern(label, number))

    # Also use the page/object pair when a paper preserves those tokens in
    # prose or links, e.g. "_page_2_Picture_0" / "page-2-0".
    page_match = re.search(r'_page_(\d+)_(?:Figure|Picture|Diagram|Table)_(\d+)$', image_name, re.IGNORECASE)
    if page_match:
        page, object_number = page_match.groups()
        references.extend([
            rf'\bpage[-_ ]{page}[-_ ]{object_number}\b',
            rf'#page-{page}-{object_number}\b',
        ])

    references.extend(label_patterns)
    selected = {
        index for index, block in enumerate(blocks)
        if any(re.search(pattern, block, re.IGNORECASE) for pattern in references)
    }
    selected.add(image_index)

    # Keep the complete Markdown blocks, preserving their original paper order.
    return "\n\n".join(blocks[index] for index in sorted(selected))


def _insert_captions_in_place(markdown: str, captions_by_name: dict) -> str:
    """Replace Marker image links with their captions at the same document position."""
    for image_name, caption in captions_by_name.items():
        if not caption or caption.startswith("[Image transcription failed:"):
            continue
        image_pattern = re.compile(
            rf'!\[[^\]]*\]\([^)]*{re.escape(image_name)}[^)]*\)',
            re.IGNORECASE,
        )
        replacement = f"**Visual content ({Path(image_name).stem}):**\n\n{caption}"
        markdown = image_pattern.sub(replacement, markdown)
    return markdown


def extract_pdf_data_with_marker(
    upload_files,
    progress_callback=None,
    mode: str = "fast",
    api_key: str | None = None,
    image_model: str | None = IMAGE_MODELS[0],
):
    """Parse PDFs with Marker and return the same document mapping as other backends."""
    documents_data = {}
    total_files = len(upload_files)
    image_root = Path("extracted_visuals")
    text_root = Path("extracted_texts")
    image_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

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
            source_markdown = markdown

            image_paths = [
                path
                for path in output_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            captions_by_name = {}
            if api_key and image_paths:
                caption_client = get_openrouter_client(api_key)
                for image_path in image_paths:
                    context = _image_context(source_markdown, image_path)
                    caption = generate_image_caption(
                        caption_client,
                        image_model,
                        image_path.read_bytes(),
                        context,
                        mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
                    ).strip()
                    captions_by_name[image_path.name] = caption
                markdown = _insert_captions_in_place(markdown, captions_by_name)

            full_paper_text = f"--- START OF PAPER: {filename} ---\n\n{markdown}\n"
            documents_data[filename] = full_paper_text
            (text_root / f"{source_stem}_processed.txt").write_text(
                full_paper_text,
                encoding="utf-8",
            )

            paper_image_dir = image_root / source_stem
            if paper_image_dir.exists():
                shutil.rmtree(paper_image_dir)
            paper_image_dir.mkdir(parents=True, exist_ok=True)
            for image_path in output_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                }:
                    destination = paper_image_dir / image_path.name
                    shutil.copy2(image_path, destination)
                    context = _image_context(source_markdown, image_path)
                    caption = captions_by_name.get(image_path.name, "")
                    sidecar = destination.with_suffix(".txt")
                    sidecar.write_text(
                        f"Image: {image_path.name}\n\n"
                        f"Relevant extracted text:\n{context}\n\n"
                        f"AI-generated caption:\n{caption or '[No caption generated]'}\n",
                        encoding="utf-8",
                    )

    return documents_data
