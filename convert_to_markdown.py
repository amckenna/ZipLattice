"""Standalone document-to-Markdown converter for use with ZipLattice.

Converts PDF, Word (.docx), and HTML files into Markdown suitable for
ingestion by the ZipLattice knowledge graph.

Usage (CLI):
    python convert_to_markdown.py document.pdf                    # stdout
    python convert_to_markdown.py document.pdf -o document.md     # to file
    python convert_to_markdown.py f1.pdf f2.docx -d output_dir/   # batch

Usage (library):
    from convert_to_markdown import convert
    md = convert("document.pdf")
    md = convert("document.pdf", output="document.md")

Dependencies:
    pip install pymupdf4llm mammoth markdownify
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Format-specific converters
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}


def _markdownify(html: str, **kwargs) -> str:
    """Convert HTML to Markdown, skipping images with base64 data URIs."""
    try:
        from markdownify import MarkdownConverter
    except ImportError:
        raise SystemExit(
            "Markdown conversion requires markdownify: pip install markdownify"
        )

    class _Converter(MarkdownConverter):
        def convert_img(self, el, text, convert_as_inline):
            src = el.get("src", "")
            if src.startswith("data:"):
                return ""
            return super().convert_img(el, text, convert_as_inline)

    kwargs.setdefault("heading_style", "ATX")
    return _Converter(**kwargs).convert(html)


def convert_pdf(path: str | Path) -> str:
    """Convert a PDF file to Markdown using pymupdf4llm."""
    try:
        import pymupdf4llm
    except ImportError:
        raise SystemExit(
            "PDF conversion requires pymupdf4llm: pip install pymupdf4llm"
        )
    return pymupdf4llm.to_markdown(str(path))


def convert_docx(path: str | Path) -> str:
    """Convert a Word (.docx) file to Markdown via mammoth + markdownify."""
    try:
        import mammoth
    except ImportError:
        raise SystemExit(
            "DOCX conversion requires mammoth: pip install mammoth"
        )

    with open(path, "rb") as f:
        result = mammoth.convert_to_html(f)
    return _markdownify(result.value)


def convert_html(path: str | Path) -> str:
    """Convert a local HTML file to Markdown using markdownify."""
    html = Path(path).read_text(encoding="utf-8")

    # Strip non-content tags before conversion
    for tag in ("script", "style", "nav", "footer"):
        html = re.sub(
            rf"<{tag}[\s>].*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

    return _markdownify(html)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_CONVERTERS = {
    ".pdf": convert_pdf,
    ".docx": convert_docx,
    ".html": convert_html,
    ".htm": convert_html,
}


def convert(source: str | Path, output: str | Path | None = None) -> str:
    """Convert a document to Markdown.

    Parameters
    ----------
    source : str or Path
        Path to a PDF, DOCX, or HTML file.
    output : str or Path, optional
        If given, write the Markdown to this file path.

    Returns
    -------
    str
        The converted Markdown text.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    ext = source.suffix.lower()
    converter = _CONVERTERS.get(ext)
    if converter is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    md = converter(source)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")

    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF, DOCX, and HTML files to Markdown."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more input files to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (single-file mode).",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        help="Output directory (batch mode). Filenames are derived from inputs.",
    )
    args = parser.parse_args()

    if args.output and args.output_dir:
        parser.error("Use -o or -d, not both.")
    if args.output and len(args.files) > 1:
        parser.error("-o can only be used with a single input file.")

    for filepath in args.files:
        src = Path(filepath)
        if args.output:
            out = Path(args.output)
        elif args.output_dir:
            out = Path(args.output_dir) / (src.stem + ".md")
        else:
            out = None

        md = convert(src, output=out)

        if out is None:
            sys.stdout.write(md)
        else:
            print(f"Converted {src.name} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
