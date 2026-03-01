"""Extract text from PDFs using PyMuPDF, with Tesseract OCR fallback for scanned pages."""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def extract_page_text(page: fitz.Page) -> str:
    """Extract text from a single page, using OCR if the page is scanned.

    Args:
        page: A PyMuPDF page object.

    Returns:
        Extracted text string.
    """
    text = page.get_text()
    if text.strip():
        return text

    # Page has no extractable text, likely scanned. Render and OCR.
    logger.debug("Page %d has no text, falling back to Tesseract OCR", page.number)
    pix = page.get_pixmap(dpi=300)
    with tempfile.NamedTemporaryFile(suffix=".png") as img_file:
        pix.save(img_file.name)
        try:
            result = subprocess.run(  # noqa: S603
                ["tesseract", img_file.name, "stdout"],  # noqa: S607
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Tesseract is not installed. "
                "Install with: brew install tesseract (macOS) "
                "or apt-get install tesseract-ocr (Linux)"
            ) from None
        except subprocess.CalledProcessError as exc:
            stderr_text = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Tesseract OCR failed on page {page.number} "
                f"(exit code {exc.returncode}): {stderr_text}"
            ) from exc
    return result.stdout.decode("utf-8", errors="replace")


def render_pages_as_base64(
    pdf_path: str | Path,
    *,
    max_pages: int = 10,
    dpi: int = 150,
) -> list[str]:
    """Render PDF pages as base64-encoded PNG images.

    Renders up to max_pages from the start of the document. Uses a lower DPI
    than OCR since these are for LLM vision, not character recognition.

    Args:
        pdf_path: Path to the PDF file.
        max_pages: Maximum number of pages to render.
        dpi: Resolution for rendering.

    Returns:
        List of base64-encoded PNG strings.
    """
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        images = []
        for page in doc[:max_pages]:
            pix = page.get_pixmap(dpi=dpi)
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    logger.debug("Rendered %d/%d pages from %s as images", len(images), total_pages, pdf_path)
    return images


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract all text from a PDF, concatenating pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full document text with pages separated by newlines.
    """
    with fitz.open(pdf_path) as doc:
        pages = [extract_page_text(page) for page in doc]
    return "\n".join(pages)
