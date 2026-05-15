"""
Ingestion: PDF Processor
Handles both text-layer PDFs and scanned (image-only) PDFs.

Strategy per skill guidance:
  - Text PDFs: pdfplumber for text extraction
  - Scanned PDFs (like POHs): pdftoppm → images → pytesseract OCR
  - Tables: pdfplumber first, rasterize if garbled
  - Always rasterize target pages for vision model input

Outputs per page:
  - Raw OCR text (for ChromaDB)
  - Page image at 300 DPI (for vision model spec extraction)
  - Page metadata (page number, doc_id, citation)
"""

import json
import base64
from pathlib import Path
from typing import Optional
from loguru import logger

import fitz  # PyMuPDF
import pdfplumber

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed — OCR fallback unavailable")


from ingestion.config import DATA_ROOT, PAGE_IMAGES_DIR
PROJECT_ROOT = Path(__file__).parent.parent


class PDFProcessor:
    """
    Processes PDF documents into:
      1. Per-page text (OCR for scanned, direct for text PDFs)
      2. Per-page images at 300 DPI (for vision model extraction)
      3. Page metadata with citation
    """

    def __init__(
        self,
        output_dir: Path = PAGE_IMAGES_DIR,
        dpi: int = 300,
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── PUBLIC ENTRY POINT ──────────────────────────────────────

    def process(
        self,
        pdf_path: Path,
        doc_id: str,
        citation: dict,
        page_range: Optional[tuple] = None,
    ) -> list[dict]:
        """
        Process a PDF into page records.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier from DOCUMENT_REGISTRY
            citation: Citation metadata dict
            page_range: Optional (start, end) 1-indexed page numbers to process.
                        None = process all pages.

        Returns:
            List of page records:
            {
                doc_id, page_number, text, image_path,
                image_base64, citation, is_scanned
            }
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path.name} (doc_id={doc_id})")

        # Detect if PDF has extractable text or is scanned
        is_scanned = self._detect_scanned(pdf_path)
        logger.info(f"  Type: {'SCANNED (image-only)' if is_scanned else 'TEXT (extractable)'}")

        # Open with PyMuPDF for page rendering
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        logger.info(f"  Pages: {total_pages}")

        # Determine page range
        if page_range:
            start_page = max(0, page_range[0] - 1)   # Convert to 0-indexed
            end_page = min(total_pages - 1, page_range[1] - 1)
        else:
            start_page = 0
            end_page = total_pages - 1

        pages_to_process = list(range(start_page, end_page + 1))
        logger.info(f"  Processing pages: {start_page+1} to {end_page+1}")

        page_records = []

        for page_idx in pages_to_process:
            page_num = page_idx + 1  # 1-indexed for display
            logger.debug(f"  Page {page_num}/{total_pages}")

            # Render page to image
            image_path, image_base64 = self._render_page(
                doc, page_idx, doc_id, page_num
            )

            # Extract text
            if is_scanned:
                text = self._ocr_page(image_path)
            else:
                text = self._extract_text_page(pdf_path, page_idx)

            record = {
                "doc_id": doc_id,
                "page_number": page_num,
                "total_pages": total_pages,
                "text": text,
                "image_path": str(image_path),
                "image_base64": image_base64,
                "citation": citation,
                "is_scanned": is_scanned,
                "dpi": self.dpi,
            }
            page_records.append(record)

        doc.close()
        logger.success(f"Processed {len(page_records)} pages from {pdf_path.name}")
        return page_records

    # ─── SCAN DETECTION ──────────────────────────────────────────

    def _detect_scanned(self, pdf_path: Path) -> bool:
        """
        Detect if PDF is scanned (image-only) or has extractable text.
        Strategy: Try extracting text from first 3 pages.
        If total chars < 100, treat as scanned.
        """
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                sample_pages = min(3, len(pdf.pages))
                total_chars = 0
                for i in range(sample_pages):
                    text = pdf.pages[i].extract_text() or ""
                    total_chars += len(text.strip())
                return total_chars < 100
        except Exception:
            return True  # Assume scanned if extraction fails

    # ─── PAGE RENDERING ──────────────────────────────────────────

    def _render_page(
        self,
        doc: fitz.Document,
        page_idx: int,
        doc_id: str,
        page_num: int,
    ) -> tuple[Path, str]:
        """
        Render PDF page to image at self.dpi.
        Returns (image_path, base64_encoded_png).
        """
        page = doc[page_idx]
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # Save image
        img_filename = f"{doc_id}_page_{page_num:04d}.png"
        img_path = self.output_dir / img_filename
        pix.save(str(img_path))

        # Base64 encode for vision model
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return img_path, b64

    # ─── TEXT EXTRACTION ─────────────────────────────────────────

    def _extract_text_page(self, pdf_path: Path, page_idx: int) -> str:
        """Extract text from a text-layer PDF page using pdfplumber."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page = pdf.pages[page_idx]
                text = page.extract_text() or ""
                return text.strip()
        except Exception as e:
            logger.warning(f"Text extraction failed page {page_idx+1}: {e}")
            return ""

    # ─── OCR ─────────────────────────────────────────────────────

    def _ocr_page(self, image_path: Path) -> str:
        """
        OCR a page image using pytesseract.
        Returns extracted text or empty string if OCR unavailable.

        Note: OCR text goes to ChromaDB for semantic search.
        It is NOT used directly for numeric spec values.
        Numeric values are extracted by the vision model with schema constraint.
        """
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract unavailable — returning empty OCR text")
            return ""

        try:
            img = Image.open(str(image_path))
            # PSM 6: Assume uniform block of text (best for POH pages)
            config = "--psm 6 --oem 3"
            text = pytesseract.image_to_string(img, config=config)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""

    # ─── TABLE EXTRACTION ────────────────────────────────────────

    def extract_tables_from_page(self, pdf_path: Path, page_num: int) -> list:
        """
        Extract tables from a specific page using pdfplumber.
        page_num is 1-indexed.
        Returns list of tables (each table is list of rows).
        """
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                return tables or []
        except Exception as e:
            logger.warning(f"Table extraction failed page {page_num}: {e}")
            return []

    # ─── BATCH PROCESSING ────────────────────────────────────────

    def save_page_records(self, records: list[dict], doc_id: str):
        """Save page records to JSON for downstream processing."""
        output_path = self.output_dir / f"{doc_id}_pages.json"
        # Don't save base64 to disk (too large) — save path reference only
        slim_records = []
        for r in records:
            slim = {k: v for k, v in r.items() if k != "image_base64"}
            slim_records.append(slim)

        with open(output_path, "w") as f:
            json.dump(slim_records, f, indent=2)
        logger.info(f"Saved page records: {output_path}")
        return output_path