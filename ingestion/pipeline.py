"""
Ingestion: Main Pipeline Runner
Orchestrates the full ingestion pipeline end to end.

Stages (build-time only, runs once):
  Stage 1: Download auto-downloadable documents
  Stage 2: Process PDFs → page images + OCR text
  Stage 3: Extract specs via schema-constrained vision model
  Stage 4: Human validation gate (mandatory)
  Stage 5: Build physics_cache.json + ChromaDB vector store

Usage:
  conda activate aviate
  cd /path/to/failurescope
  python -m ingestion.pipeline --stage all
  python -m ingestion.pipeline --stage download
  python -m ingestion.pipeline --stage process --doc cessna_182q_poh_1977
  python -m ingestion.pipeline --stage extract --doc cessna_182q_poh_1977
  python -m ingestion.pipeline --stage validate --doc cessna_182q_poh_1977
  python -m ingestion.pipeline --stage build-cache
  python -m ingestion.pipeline --status
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.config import (
    DATA_ROOT, PAGE_IMAGES_DIR, VALIDATED_DIR,
    CACHE_DIR, ensure_dirs
)
from ingestion.downloader import DocumentDownloader
from ingestion.pdf_processor import PDFProcessor
from ingestion.spec_extractor import SpecExtractor
from ingestion.ocr_extractor import OCRTextExtractor
from ingestion.diagram_extractor import DiagramExtractor
from ingestion.validator_cli import ValidationCLI
from ingestion.cache_builder import CacheBuilder, VectorStoreBuilder
from ingestion.schemas import DOCUMENT_REGISTRY, POH_SCHEMA, PHYSICS_SCHEMA


def get_schema_for_doc(doc_entry: dict) -> dict:
    """Return the correct schema dict for a document entry."""
    doc_type = doc_entry.get("doc_type", "POH")
    if doc_type in ("POH", "MAINTENANCE_MANUAL", "OWNERS_MANUAL"):
        return POH_SCHEMA
    elif doc_type == "PHYSICS_REF":
        return PHYSICS_SCHEMA
    return POH_SCHEMA


# ─── STAGE 1: DOWNLOAD ───────────────────────────────────────

def stage_download(args):
    logger.info("=== STAGE 1: Download Auto Documents ===")
    downloader = DocumentDownloader(PROJECT_ROOT)
    results = downloader.download_all(
        force=getattr(args, "force", False),
        skip_large=getattr(args, "skip_large", False),
    )
    return results


# ─── STAGE 2: PROCESS PDF ────────────────────────────────────

def stage_process(args):
    doc_id = args.doc
    if not doc_id:
        logger.error("--doc required for process stage")
        sys.exit(1)

    if doc_id not in DOCUMENT_REGISTRY:
        logger.error(f"Unknown doc_id: {doc_id}. Check ingestion/schemas.py DOCUMENT_REGISTRY")
        sys.exit(1)

    logger.info(f"=== STAGE 2: Process PDF — {doc_id} ===")

    entry = DOCUMENT_REGISTRY[doc_id]
    pdf_path = DATA_ROOT / entry["expected_path"].replace("data/", "")

    if not pdf_path.exists():
        logger.error(
            f"PDF not found: {pdf_path}\n"
            f"Run: python -m ingestion.dataset_linker --link\n"
            f"to copy files from /opt/datasets/ to correct locations."
        )
        sys.exit(1)

    processor = PDFProcessor(
        output_dir=PAGE_IMAGES_DIR / doc_id,
        dpi=300,
    )

    page_range = None
    if hasattr(args, "pages") and args.pages:
        parts = args.pages.split("-")
        page_range = (int(parts[0]), int(parts[1]))

    page_records = processor.process(
        pdf_path=pdf_path,
        doc_id=doc_id,
        citation=entry["citation"],
        page_range=page_range,
    )

    output_path = processor.save_page_records(page_records, doc_id)
    logger.success(f"Page records saved: {output_path}")
    return page_records


# ─── STAGE 3: EXTRACT SPECS ──────────────────────────────────

def stage_extract(args):
    doc_id = args.doc
    if not doc_id:
        logger.error("--doc required for extract stage")
        sys.exit(1)

    logger.info(f"=== STAGE 3: Extract Specs — {doc_id} ===")

    # Load page records
    pages_json = PAGE_IMAGES_DIR / doc_id / f"{doc_id}_pages.json"
    if not pages_json.exists():
        logger.error(f"Page records not found: {pages_json}\nRun stage 'process' first.")
        sys.exit(1)

    with open(pages_json) as f:
        page_records = json.load(f)

    entry = DOCUMENT_REGISTRY[doc_id]
    citation = entry["citation"]

    # Use OCR text extractor (deterministic, no vision model)
    # Vision model (SpecExtractor) reserved for future fallback
    extractor = OCRTextExtractor()
    extracted = extractor.extract_all(page_records, doc_id, citation)

    # Save extracted (pending validation)
    output_path = extractor.save_extracted(extracted, doc_id)

    found = sum(1 for v in extracted.values() if v.get("raw_value") is not None)
    logger.success(f"Extraction complete: {found}/{len(extracted)} fields found → {output_path}")
    return extracted


# ─── STAGE 4: HUMAN VALIDATION ───────────────────────────────

def stage_validate(args):
    doc_id = args.doc
    if not doc_id:
        logger.error("--doc required for validate stage")
        sys.exit(1)

    logger.info(f"=== STAGE 4: Human Validation — {doc_id} ===")

    extracted_path = VALIDATED_DIR / f"{doc_id}_extracted.json"
    if not extracted_path.exists():
        logger.error(f"Extracted specs not found: {extracted_path}\nRun stage 'extract' first.")
        sys.exit(1)

    with open(extracted_path) as f:
        extracted = json.load(f)

    validator_name = getattr(args, "validator", "developer")
    cli = ValidationCLI(validator_name=validator_name)

    output_path = VALIDATED_DIR / f"{doc_id}_validated.json"
    validated = cli.validate_extracted_specs(extracted, output_path=output_path)

    return validated


# ─── STAGE 5: BUILD CACHE ────────────────────────────────────

def stage_build_cache(args):
    logger.info("=== STAGE 5: Build physics_cache.json + ChromaDB ===")

    # Build physics cache from all validated specs
    builder = CacheBuilder()
    cache_path = builder.build()

    if cache_path:
        logger.success(f"physics_cache.json: {cache_path}")

    # Build ChromaDB from OCR text
    logger.info("Building ChromaDB vector store from page text...")
    vs_builder = VectorStoreBuilder()

    pages_root = PAGE_IMAGES_DIR
    total_chunks = 0

    for doc_pages_dir in pages_root.iterdir():
        if not doc_pages_dir.is_dir():
            continue
        pages_json = doc_pages_dir / f"{doc_pages_dir.name}_pages.json"
        if not pages_json.exists():
            continue

        with open(pages_json) as f:
            records = json.load(f)

        chunks = vs_builder.ingest_pages(records)
        total_chunks += chunks

    vs_builder.stats()
    logger.success(f"ChromaDB built: {total_chunks} total chunks")


# ─── STATUS ──────────────────────────────────────────────────

def show_status(args):
    downloader = DocumentDownloader(PROJECT_ROOT)
    downloader.status()

    cache_path = PROJECT_ROOT / "data" / "cache" / "physics_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"physics_cache.json: {len(cache.get('values', {}))} values — BUILT ✓")
    else:
        print("physics_cache.json: NOT BUILT")

    chroma_dir = PROJECT_ROOT / "data" / "chromadb"
    if (chroma_dir / "chroma.sqlite3").exists():
        print("ChromaDB: BUILT ✓")
    else:
        print("ChromaDB: NOT BUILT")
    print()


# ─── CLI ENTRYPOINT ──────────────────────────────────────────

def stage_diagram_extract(args):
    """
    Stage 3b: Vision-based extraction for image-heavy documents.
    Runs AFTER stage 'process'. Complements OCR extraction.
    Only processes pages with <50 chars OCR text.
    """
    doc_id = args.doc
    if not doc_id:
        logger.error("--doc required for diagram-extract stage")
        sys.exit(1)

    logger.info(f"=== STAGE 3b: Diagram Extract — {doc_id} ===")

    pages_json = PAGE_IMAGES_DIR / doc_id / f"{doc_id}_pages.json"
    if not pages_json.exists():
        logger.error(f"Page records not found: {pages_json}\nRun stage 'process' first.")
        sys.exit(1)

    with open(pages_json) as f:
        page_records = json.load(f)

    # Count image-dominant pages
    image_pages = sum(1 for p in page_records if len(p.get('text','') or '') < 50)
    logger.info(f"Document has {len(page_records)} pages, {image_pages} image-dominant")

    if image_pages == 0:
        logger.info("No image-dominant pages found. Diagram extraction not needed.")
        return {}

    if doc_id not in DOCUMENT_REGISTRY:
        logger.error(f"Unknown doc_id: {doc_id}. Check ingestion/schemas.py DOCUMENT_REGISTRY")
        sys.exit(1)

    entry = DOCUMENT_REGISTRY[doc_id]
    citation = entry["citation"]

    extractor = DiagramExtractor()
    results = extractor.extract_from_document(doc_id, page_records, citation)

    output_path = extractor.save_extracted(results, doc_id)
    found = len(results)
    logger.success(f"Diagram extraction complete: {found} fields found → {output_path}")

    if found == 0:
        logger.warning("No fields extracted from diagrams. Check llava:7b is running.")
    else:
        logger.info("Run --stage validate-diagram --doc {doc_id} to validate these values.")

    return results


def stage_validate_diagram(args):
    """
    Stage 4b: Human validation for diagram-extracted values.
    Same gate as stage 'validate' but reads from diagram extraction file.
    """
    doc_id = args.doc
    if not doc_id:
        logger.error("--doc required for validate-diagram stage")
        sys.exit(1)

    logger.info(f"=== STAGE 4b: Validate Diagram Extractions — {doc_id} ===")

    extracted_path = VALIDATED_DIR / f"{doc_id}_diagram_extracted.json"
    if not extracted_path.exists():
        logger.error(f"Diagram extracted specs not found: {extracted_path}\nRun stage 'diagram-extract' first.")
        sys.exit(1)

    with open(extracted_path) as f:
        extracted = json.load(f)

    validator_name = getattr(args, "validator", "developer")
    cli = ValidationCLI(validator_name=validator_name)

    output_path = VALIDATED_DIR / f"{doc_id}_diagram_validated.json"
    validated = cli.validate_extracted_specs(extracted, doc_id, output_path)

    accepted = sum(1 for v in validated.values() if v.get("status") in ("accepted", "corrected"))
    logger.success(
        f"Diagram validation complete: {accepted}/{len(validated)} accepted → {output_path}"
    )
    logger.info("Run --stage build-cache to include these values in physics_cache.json")
    return validated


def main():
    parser = argparse.ArgumentParser(
        description="FailureScope Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["download", "process", "extract", "validate", "build-cache", "diagram-extract", "validate-diagram", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--doc", help="Document ID from DOCUMENT_REGISTRY")
    parser.add_argument("--pages", help="Page range e.g. '1-50'")
    parser.add_argument("--validator", default="developer", help="Validator name for audit trail")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-large", action="store_true", help="Skip files >100MB")
    parser.add_argument("--status", action="store_true", help="Show document status")

    args = parser.parse_args()

    if args.status:
        show_status(args)
        return

    if not args.stage:
        parser.print_help()
        return

    if args.stage == "download" or args.stage == "all":
        stage_download(args)

    if args.stage == "process":
        stage_process(args)

    if args.stage == "extract":
        stage_extract(args)

    if args.stage == "validate":
        stage_validate(args)

    if args.stage == "build-cache":
        stage_build_cache(args)

    if args.stage == "all":
        logger.info("Full pipeline requires --doc for process/extract/validate stages.")
        logger.info("Run stages individually after placing POH documents.")


if __name__ == "__main__":
    main()