"""
Ingestion: Dataset Linker
Scans /opt/datasets/, matches files to DOCUMENT_REGISTRY entries,
and copies them to the correct locations under /opt/failurescope/data/.

Usage:
  python -m ingestion.dataset_linker --scan     # Show what was found
  python -m ingestion.dataset_linker --link     # Copy matched files
  python -m ingestion.dataset_linker --status   # Show link status

This runs ONCE at setup. Does not move original files — copies only.
"""

import argparse
import shutil
import sys
from pathlib import Path
from loguru import logger

from ingestion.config import DATASETS_ROOT, MANUAL_DOC_HINTS, ensure_dirs


def scan_datasets() -> dict[str, list[Path]]:
    """
    Scan /opt/datasets/ recursively for all PDF files.
    Returns dict: {filename_lower: [full_path, ...]}
    """
    if not DATASETS_ROOT.exists():
        logger.error(f"/opt/datasets/ does not exist. Create it and place your POH files there.")
        return {}

    all_pdfs = list(DATASETS_ROOT.rglob("*.pdf")) + list(DATASETS_ROOT.rglob("*.PDF"))
    logger.info(f"Found {len(all_pdfs)} PDF files in {DATASETS_ROOT}")

    index = {}
    for p in all_pdfs:
        key = p.name.lower()
        if key not in index:
            index[key] = []
        index[key].append(p)

    return index


def match_files(pdf_index: dict[str, list[Path]]) -> dict[str, Path]:
    """
    Match PDF files to MANUAL_DOC_HINTS by search_terms.
    Returns dict: {doc_id: matched_path}
    """
    matches = {}
    all_paths = [p for paths in pdf_index.values() for p in paths]

    for doc_id, hint in MANUAL_DOC_HINTS.items():
        search_terms = hint["search_terms"]
        best_match = None

        for candidate_path in all_paths:
            name_lower = candidate_path.name.lower()
            full_lower = str(candidate_path).lower()

            # Score by how many search terms match
            score = sum(1 for term in search_terms if term.lower() in full_lower)

            if score > 0:
                if best_match is None or score > best_match[0]:
                    best_match = (score, candidate_path)

        if best_match:
            matches[doc_id] = best_match[1]
            logger.info(f"  MATCHED: {doc_id}")
            logger.info(f"    → {best_match[1]}")
        else:
            logger.warning(f"  NOT FOUND: {doc_id}")
            logger.warning(f"    Search terms: {search_terms}")

    return matches


def link_files(matches: dict[str, Path], dry_run: bool = False) -> dict[str, str]:
    """
    Copy matched files to their destination directories.
    Returns dict: {doc_id: dest_path_str}
    """
    ensure_dirs()
    results = {}

    for doc_id, src_path in matches.items():
        hint = MANUAL_DOC_HINTS[doc_id]
        dest_dir = hint["dest_dir"]
        dest_name = hint["dest_name"]
        dest_path = dest_dir / dest_name

        if dest_path.exists():
            logger.info(f"  ALREADY EXISTS: {doc_id} → {dest_path}")
            results[doc_id] = str(dest_path)
            continue

        if dry_run:
            logger.info(f"  DRY RUN: Would copy {src_path} → {dest_path}")
            results[doc_id] = str(dest_path)
            continue

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_path), str(dest_path))
            logger.success(f"  COPIED: {doc_id}")
            logger.success(f"    {src_path} → {dest_path}")
            results[doc_id] = str(dest_path)
        except Exception as e:
            logger.error(f"  FAILED to copy {doc_id}: {e}")

    return results


def print_status():
    """Print current link status for all manual documents."""
    print("\n" + "="*60)
    print("MANUAL DOCUMENT STATUS")
    print("="*60)

    for doc_id, hint in MANUAL_DOC_HINTS.items():
        dest_path = hint["dest_dir"] / hint["dest_name"]
        status = "✓ PRESENT" if dest_path.exists() else "✗ MISSING"
        size = ""
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            size = f"  ({size_mb:.1f} MB)"
        print(f"  {status}  {doc_id}{size}")
        if not dest_path.exists():
            print(f"           Expected: {dest_path}")

    print("="*60)
    print(f"Source directory: {DATASETS_ROOT}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FailureScope Dataset Linker — maps /opt/datasets/ to project"
    )
    parser.add_argument("--scan", action="store_true", help="Scan /opt/datasets/ and show matches")
    parser.add_argument("--link", action="store_true", help="Copy matched files to project")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied (no actual copy)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if not (args.scan or args.link or args.dry_run):
        parser.print_help()
        return

    logger.info(f"Scanning {DATASETS_ROOT}...")
    pdf_index = scan_datasets()

    if not pdf_index:
        sys.exit(1)

    print("\n" + "="*60)
    print("FILE MATCHING RESULTS")
    print("="*60)
    matches = match_files(pdf_index)

    print(f"\nMatched: {len(matches)}/{len(MANUAL_DOC_HINTS)} documents")

    if args.link or args.dry_run:
        print("\n" + "="*60)
        print("COPYING FILES" if args.link else "DRY RUN — NO FILES COPIED")
        print("="*60)
        link_files(matches, dry_run=args.dry_run)

    print_status()


if __name__ == "__main__":
    main()