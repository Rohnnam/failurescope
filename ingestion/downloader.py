"""
Ingestion: Automatic Document Downloader
Downloads all Source Group 2 (physics) and Source Group 3 (NTSB) documents.
Source Group 1 (POH) must be placed manually — see README.

All downloads are official government or NASA sources.
Every file is stored with its citation metadata.
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from loguru import logger

# Project root — resolved relative to this file
from ingestion.config import DATA_ROOT, ensure_dirs


# ─────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MANIFEST
# Only Source Group 2 (physics) + Source Group 3 (NTSB)
# Source Group 1 (POH) = manual placement
# ─────────────────────────────────────────────────────────────

AUTO_DOWNLOADS = [

    # ── Physics documents ──────────────────────────────────────
    {
        "doc_id": "faa_phak_8083_25c",
        "url": "https://www.faa.gov/sites/faa.gov/files/2023-11/pilot_handbook.pdf",
        "dest": "data/physics_docs/faa_h8083_25c_phak.pdf",
        "citation": {
            "document": "Pilot's Handbook of Aeronautical Knowledge",
            "id": "FAA-H-8083-25C",
            "publisher": "FAA",
            "year": 2023,
            "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/phak",
        },
        "size_mb_approx": 180,
    },
    {
        "doc_id": "faa_afh_8083_3c",
        "url": "https://www.faa.gov/sites/faa.gov/files/2023-11/airplane_flying_handbook.pdf",
        "dest": "data/physics_docs/faa_h8083_3c_afh.pdf",
        "citation": {
            "document": "Airplane Flying Handbook",
            "id": "FAA-H-8083-3C",
            "publisher": "FAA",
            "year": 2023,
            "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/airplane_handbook",
        },
        "size_mb_approx": 260,
    },
    {
        "doc_id": "faa_ac43_13_1b",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_Full.pdf",
        "dest": "data/physics_docs/faa_ac43_13_1b.pdf",
        "citation": {
            "document": "Acceptable Methods Techniques and Practices - Aircraft Inspection and Repair",
            "id": "FAA AC 43.13-1B",
            "publisher": "FAA",
            "year": 1998,
        },
        "size_mb_approx": 50,
    },
    {
        "doc_id": "nasa_ga_reliability_2001",
        "url": "https://www.cs.odu.edu/~mln/ltrs-pdfs/NASA-2001-cr210647.pdf",
        "dest": "data/physics_docs/nasa_cr2001_210647.pdf",
        "citation": {
            "document": "General Aviation Aircraft Reliability Study",
            "id": "NASA/CR-2001-210647",
            "publisher": "NASA",
            "year": 2001,
        },
        "size_mb_approx": 5,
    },

    # ── NTSB data ──────────────────────────────────────────────
    {
        "doc_id": "ntsb_avall",
        "url": "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C3494",
        "dest": "data/ntsb_data/avall.zip",
        "citation": {
            "document": "NTSB Aviation Accident Database 1982-present",
            "publisher": "NTSB",
            "url": "https://data.ntsb.gov/avdata",
        },
        "size_mb_approx": 500,
    },
    {
        "doc_id": "ntsb_pre1982",
        "url": "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C3495",
        "dest": "data/ntsb_data/pre1982.zip",
        "citation": {
            "document": "NTSB Aviation Accident Database 1962-1981",
            "publisher": "NTSB",
            "url": "https://data.ntsb.gov/avdata",
        },
        "size_mb_approx": 100,
    },
]


# ─────────────────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────────────────

class DocumentDownloader:

    def __init__(self):
        self.root = DATA_ROOT
        self.manifest_path = DATA_ROOT / "download_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download_file(self, url: str, dest: Path, doc_id: str, size_mb: int):
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {doc_id} (~{size_mb}MB) → {dest}")
        logger.info(f"Source: {url}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*",
        }

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=120)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 / total
                            if downloaded % (1024 * 1024 * 10) < 65536:
                                logger.info(f"  {doc_id}: {pct:.1f}%")

            logger.success(f"Downloaded: {dest}")
            return True
        except Exception as e:
            logger.error(f"Download failed for {doc_id}: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def download_all(self, force: bool = False, skip_large: bool = False):
        """
        Download all auto-downloadable documents.

        Args:
            force: Re-download even if file exists
            skip_large: Skip files > 100MB (for testing)
        """
        results = {"success": [], "skipped": [], "failed": []}

        for item in AUTO_DOWNLOADS:
            doc_id = item["doc_id"]
            dest = DATA_ROOT / item["dest"].replace("data/", "")
            url = item["url"]
            size_mb = item.get("size_mb_approx", 0)

            # Skip large files if requested
            if skip_large and size_mb > 100:
                logger.warning(f"Skipping {doc_id} ({size_mb}MB) — skip_large=True")
                results["skipped"].append(doc_id)
                continue

            # Skip if already downloaded
            if dest.exists() and not force:
                if doc_id in self.manifest:
                    logger.info(f"Already downloaded: {doc_id}")
                    results["skipped"].append(doc_id)
                    continue

            # Download
            success = self._download_file(url, dest, doc_id, size_mb)

            if success:
                self.manifest[doc_id] = {
                    "doc_id": doc_id,
                    "dest": str(dest),
                    "citation": item["citation"],
                    "sha256": self._sha256(dest),
                    "size_bytes": dest.stat().st_size,
                    "status": "downloaded",
                }
                self._save_manifest()
                results["success"].append(doc_id)
            else:
                results["failed"].append(doc_id)

        logger.info(f"Download complete: {len(results['success'])} success, "
                    f"{len(results['skipped'])} skipped, "
                    f"{len(results['failed'])} failed")

        if results["failed"]:
            logger.warning(f"Failed downloads: {results['failed']}")
            logger.warning("Try downloading these manually and placing in correct data/ folder.")

        return results

    def register_manual_document(self, doc_id: str, file_path: str, citation: dict):
        """
        Register a manually placed document (POH files).
        Call this after manually placing a POH in the correct data/ folder.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self.manifest[doc_id] = {
            "doc_id": doc_id,
            "dest": str(path),
            "citation": citation,
            "sha256": self._sha256(path),
            "size_bytes": path.stat().st_size,
            "status": "manually_placed",
        }
        self._save_manifest()
        logger.success(f"Registered manual document: {doc_id} → {path}")

    def status(self):
        """Print status of all expected documents."""
        print("\n=== Document Status ===\n")

        # Auto downloads
        print("AUTO DOWNLOADS (physics + NTSB):")
        for item in AUTO_DOWNLOADS:
            doc_id = item["doc_id"]
            dest = DATA_ROOT / item["dest"].replace("data/", "")
            exists = dest.exists()
            registered = doc_id in self.manifest
            status = "✓ OK" if (exists and registered) else ("FILE EXISTS" if exists else "✗ MISSING")
            print(f"  {status:12} {doc_id}")

        # Manual POHs
        print("\nMANUAL DOCUMENTS (POH files — you must place these):")
        manual_docs = [
            ("cessna_182q_poh_1977", "data/aircraft_docs/cessna_182/cessna_182q_poh_1977.pdf"),
            ("cessna_182p_poh_1976", "data/aircraft_docs/cessna_182/cessna_182p_poh_1976.pdf"),
            ("cessna_172n_poh_1978", "data/aircraft_docs/cessna_172/cessna_172n_poh_1978.pdf"),
            ("beechcraft_bonanza_v35_poh", "data/aircraft_docs/beechcraft_bonanza/bonanza_v35_poh.pdf"),
            ("piper_seneca_ii_poh", "data/aircraft_docs/piper_seneca/seneca_ii_poh.pdf"),
            ("piper_cherokee_warrior_mm", "data/aircraft_docs/piper_cherokee/cherokee_warrior_mm_1995.pdf"),
            ("cessna_310k_owners_manual", "data/aircraft_docs/cessna_310/cessna_310k_1966.pdf"),
        ]
        for doc_id, rel_path in manual_docs:
            dest = self.root / rel_path
            exists = dest.exists()
            registered = doc_id in self.manifest
            status = "✓ OK" if (exists and registered) else ("FILE EXISTS (not registered)" if exists else "✗ MISSING")
            print(f"  {status:30} {rel_path}")

        print()


if __name__ == "__main__":
    downloader = DocumentDownloader()
    downloader.status()

    import sys
    if "--download" in sys.argv:
        skip_large = "--skip-large" in sys.argv
        downloader.download_all(skip_large=skip_large)