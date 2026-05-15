"""
Ingestion: Human Validation CLI
THE MANDATORY GATE between vision extraction and physics_cache.json.

BUILD-TIME ONLY. Runs once per document set.
Every extracted value must be accepted, rejected, or corrected by a human
before it enters validated_specs.json or physics_cache.json.

Displays:
  - The extracted value
  - The source document + page
  - The page image path (open it to verify)
  - Accept / Reject / Correct options

Source: Established architectural principle:
"Human is final arbiter. Zero values enter cache without human confirmation."
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from ingestion.config import VALIDATED_DIR
PROJECT_ROOT = Path(__file__).parent.parent


class ValidationCLI:
    """
    Terminal UI for validating extracted specs.
    Shows extracted value + source info.
    User accepts, rejects, or corrects each value.
    """

    def __init__(self, validator_name: str = "developer"):
        self.validator_name = validator_name
        self.session_start = datetime.utcnow().isoformat()

    # ─── MAIN VALIDATION LOOP ────────────────────────────────────

    def validate_extracted_specs(
        self,
        extracted: dict,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Interactive validation of all extracted specs.

        Args:
            extracted: Output from SpecExtractor.extract_all_sections()
            output_path: Where to save validated specs JSON.
                         Defaults to data/validated/{doc_id}_validated.json

        Returns:
            validated_specs dict (accepted + corrected values only)
        """
        if not extracted:
            logger.warning("No extracted specs to validate")
            return {}

        # Get doc_id from first record
        first = next(iter(extracted.values()))
        doc_id = first.get("doc_id", "unknown")

        if output_path is None:
            output_path = VALIDATED_DIR / f"{doc_id}_validated.json"

        # Load existing validation if resuming
        validated = self._load_existing(output_path)
        already_done = set(validated.keys())

        # Get pending fields
        pending = {
            fid: spec
            for fid, spec in extracted.items()
            if fid not in already_done
        }

        if not pending:
            logger.info("All fields already validated. Nothing to do.")
            return validated

        print(self._header(doc_id, len(pending), len(already_done)))

        for i, (field_id, spec) in enumerate(pending.items(), 1):
            result = self._validate_single(
                field_id, spec, i, len(pending)
            )

            if result["action"] == "accept":
                validated[field_id] = self._build_validated_record(
                    field_id, spec, result["final_value"], "accepted"
                )
            elif result["action"] == "correct":
                validated[field_id] = self._build_validated_record(
                    field_id, spec, result["final_value"], "corrected",
                    correction_note=result.get("note", "")
                )
            elif result["action"] == "reject":
                validated[field_id] = self._build_validated_record(
                    field_id, spec, None, "rejected",
                    correction_note=result.get("note", "")
                )
            elif result["action"] == "skip":
                logger.info(f"Skipped: {field_id}")
                continue
            elif result["action"] == "quit":
                logger.info("Validation session ended by user.")
                self._save(validated, output_path)
                print(f"\nProgress saved to: {output_path}")
                print("Run again to continue from where you left off.")
                break

            # Save after each decision (crash-safe)
            self._save(validated, output_path)

        # Summary
        accepted = sum(1 for v in validated.values() if v["status"] == "accepted")
        corrected = sum(1 for v in validated.values() if v["status"] == "corrected")
        rejected = sum(1 for v in validated.values() if v["status"] == "rejected")

        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE")
        print(f"  Accepted:  {accepted}")
        print(f"  Corrected: {corrected}")
        print(f"  Rejected:  {rejected}")
        print(f"  Output:    {output_path}")
        print(f"{'='*60}\n")

        return validated

    # ─── SINGLE FIELD VALIDATION ─────────────────────────────────

    def _validate_single(
        self,
        field_id: str,
        spec: dict,
        index: int,
        total: int,
    ) -> dict:
        """Validate a single extracted field interactively."""

        raw_value = spec.get("raw_value")
        unit = spec.get("unit", "")
        doc_id = spec.get("doc_id", "")
        page_num = spec.get("page_number", "?")
        image_path = spec.get("image_path", "")
        citation = spec.get("citation", {})
        layer_use = spec.get("layer_use", "")

        # Display field info
        print(f"\n{'─'*60}")
        print(f"[{index}/{total}] Field: {field_id}")
        print(f"{'─'*60}")
        print(f"  Document:  {citation.get('document', doc_id)}")
        print(f"  Page:      {page_num}")
        print(f"  Used by:   {layer_use}")
        print(f"  Unit:      {unit}")
        print(f"")

        if raw_value is not None:
            print(f"  EXTRACTED: {raw_value}")
        else:
            print(f"  EXTRACTED: [null — not found on this page]")

        print(f"")
        print(f"  Image:     {image_path}")
        print(f"  (Open image to verify against source document)")
        print(f"")

        # Open image if possible
        self._try_open_image(image_path)

        # Get user decision
        while True:
            if raw_value is not None:
                prompt = "  [A]ccept  [C]orrect  [R]eject  [S]kip  [Q]uit > "
            else:
                prompt = "  [M]anual entry  [R]eject  [S]kip  [Q]uit > "

            choice = input(prompt).strip().upper()

            if choice == "A" and raw_value is not None:
                return {"action": "accept", "final_value": raw_value}

            elif choice in ("C", "M"):
                correct_val = input(f"  Enter correct value (unit: {unit}): ").strip()
                if not correct_val:
                    print("  No value entered. Try again.")
                    continue
                note = input("  Note (optional, e.g. 'page 47 table'): ").strip()
                return {
                    "action": "correct",
                    "final_value": correct_val,
                    "note": note,
                }

            elif choice == "R":
                note = input("  Rejection reason (optional): ").strip()
                return {"action": "reject", "note": note}

            elif choice == "S":
                return {"action": "skip"}

            elif choice == "Q":
                return {"action": "quit"}

            else:
                print("  Invalid choice. Try again.")

    # ─── HELPERS ─────────────────────────────────────────────────

    def _try_open_image(self, image_path: str):
        """Attempt to open image with system viewer for human review."""
        if not image_path or not Path(image_path).exists():
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(
                    ["xdg-open", image_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass  # Silent — viewing is optional

    def _build_validated_record(
        self,
        field_id: str,
        spec: dict,
        final_value,
        status: str,
        correction_note: str = "",
    ) -> dict:
        return {
            "field_id": field_id,
            "final_value": final_value,
            "raw_extracted_value": spec.get("raw_value"),
            "unit": spec.get("unit"),
            "status": status,
            "correction_note": correction_note,
            "doc_id": spec.get("doc_id"),
            "page_number": spec.get("page_number"),
            "image_path": spec.get("image_path"),
            "citation": spec.get("citation"),
            "layer_use": spec.get("layer_use"),
            "validated_by": self.validator_name,
            "validated_at": datetime.utcnow().isoformat(),
        }

    def _load_existing(self, output_path: Path) -> dict:
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
        return {}

    def _save(self, validated: dict, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(validated, f, indent=2)

    def _header(self, doc_id: str, pending: int, done: int) -> str:
        return f"""
{'='*60}
FailureScope — Document Validation
{'='*60}
Document:  {doc_id}
Validator: {self.validator_name}
Pending:   {pending} fields
Done:      {done} fields (already validated)
{'='*60}

For each field:
  Open the image file shown to see the source page.
  Verify the extracted value against what is printed.
  Accept if correct. Correct if wrong. Reject if not found.

IMPORTANT: Every value here will be used as ground truth
in the physics simulation. No assumptions. Document only.
{'='*60}
"""