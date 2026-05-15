"""
Ingestion: Schema-Constrained Spec Extractor
Sends specific page images to Ollama llava vision model.

CRITICAL DESIGN PRINCIPLE:
  - Vision model is a READER, not an INTERPRETER
  - Prompt constrains output to EXACT fields in schema only
  - If value cannot be clearly read → null (never inferred)
  - All outputs go to human validation before entering cache

Hallucination risk: LOW (schema-constrained, null on uncertainty)
Human validation gate: MANDATORY after this step
"""

import json
import re
from pathlib import Path
from typing import Optional
from loguru import logger

import ollama

from ingestion.schemas import POH_SCHEMA, PHYSICS_SCHEMA


# Vision model — pulled separately from mistral:7b
VISION_MODEL = "llava:7b"


class SpecExtractor:
    """
    Extracts specific numeric/text specs from page images
    using schema-constrained vision model prompts.
    """

    def __init__(
        self,
        model: str = VISION_MODEL,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        logger.info(f"SpecExtractor initialized: model={model}")

    # ─── PUBLIC ENTRY POINT ──────────────────────────────────────

    def extract_specs(
        self,
        page_record: dict,
        schema_section: str,
        schema_dict: dict,
    ) -> dict:
        """
        Extract specs from a single page using vision model.

        Args:
            page_record: From PDFProcessor.process() — contains image_base64
            schema_section: Key in schema e.g. "electrical_system"
            schema_dict: Schema dict e.g. POH_SCHEMA

        Returns:
            {
                field_id: {
                    "raw_value": extracted value or null,
                    "unit": expected unit,
                    "confidence": "high"|"low"|"null",
                    "page_number": int,
                    "doc_id": str,
                    "citation": dict,
                    "status": "pending_validation"
                }
            }
        """
        if schema_section not in schema_dict:
            raise ValueError(f"Schema section '{schema_section}' not found")

        fields = schema_dict[schema_section]["fields"]
        image_b64 = page_record.get("image_base64")

        if not image_b64:
            logger.error("No image_base64 in page record")
            return {}

        # Build extraction prompt
        prompt = self._build_prompt(fields, page_record["page_number"])

        # Call vision model
        raw_output = self._call_vision_model(image_b64, prompt)

        # Parse JSON response
        extracted = self._parse_response(raw_output, fields)

        # Annotate with metadata
        results = {}
        for field in fields:
            fid = field["field_id"]
            raw_val = extracted.get(fid)
            results[fid] = {
                "raw_value": raw_val,
                "unit": field["unit"],
                "confidence": "null" if raw_val is None else "low",  # human validates
                "page_number": page_record["page_number"],
                "doc_id": page_record["doc_id"],
                "citation": page_record["citation"],
                "layer_use": field["layer_use"],
                "status": "pending_validation",
                "image_path": page_record["image_path"],
            }

        logger.debug(
            f"Extracted {sum(1 for v in results.values() if v['raw_value'] is not None)}"
            f"/{len(fields)} fields from page {page_record['page_number']}"
        )
        return results

    def extract_all_sections(
        self,
        page_records: list[dict],
        schema_dict: dict,
        target_sections: Optional[list] = None,
    ) -> dict:
        """
        Extract all schema sections across all pages.
        Finds best page for each section based on OCR text hints.

        Args:
            page_records: All pages from PDFProcessor
            schema_dict: POH_SCHEMA or PHYSICS_SCHEMA
            target_sections: Optional list of sections to extract.
                             None = all sections.

        Returns:
            Merged dict of all extracted field results.
        """
        sections = target_sections or list(schema_dict.keys())
        all_results = {}

        for section in sections:
            logger.info(f"Extracting section: {section}")

            # Find best candidate pages for this section
            candidate_pages = self._find_candidate_pages(
                page_records,
                schema_dict[section]
            )

            if not candidate_pages:
                logger.warning(f"No candidate pages found for section: {section}")
                candidate_pages = page_records[:5]  # Fallback to first 5 pages

            # Try extraction on top 2 candidate pages
            for page_record in candidate_pages[:2]:
                results = self.extract_specs(page_record, section, schema_dict)
                found_values = {k: v for k, v in results.items() if v["raw_value"] is not None}

                # Merge — only add fields not yet found
                for fid, val in found_values.items():
                    if fid not in all_results or all_results[fid]["raw_value"] is None:
                        all_results[fid] = val

        logger.info(f"Total fields extracted: {len(all_results)}")
        return all_results

    # ─── VISION MODEL CALL ───────────────────────────────────────

    def _call_vision_model(self, image_b64: str, prompt: str) -> str:
        """Call Ollama llava vision model with page image and extraction prompt."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                options={
                    "temperature": 0.0,   # Deterministic
                    "num_predict": 1024,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Vision model call failed: {e}")
            raise RuntimeError(
                f"Failed to call {self.model} at {self.host}. "
                f"Ensure ollama pull {self.model} has been run. Error: {e}"
            )

    # ─── PROMPT BUILDER ──────────────────────────────────────────

    def _build_prompt(self, fields: list[dict], page_num: int) -> str:
        """
        Build schema-constrained extraction prompt.

        CRITICAL: Prompt instructs model to return null if value not clearly visible.
        Prevents hallucination by design.
        """
        field_list = "\n".join([
            f'  "{f["field_id"]}": <{f["unit"]} or null> // {f["description"]}'
            for f in fields
        ])

        field_ids = [f["field_id"] for f in fields]
        empty_json = json.dumps({fid: None for fid in field_ids}, indent=2)

        return f"""You are reading page {page_num} of an aircraft technical document.

TASK: Extract ONLY the specific values listed below from what you can see on this page.

STRICT RULES:
1. Output ONLY a valid JSON object. No explanation, no markdown, no backticks.
2. If a value is not clearly visible on this page, return null for that field.
3. NEVER guess, estimate, or infer a value. Only extract what is explicitly written.
4. Return the numeric value only, not units. Units are handled separately.
5. For text fields, return the exact text as written.

FIELDS TO EXTRACT:
{field_list}

If NONE of these fields appear on this page, return:
{empty_json}

OUTPUT (JSON only):"""

    # ─── JSON PARSING ────────────────────────────────────────────

    def _parse_response(self, raw: str, fields: list[dict]) -> dict:
        """Parse vision model JSON response. Returns dict with field_id keys."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        # Find JSON boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

        try:
            parsed = json.loads(cleaned)
            # Validate keys match schema
            valid_ids = {f["field_id"] for f in fields}
            return {k: v for k, v in parsed.items() if k in valid_ids}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw: {cleaned[:200]}")
            # Return all nulls on parse failure
            return {f["field_id"]: None for f in fields}

    # ─── PAGE CANDIDATE FINDER ───────────────────────────────────

    def _find_candidate_pages(
        self,
        page_records: list[dict],
        section_schema: dict,
    ) -> list[dict]:
        """
        Find pages most likely to contain target section using OCR text hints.
        Scores pages by keyword matches from section_hint fields.
        """
        fields = section_schema.get("fields", [])
        hints = []
        for f in fields:
            hint = f.get("section_hint", "").lower()
            hints.extend(hint.split())

        # Remove common words
        stop_words = {"or", "and", "the", "in", "of", "for", "a", "an", "to"}
        keywords = [w for w in hints if w not in stop_words and len(w) > 3]

        if not keywords:
            return page_records[:3]

        # Score each page
        scored = []
        for record in page_records:
            text_lower = (record.get("text") or "").lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored] if scored else page_records[:3]