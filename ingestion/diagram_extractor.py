"""
Ingestion: Diagram Extractor
Vision-based extraction for pages that are image-only (diagrams, tables, schematics).

Architecture (per user design):
  Stage 1: Universal Page Classifier — llava:7b classifies page type
           No aircraft context in prompt. Scalable to any aircraft.
  Stage 2: OCR gate — only process pages with <50 chars OCR text
  Stage 3: Generic Value Extraction — extracts label+value+unit from image
           No aircraft context. Document labels itself.
  Stage 4: Label-to-field mapping — deterministic dictionary, not LLM
  Stage 5: Human validation gate — mandatory, same as text extraction

Hallucination risk: LOW
  - Classifier only returns boolean flags
  - Extractor reads printed labels, does not infer
  - Label mapping is a deterministic dictionary
  - Human gate is mandatory for all values
"""

import json
import base64
import re
from pathlib import Path
from typing import Optional
from loguru import logger

import ollama

from ingestion.config import VALIDATED_DIR, PAGE_IMAGES_DIR


# ─────────────────────────────────────────────────────────────
# LABEL-TO-FIELD MAPPING
# Deterministic dictionary. No model. Extend per document type.
# Keys: lowercase, stripped labels found in document diagrams/tables
# Values: field_id in physics_cache.json schema
# ─────────────────────────────────────────────────────────────

LABEL_TO_FIELD = {
    # Engine performance
    "max rpm":                    "max_rpm",
    "maximum rpm":                "max_rpm",
    "rated rpm":                  "max_rpm",
    "crankshaft speed":           "max_rpm",
    "maximum continuous":         "max_rpm",
    "max continuous rpm":         "max_rpm",
    "bhp":                        "max_power_bhp",
    "brake horsepower":           "max_power_bhp",
    "rated bhp":                  "max_power_bhp",
    "maximum power":              "max_power_bhp",
    "rated maximum":              "max_power_bhp",
    "engine model":               "engine_model",
    "engine type":                "engine_model",
    "model":                      "engine_model",

    # Electrical
    "system voltage":             "system_voltage",
    "bus voltage":                "system_voltage",
    "nominal voltage":            "system_voltage",
    "alternator output":          "alternator_output_amps",
    "alternator":                 "alternator_output_amps",
    "generator output":           "alternator_output_amps",
    "battery capacity":           "battery_capacity_ah",
    "battery":                    "battery_capacity_ah",
    "ampere hour":                "battery_capacity_ah",
    "amp hour":                   "battery_capacity_ah",
    "min voltage":                "min_fuel_pump_voltage",
    "minimum voltage":            "min_fuel_pump_voltage",
    "fuel pump voltage":          "min_fuel_pump_voltage",
    "fuel pump minimum":          "min_fuel_pump_voltage",

    # Fuel system
    "fuel flow":                  "cruise_fuel_flow_gph",
    "fuel consumption":           "cruise_fuel_flow_gph",
    "gph":                        "cruise_fuel_flow_gph",
    "fuel capacity":              "fuel_capacity_total_gal",
    "total fuel":                 "fuel_capacity_total_gal",
    "usable fuel":                "fuel_capacity_total_gal",
    "tank capacity":              "fuel_capacity_left_gal",
    "each tank":                  "fuel_capacity_left_gal",

    # Flight envelope
    "vne":                        "vne_knots",
    "never exceed":               "vne_knots",
    "never-exceed":               "vne_knots",
    "vs0":                        "vs0_knots",
    "stall speed":                "vs0_knots",
    "stalling speed":             "vs0_knots",
    "service ceiling":            "service_ceiling_ft",
    "ceiling":                    "service_ceiling_ft",
    "max weight":                 "max_gross_weight_lbs",
    "maximum weight":             "max_gross_weight_lbs",
    "gross weight":               "max_gross_weight_lbs",
    "max takeoff":                "max_gross_weight_lbs",
}


# ─────────────────────────────────────────────────────────────
# CLASSIFIER PROMPT
# Generic — no aircraft context, no document name
# Scalable across all aircraft types
# ─────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """Look at this page image.
Respond ONLY with a JSON object. No explanation.

{
  "has_table": true or false,
  "has_diagram": true or false,
  "has_schematic": true or false,
  "has_dense_text": true or false,
  "page_type": "table" or "diagram" or "schematic" or "text" or "mixed" or "blank"
}

Definitions:
- has_table: rows and columns of data with numbers
- has_diagram: technical drawing, wiring diagram, or labeled illustration
- has_schematic: circuit diagram or flow diagram with symbols
- has_dense_text: more than half the page is readable text paragraphs
- page_type: the dominant content type"""


# ─────────────────────────────────────────────────────────────
# EXTRACTOR PROMPT
# Generic — extracts labels + values from any document image
# No aircraft context = scalable
# ─────────────────────────────────────────────────────────────

EXTRACTOR_PROMPT = """Look at this page image containing a technical diagram, table, or schematic.

Extract all labeled numeric values visible on the page.
Respond ONLY with a JSON array. No explanation.

[
  {"label": "the label text exactly as printed", "value": "the numeric value as printed", "unit": "unit if shown or null"},
  ...
]

Rules:
- Include only values that have a clear label
- Copy the label exactly as it appears on the page
- Copy the value exactly as it appears (do not convert units)
- If a value is unclear or partially visible, use null for value
- Never guess or infer values
- Return empty array [] if no labeled values found"""


# ─────────────────────────────────────────────────────────────
# DIAGRAM EXTRACTOR CLASS
# ─────────────────────────────────────────────────────────────

class DiagramExtractor:
    """
    Extracts values from image-heavy document pages using llava:7b.

    Pipeline per page:
      1. Check OCR text length — skip if >= 50 chars (text extraction covers it)
      2. Classify page type with generic prompt
      3. If diagram/table/schematic: extract label+value pairs
      4. Map labels to field_ids using LABEL_TO_FIELD dictionary
      5. Return structured extraction records for human validation
    """

    OCR_TEXT_THRESHOLD = 50  # Pages below this get vision extraction

    def __init__(
        self,
        model: str = "llava:7b",
        host: str = "http://localhost:11434",
    ):
        self.model  = model
        self.client = ollama.Client(host=host)
        logger.info(f"DiagramExtractor initialized: model={model}")

    # ── PUBLIC ENTRY POINT ───────────────────────────────────

    def extract_from_document(
        self,
        doc_id: str,
        page_records: list[dict],
        citation: dict,
    ) -> dict:
        """
        Process all pages in a document.
        Only applies vision extraction to pages with insufficient OCR text.

        Returns dict of field_id → extraction record (same format as OCRTextExtractor)
        """
        results = {}
        image_pages = []
        skipped_text = 0

        for record in page_records:
            page_num  = record.get("page_number", 0)
            ocr_text  = record.get("text", "") or ""
            img_path  = record.get("image_path")

            if len(ocr_text.strip()) >= self.OCR_TEXT_THRESHOLD:
                skipped_text += 1
                continue

            if not img_path or not Path(img_path).exists():
                logger.debug(f"Page {page_num}: no image file, skipping")
                continue

            image_pages.append((page_num, img_path, ocr_text))

        logger.info(
            f"DiagramExtractor: {len(image_pages)} image pages to process, "
            f"{skipped_text} skipped (sufficient OCR text)"
        )

        for page_num, img_path, ocr_text in image_pages:
            logger.debug(f"Processing page {page_num}...")

            # Stage 1: Classify page
            classification = self._classify_page(img_path)
            if not classification:
                continue

            page_type = classification.get("page_type", "blank")
            has_content = any([
                classification.get("has_table"),
                classification.get("has_diagram"),
                classification.get("has_schematic"),
            ])

            if not has_content or page_type == "blank":
                logger.debug(f"Page {page_num}: {page_type}, no extractable content")
                continue

            logger.debug(f"Page {page_num}: {page_type} — extracting values")

            # Stage 2: Extract label+value pairs
            raw_extractions = self._extract_values(img_path)
            if not raw_extractions:
                continue

            logger.debug(f"Page {page_num}: found {len(raw_extractions)} labeled values")

            # Stage 3: Map labels to field_ids
            for item in raw_extractions:
                label = (item.get("label") or "").lower().strip()
                value = item.get("value")
                unit  = item.get("unit")

                if not label or value is None:
                    continue

                field_id = self._map_label_to_field(label)
                if not field_id:
                    logger.debug(f"Page {page_num}: label '{label}' not in mapping, skipping")
                    continue

                # Only take first match per field_id
                if field_id not in results:
                    results[field_id] = {
                        "raw_value":          value,
                        "unit":               unit,
                        "confidence":         "vision_extraction",
                        "page_number":        page_num,
                        "doc_id":             doc_id,
                        "citation":           citation,
                        "layer_use":          self._get_layer_use(field_id),
                        "status":             "pending_validation",
                        "extraction_method":  "vision_llava",
                        "original_label":     item.get("label"),
                        "page_type":          page_type,
                    }
                    logger.success(
                        f"  FOUND via vision: {field_id} = {value!r} "
                        f"(label: '{item.get('label')}', page {page_num})"
                    )

        found = len(results)
        logger.info(f"DiagramExtractor complete: {found} fields found across {len(image_pages)} image pages")
        return results

    # ── CLASSIFICATION ───────────────────────────────────────

    def _classify_page(self, img_path: str) -> Optional[dict]:
        """Classify page type using generic vision prompt."""
        img_b64 = self._encode_image(img_path)
        if not img_b64:
            return None

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role":    "user",
                    "content": CLASSIFIER_PROMPT,
                    "images":  [img_b64],
                }],
                options={"temperature": 0.0, "num_predict": 128},
            )
            raw = response["message"]["content"].strip()
            return self._parse_json_response(raw)
        except Exception as e:
            logger.debug(f"Classification failed for {img_path}: {e}")
            return None

    # ── VALUE EXTRACTION ─────────────────────────────────────

    def _extract_values(self, img_path: str) -> list[dict]:
        """Extract labeled values from diagram/table page."""
        img_b64 = self._encode_image(img_path)
        if not img_b64:
            return []

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role":    "user",
                    "content": EXTRACTOR_PROMPT,
                    "images":  [img_b64],
                }],
                options={"temperature": 0.0, "num_predict": 512},
            )
            raw = response["message"]["content"].strip()
            parsed = self._parse_json_response(raw)

            if isinstance(parsed, list):
                return parsed
            return []
        except Exception as e:
            logger.debug(f"Extraction failed for {img_path}: {e}")
            return []

    # ── LABEL MAPPING ────────────────────────────────────────

    def _map_label_to_field(self, label: str) -> Optional[str]:
        """
        Map extracted label to schema field_id.
        Deterministic dictionary lookup — no model involved.
        Tries exact match first, then partial match.
        """
        label_clean = label.lower().strip()

        # Exact match
        if label_clean in LABEL_TO_FIELD:
            return LABEL_TO_FIELD[label_clean]

        # Partial match — label contains a known key
        for key, field_id in LABEL_TO_FIELD.items():
            if key in label_clean:
                return field_id

        return None

    # ── UTILITIES ────────────────────────────────────────────

    def _encode_image(self, img_path: str) -> Optional[str]:
        """Encode image file to base64 string for Ollama API."""
        path = Path(img_path)
        if not path.exists():
            logger.debug(f"Image not found: {img_path}")
            return None
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception as e:
            logger.debug(f"Image encode failed: {e}")
            return None

    def _parse_json_response(self, raw: str) -> Optional[object]:
        """Parse JSON from llava response, stripping markdown fences."""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Find JSON boundaries
        if cleaned.startswith("["):
            end = cleaned.rfind("]") + 1
            if end > 0:
                cleaned = cleaned[:end]
        elif cleaned.startswith("{"):
            end = cleaned.rfind("}") + 1
            if end > 0:
                cleaned = cleaned[:end]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug(f"JSON parse failed: {cleaned[:100]}")
            return None

    def _get_layer_use(self, field_id: str) -> str:
        layer_map = {
            "max_rpm":               "layer1_constraint_graph",
            "max_power_bhp":         "layer2_simulation",
            "engine_model":          "layer1_constraint_graph",
            "system_voltage":        "layer1_constraint_graph",
            "alternator_output_amps":"layer1_constraint_graph",
            "battery_capacity_ah":   "layer1_constraint_graph",
            "min_fuel_pump_voltage": "layer1_constraint_graph",
            "cruise_fuel_flow_gph":  "layer2_simulation",
            "fuel_capacity_total_gal":"layer0_defaults",
            "fuel_capacity_left_gal":"layer0_defaults",
            "vne_knots":             "layer0_validator",
            "vs0_knots":             "layer0_validator",
            "service_ceiling_ft":    "layer0_validator",
            "max_gross_weight_lbs":  "layer2_simulation",
        }
        return layer_map.get(field_id, "unknown")


    def save_extracted(self, results: dict, doc_id: str) -> Path:
        """Save extraction results to pending validation file."""
        output_path = VALIDATED_DIR / f"{doc_id}_diagram_extracted.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved diagram extracted specs: {output_path}")
        return output_path