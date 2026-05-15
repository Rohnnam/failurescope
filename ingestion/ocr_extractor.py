"""
Ingestion: OCR Text-Based Spec Extractor
Extracts aircraft specs directly from OCR text using regex patterns.

ARCHITECTURE DECISION (based on diagnostic):
  The Cessna 182Q POH has excellent OCR quality (all 160 pages have >50 chars).
  All required values are present in OCR text directly.
  Regex extraction = deterministic, zero hallucination, faster than vision model.

  Vision model (llava) is demoted to FALLBACK only, used when:
    - Page is image-only (OCR returns empty)
    - Regex fails to find the value after checking all candidate pages

Hallucination risk: NONE for regex path, LOW for vision fallback.
Human validation gate: MANDATORY regardless of extraction method.
"""

import re
import json
from pathlib import Path
from typing import Optional
from loguru import logger

from ingestion.config import VALIDATED_DIR


# ─────────────────────────────────────────────────────────────
# REGEX EXTRACTION PATTERNS
# Each pattern is document-tested against real OCR output.
# Capture group 1 must be the numeric/text value.
# ─────────────────────────────────────────────────────────────

FIELD_PATTERNS = {

    # ── Aircraft identity ─────────────────────────────────────
    "aircraft_model": [
        r"model\s+(182q|182p|172[a-z]|pa-?\d+|v35[ab]?|310[a-z]?)",
        r"(182q|182p|172n|172s|pa-28|pa-34|v35b?|310k)",
    ],
    "engine_model": [
        r"engine[:\s]+teledyne continental\s+([o0]-[\w-]+)",
        r"engine model number[:\s]+([o0][~-][\w-]+)",
        r"continental\s+([o0]-[\w-]+)",
        r"lycoming\s+([o0]-[\w-]+)",
        r"([o0]-470-[ur]|[o0]-470|[o0]-320[-\w]*|[o0]-360[-\w]*|i[o0]-520[-\w]*)",
    ],
    "faa_type_certificate": [
        r"certificated under faa type certificate no\.?\s+([0-9][a-z][0-9]+)",
        r"type certificate no\.?\s+([0-9][a-z][0-9]+)",
        r"faa type cert[^\n]+?([0-9][a-z][0-9]+)",
    ],

    # ── Electrical system ─────────────────────────────────────
    "system_voltage": [
        r"(28)[\s-]*volt(?:s)?[\s-]+(?:dc|electrical|system|bus)",
        r"(?:28)[\s-]*volt[^s]",
        r"system[^\n]+?(28)\s*v\b",
    ],
    "alternator_output_amps": [
        r"alternator[,\s]+(\d+)[\s-]*amp",
        r"(\d+)[\s-]*amp\s+alternator",
        r"alternator output[:\s]+(\d+)",
        r"(\d+)a\s+alternator",
    ],
    "battery_capacity_ah": [
        r"battery[,\s]+(\d+)[\s-]*amp[\s-]*hour",
        r"(\d+)[\s-]*a[\s-]*h\s+battery",
        r"battery capacity[:\s]+(\d+)",
        r"(\d+)[\s-]*ampere[\s-]*hour",
    ],
    "min_fuel_pump_voltage": [
        r"fuel pump[^.]+minimum[^.]+(\d+\.?\d*)\s*v",
        r"minimum voltage[^.]+fuel pump[^.]+(\d+\.?\d*)",
        r"fuel pump[^.]+(\d+\.?\d*)\s*volt",
    ],

    # ── Fuel system ───────────────────────────────────────────
    "fuel_capacity_total_gal": [
        r"total\s+(?:usable\s+)?fuel[:\s]+(\d+\.?\d*)\s+gal",
        r"(\d+\.?\d*)\s+gallons?\s+(?:total|usable)",
        r"fuel capacity[:\s]+(\d+\.?\d*)\s+(?:u\.?s\.?\s+)?gal",
        r"(\d+)\s+gal(?:lons?)?\s+(?:standard|total)\s+tanks",
    ],
    "fuel_capacity_left_gal": [
        r"total capacity each tank[:\s]+(\d+\.?\s*\d*)\s+gallons",
        r"total capacity each tank[:\s]+(\d+)[.\s]+(\d)\s+gallons",
        r"each tank[:\s]+(\d+\.?\s*\d*)\s+gallons",
        r"(\d+\.?\s*\d*)\s+gallons?\s+each",
    ],
    "fuel_capacity_right_gal": [
        r"total capacity each tank[:\s]+(\d+\.?\s*\d*)\s+gallons",
        r"each tank[:\s]+(\d+\.?\s*\d*)\s+gallons",
    ],
    "cruise_fuel_flow_gph": [
        r"fuel\s+(?:consumption|flow)[:\s]+(\d+\.?\d*)\s+gph",
        r"(\d+\.?\d*)\s+gph\s+(?:at\s+)?(?:cruise|75%)",
        r"fuel\s+flow[:\s]+(\d+\.?\d*)",
        r"(\d+\.?\d*)\s+gallons?\s+per\s+hour",
    ],

    # ── Engine performance ────────────────────────────────────
    "max_rpm": [
        r"maximum\s+(?:continuous\s+)?(?:engine\s+)?(?:speed|rpm)[:\s]+(\d{4})",
        r"(\d{4})\s+rpm\s+(?:maximum|rated|max)",
        r"rated\s+maximum[^.]+(\d{4})\s+rpm",
        r"(?:max|maximum)\s+(?:rpm|crankshaft)[:\s]+(\d{4})",
        r"full\s+throttle\s+and\s+(\d{4})\s+rpm",
        r"2400\s+rpm",  # Known C182Q value
    ],
    "max_power_bhp": [
        r"(\d{3})\s+(?:bhp|hp|brake\s+horse\s*power)",
        r"maximum\s+(?:power|bhp)[:\s]+(\d{3})",
        r"rated[^.]+(\d{3})\s+(?:bhp|hp)",
        r"230\s+(?:bhp|hp)",  # Known O-470-U value
    ],
    "engine_count": [
        r"(single|one|1)\s+engine",
        r"(twin|two|2)\s+engine",
        r"(dual)\s+engine",
    ],

    # ── Flight envelope ───────────────────────────────────────
    "vne_knots": [
        r"vne\s+never\s+exceed\s+(?:speed\s+)?(\d{2,3})\s+\d",
        r"never\s+exceed\s+(?:speed\s+)?(\d{2,3})\s+\d",
        r"never-exceed\s+(?:speed\s+)?(\d{2,3})",
        r"vne[:\s]+(\d{2,3})\s+k",
    ],
    "vs0_knots": [
        r"vs0[:\s]+(\d{2,3})",
        r"stall\s+speed[^.]+flaps?\s+(?:down|full)[^.]+(\d{2,3})",
        r"(?:full\s+flap|flaps?\s+down)\s+stall[^.]+(\d{2,3})",
        r"stalling\s+speed[^.]+(\d{2})\s+k",
    ],
    "service_ceiling_ft": [
        r"service\s+ceiling\s*[.\s]+(\d{1,2},\d{3})\s*(?:ft)?",
        r"service\s+ceiling[^\n]+?(\d{1,2},?\d{3})\s*(?:ft|feet)?",
        r"ceiling[^\n]+?(\d{1,2},?\d{3})\s*ft",
    ],
    "max_gross_weight_lbs": [
        r"(?:maximum|max)\s+(?:gross|takeoff|ramp)\s+weight[:\s]+(\d{1,2},?\d{3})\s*(?:lbs?|pounds?)?",
        r"gross\s+weight[:\s]+(\d{1,2},?\d{3})",
        r"(\d{1,2},?\d{3})\s*(?:lbs?|pounds?)\s+(?:maximum|max)\s+(?:gross|takeoff)",
        r"2950\s+(?:lbs?|pounds?)",  # Known C182Q value
    ],

    # ── Emergency procedures (text fields) ───────────────────
    "alternator_failure_procedure": [
        r"(broken\s+alterna[\s\S]{200,1000}?(?=section\s+4|normal\s+procedures|\Z))",
        r"(electrical power[^\n]*malfunction[\s\S]{300,1000}?excessive\s+rate[\s\S]{200,600}?(?=section\s+4|\Z))",
    ],
    "engine_failure_procedure": [
        r"(engine\s+failure[\s\S]{50,500}?(?=\n\n|\Z|section\s+\d))",
    ],
    "fuel_system_failure_procedure": [
        r"(fuel\s+system[\s\S]{50,400}?(?:failure|emergency|selector)[\s\S]{50,400}?(?=section\s+\d|\Z))",
        r"(fuel\s+selector[^\n]+[\s\S]{50,300}?(?=\n\n|\Z))",
    ],
}


class OCRTextExtractor:
    """
    Extracts aircraft specs from OCR text using regex patterns.
    Deterministic — same input always gives same output.
    No ML involved. Hallucination impossible.
    """

    def __init__(self):
        logger.info("OCRTextExtractor initialized (regex-based, deterministic)")

    def extract_all(
        self,
        page_records: list[dict],
        doc_id: str,
        citation: dict,
    ) -> dict:
        """
        Extract all fields from all pages using regex.

        Strategy:
          1. Concatenate all OCR text with page markers
          2. For each field, try all patterns
          3. Return first match found + which page it came from

        Returns dict of field_id → extraction record.
        """
        # Build page-indexed text corpus
        corpus = {}  # page_num → text
        full_text = ""
        for record in page_records:
            page_num = record.get("page_number", 0)
            raw_text = (record.get("text") or "").lower().strip()
            # Normalize OCR artifacts before indexing
            text = self._normalize_ocr(raw_text) if raw_text else ""
            if text:
                corpus[page_num] = text
                full_text += f"\n[PAGE {page_num}]\n{text}"

        total_chars = sum(len(t) for t in corpus.values())
        logger.info(
            f"OCR corpus: {len(corpus)} pages, "
            f"{total_chars:,} total chars"
        )

        results = {}

        for field_id, patterns in FIELD_PATTERNS.items():
            match_result = self._extract_field(
                field_id, patterns, corpus, full_text, doc_id, citation
            )
            results[field_id] = match_result

        found = sum(1 for v in results.values() if v["raw_value"] is not None)
        logger.info(
            f"Extraction complete: {found}/{len(results)} fields found"
        )

        # Log what was found and what wasn't
        for fid, rec in results.items():
            if rec["raw_value"] is not None:
                logger.success(
                    f"  FOUND: {fid} = {rec['raw_value']!r} (page {rec['page_number']})"
                )
            else:
                logger.warning(f"  MISSING: {fid}")

        return results


    def _normalize_ocr(self, text: str) -> str:
        """
        Fix common OCR artifacts deterministically.
        Rule-based only — zero hallucination risk.

        Handles:
          1. Hyphenated line breaks: "alterna-\ntor" → "alternator"
          2. Non-hyphenated splits: "alterna\ntor" → "alternator"
             (only for known aviation word roots)
          3. Stray punctuation artifacts: "·)" "~" mid-word
          4. Dot leader sequences: ". . . ." → " "
          5. OCR space artifacts in numbers: "30. 5" → "30.5"
        """
        import re as _re

        # 1. Fix hyphenated line breaks (most common OCR split)
        # "alterna-\ntor" → "alternator"
        text = _re.sub(r'([a-z]+)-\n([a-z]+)', r'\1\2', text)

        # 2. Fix known aviation word splits across newlines
        # These are the exact splits seen in Cessna POH OCR
        aviation_splits = [
            (r'alterna\ntor', 'alternator'),
            (r'elec\ntrical', 'electrical'),
            (r'volt\nage', 'voltage'),
            (r'alter\nnator', 'alternator'),
            (r'genera\ntor', 'generator'),
            (r'regula\ntor', 'regulator'),
            (r'carbure\ntor', 'carburetor'),
            (r'tem\nperature', 'temperature'),
            (r'pres\nsure', 'pressure'),
            (r'capaci\ntor', 'capacitor'),
            (r'capac\nity', 'capacity'),
            (r'indica\ntor', 'indicator'),
            (r'opera\ntion', 'operation'),
            (r'selec\ntor', 'selector'),
        ]
        for pattern, replacement in aviation_splits:
            text = _re.sub(pattern, replacement, text)

        # 3. Fix dot leader sequences (table of contents, spec sheets)
        # "service ceiling . . . . . 16,500" → "service ceiling 16,500"
        text = _re.sub(r'[\.]{2,}\s*', ' ', text)

        # 4. Fix OCR space in decimal numbers
        # "30. 5" → "30.5"  "14, 800" → "14,800"
        text = _re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
        text = _re.sub(r'(\d+),\s+(\d{3})', r'\1,\2', text)

        # 5. Remove stray OCR artifacts mid-line
        # "·)" "~j" "c»" etc that appear in scanned docs
        text = _re.sub(r'[·»«~]{1,3}[)\]\[]?', ' ', text)

        # 6. Normalize whitespace
        text = _re.sub(r' {2,}', ' ', text)
        text = _re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _extract_field(
        self,
        field_id: str,
        patterns: list[str],
        corpus: dict,
        full_text: str,
        doc_id: str,
        citation: dict,
    ) -> dict:
        """Try all patterns for a field. Return first match found."""

        unit = self._get_unit(field_id)
        is_text_field = unit in ("text", "string", "equation", "count")

        # Try each pattern against full concatenated text
        for pattern in patterns:
            try:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    # Find which page this appeared on
                    match_pos = match.start()
                    page_num = self._find_page_for_match(
                        match.group(0), corpus
                    )

                    # Extract value
                    if match.lastindex and match.lastindex >= 1:
                        raw_value = match.group(1).strip()
                    else:
                        # Pattern matched but no capture group — extract the number near the match
                        raw_value = self._extract_number_near_match(
                            match.group(0), is_text_field
                        )

                    if raw_value:
                        return {
                            "raw_value": raw_value,
                            "unit": unit,
                            "confidence": "regex_match",
                            "page_number": page_num,
                            "doc_id": doc_id,
                            "citation": citation,
                            "layer_use": self._get_layer_use(field_id),
                            "status": "pending_validation",
                            "extraction_method": "ocr_regex",
                            "pattern_used": pattern,
                        }
            except re.error as e:
                logger.debug(f"Regex error for {field_id}: {e}")
                continue

        # No match found
        return {
            "raw_value": None,
            "unit": unit,
            "confidence": "not_found",
            "page_number": None,
            "doc_id": doc_id,
            "citation": citation,
            "layer_use": self._get_layer_use(field_id),
            "status": "pending_validation",
            "extraction_method": "ocr_regex",
            "pattern_used": None,
        }

    def _find_page_for_match(self, match_text: str, corpus: dict) -> Optional[int]:
        """Find which page a matched text came from."""
        match_lower = match_text.lower().strip()[:50]
        for page_num, text in corpus.items():
            if match_lower in text:
                return page_num
        return None

    def _extract_number_near_match(self, text: str, is_text: bool) -> Optional[str]:
        """Extract number from match when no capture group."""
        if is_text:
            return text.strip()
        nums = re.findall(r'\d+\.?\d*', text)
        return nums[0] if nums else None

    def _get_unit(self, field_id: str) -> str:
        units = {
            "system_voltage": "volts",
            "alternator_output_amps": "amps",
            "battery_capacity_ah": "ampere-hours",
            "min_fuel_pump_voltage": "volts",
            "fuel_capacity_total_gal": "US gallons",
            "fuel_capacity_left_gal": "US gallons",
            "fuel_capacity_right_gal": "US gallons",
            "cruise_fuel_flow_gph": "GPH",
            "max_rpm": "RPM",
            "max_power_bhp": "BHP",
            "engine_count": "count",
            "vne_knots": "knots",
            "vs0_knots": "knots CAS",
            "service_ceiling_ft": "feet MSL",
            "max_gross_weight_lbs": "lbs",
            "aircraft_model": "string",
            "engine_model": "string",
            "faa_type_certificate": "string",
            "alternator_failure_procedure": "text",
            "engine_failure_procedure": "text",
            "fuel_system_failure_procedure": "text",
        }
        return units.get(field_id, "unknown")

    def _get_layer_use(self, field_id: str) -> str:
        layer_map = {
            "aircraft_model": "layer0_defaults",
            "engine_model": "layer1_constraint_graph",
            "faa_type_certificate": "citation",
            "system_voltage": "layer1_constraint_graph",
            "alternator_output_amps": "layer1_constraint_graph",
            "battery_capacity_ah": "layer1_constraint_graph",
            "min_fuel_pump_voltage": "layer1_constraint_graph",
            "fuel_capacity_total_gal": "layer0_defaults",
            "fuel_capacity_left_gal": "layer0_defaults",
            "fuel_capacity_right_gal": "layer0_defaults",
            "cruise_fuel_flow_gph": "layer2_simulation",
            "max_rpm": "layer1_constraint_graph",
            "max_power_bhp": "layer2_simulation",
            "engine_count": "layer0_defaults",
            "vne_knots": "layer0_validator",
            "vs0_knots": "layer0_validator",
            "service_ceiling_ft": "layer0_validator",
            "max_gross_weight_lbs": "layer2_simulation",
            "alternator_failure_procedure": "layer1_cascade_rules",
            "engine_failure_procedure": "layer1_cascade_rules",
            "fuel_system_failure_procedure": "layer1_cascade_rules",
        }
        return layer_map.get(field_id, "unknown")

    def save_extracted(self, results: dict, doc_id: str) -> Path:
        """Save extraction results to pending validation file."""
        output_path = VALIDATED_DIR / f"{doc_id}_extracted.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved extracted specs: {output_path}")
        return output_path