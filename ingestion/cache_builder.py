"""
Ingestion: Cache Builder + ChromaDB Vector Store
Builds two outputs from validated specs:

  1. physics_cache.json — frozen numeric values for Layer 2 simulator
     Every value: {value, unit, source_doc, page, citation, validated_by}
     Read-only at runtime. Never modified after build.

  2. ChromaDB vector store — text passages for Layer 1 RAG queries
     Stores emergency procedures, system descriptions, cascade logic text
     Queried during Layer 1 constraint graph construction

Source: Architectural decision established during data ingestion planning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from ingestion.config import CACHE_DIR, CHROMADB_DIR as CHROMA_DIR, VALIDATED_DIR


class CacheBuilder:
    """
    Assembles physics_cache.json from all validated_specs files.
    Called once after all documents are validated.
    Output is frozen — never modified at runtime.
    """

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def build(self, validated_dir: Path = VALIDATED_DIR) -> Path:
        """
        Scan all *_validated.json files and merge into physics_cache.json.
        Only includes accepted + corrected values (not rejected or null).
        """
        cache = {
            "_metadata": {
                "built_at": datetime.utcnow().isoformat(),
                "description": "FailureScope physics cache — build-time validated values only",
                "status": "FROZEN — do not modify at runtime",
                "source": "Human-validated extraction from official aircraft documents",
            },
            "values": {},
        }

        validated_files = list(validated_dir.glob("*_validated.json"))
        if not validated_files:
            logger.warning(f"No validated spec files found in {validated_dir}")
            return None

        logger.info(f"Building cache from {len(validated_files)} validated files")

        for vf in validated_files:
            with open(vf) as f:
                specs = json.load(f)

            for field_id, record in specs.items():
                status = record.get("status")
                if status not in ("accepted", "corrected"):
                    continue
                if record.get("final_value") is None:
                    continue

                # Convert value to appropriate Python type
                raw = record["final_value"]
                converted = self._convert_value(raw, record.get("unit", ""))

                cache["values"][field_id] = {
                    "value": converted,
                    "raw_string": str(raw),
                    "unit": record.get("unit"),
                    "doc_id": record.get("doc_id"),
                    "page_number": record.get("page_number"),
                    "citation": record.get("citation"),
                    "layer_use": record.get("layer_use"),
                    "validated_by": record.get("validated_by"),
                    "validated_at": record.get("validated_at"),
                    "status": status,
                }

        output_path = CACHE_DIR / "physics_cache.json"
        with open(output_path, "w") as f:
            json.dump(cache, f, indent=2)

        logger.success(
            f"physics_cache.json built: {len(cache['values'])} values → {output_path}"
        )
        self._print_summary(cache)
        return output_path

    def _convert_value(self, raw: str, unit: str):
        """Convert string value to numeric type where appropriate."""
        if unit in ("text", "string", "equation"):
            return str(raw)
        try:
            if "." in str(raw):
                return float(raw)
            return int(raw)
        except (ValueError, TypeError):
            return str(raw)

    def _print_summary(self, cache: dict):
        print(f"\n{'='*60}")
        print("physics_cache.json — Build Summary")
        print(f"{'='*60}")
        by_layer = {}
        for fid, val in cache["values"].items():
            layer = val.get("layer_use", "unknown")
            by_layer.setdefault(layer, []).append(fid)
        for layer, fields in sorted(by_layer.items()):
            print(f"  {layer}: {len(fields)} values")
        print(f"{'='*60}")
        print(f"  Total: {len(cache['values'])} validated values")
        print(f"  Status: FROZEN — ready for Layer 1 and Layer 2")
        print(f"{'='*60}\n")


class VectorStoreBuilder:
    """
    Builds ChromaDB vector store from OCR page text.
    Stores text passages with citation metadata for Layer 1 RAG queries.
    """

    def __init__(self, chroma_dir: Path = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy init ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
                self._collection = self._client.get_or_create_collection(
                    name="failurescope_docs",
                    metadata={"description": "FailureScope document chunks with citations"},
                )
                logger.info(f"ChromaDB initialized at {self.chroma_dir}")
            except ImportError:
                raise RuntimeError(
                    "chromadb not installed. Run: pip install chromadb"
                )
        return self._client, self._collection

    def ingest_pages(self, page_records: list[dict], chunk_size: int = 500):
        """
        Ingest OCR text from page records into ChromaDB.

        Each chunk stored with full citation metadata.
        Chunk size: 500 chars — balances context vs retrieval precision.
        """
        _, collection = self._get_client()

        documents = []
        metadatas = []
        ids = []

        for record in page_records:
            text = record.get("text", "").strip()
            if not text or len(text) < 50:
                continue

            citation = record.get("citation", {})
            doc_id = record.get("doc_id", "")
            page_num = record.get("page_number", 0)

            # Split into chunks
            chunks = self._chunk_text(text, chunk_size)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"

                documents.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "source_document": citation.get("document", ""),
                    "manufacturer": citation.get("manufacturer", citation.get("publisher", "")),
                    "year": str(citation.get("year", "")),
                    "faa_approval": citation.get("faa_approval", ""),
                    "doc_type": record.get("doc_type", ""),
                })
                ids.append(chunk_id)

        if not documents:
            logger.warning("No text content to ingest into ChromaDB")
            return 0

        # Upsert in batches of 100
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
            total_inserted += len(documents[i:i+batch_size])

        logger.success(f"ChromaDB: inserted {total_inserted} chunks from {len(page_records)} pages")
        return total_inserted

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Query ChromaDB for relevant passages.
        Returns list of {text, citation, page_number, doc_id}

        Used by Layer 1 to retrieve cascade logic and system descriptions.
        """
        _, collection = self._get_client()

        where = {"doc_id": filter_doc_id} if filter_doc_id else None

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

        passages = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            passages.append({
                "text": doc,
                "doc_id": meta.get("doc_id"),
                "page_number": meta.get("page_number"),
                "source_document": meta.get("source_document"),
                "manufacturer": meta.get("manufacturer"),
                "year": meta.get("year"),
            })

        return passages

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into overlapping chunks for better retrieval."""
        words = text.split()
        chunks = []
        step = chunk_size // 2  # 50% overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 20:
                break
            chunks.append(" ".join(chunk_words))

        return chunks if chunks else [text]

    def stats(self):
        """Print ChromaDB collection stats."""
        _, collection = self._get_client()
        count = collection.count()
        print(f"\nChromaDB Stats:")
        print(f"  Collection: failurescope_docs")
        print(f"  Total chunks: {count}")
        print(f"  Path: {self.chroma_dir}\n")