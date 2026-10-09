"""
rag_pipeline.py — BM25 Retrieval-Augmented Generation for KisanAI
==================================================================
Loads all agricultural JSON datasets from Dataset_RAG_Ready/ at startup,
builds a BM25 keyword index, and exposes retrieve_context() which returns
the most relevant text chunks for a given farmer query.

Supported datasets:
  - crop_disease_rag_ready.json      (disease symptoms + treatment)
  - fertilizer_dataset_rag_ready.json (fertilizer recommendations)
  - crop_dataset_rag_ready.json      (NPK, temp, humidity, pH, rainfall)
  - crop_yield_dataset_rag_ready.json (yield by country, temp, rainfall)
  - soil_dataset_rag_ready.json      (soil groups — NaN rows filtered out)
  - rice_dataset.json                (seasons + varieties — nested, flattened here)
  - weather_dataset_rag_ready.json   (SKIPPED — live API is used instead)

Design decisions:
  - BM25 (rank_bm25) is used instead of dense embeddings so no GPU or
    large language model is needed for retrieval.
  - crop_yield records are de-duplicated by (crop, country) to reduce
    the 225K-row flood of near-identical entries to a useful summary set.
  - soil records with all-NaN text are discarded at load time.
  - rice_dataset uses a different nested schema; it is flattened into
    plain sentences before indexing.
"""

import os
import json
import re
import string
import time
from typing import List, Optional

# rank_bm25 is the only new dependency
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is not installed. Run: pip install rank-bm25"
    )

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_RAG_Ready")

DATASET_FILES = {
    "disease"    : "crop_disease_rag_ready.json",
    "fertilizer" : "fertilizer_dataset_rag_ready.json",
    "crop"       : "crop_dataset_rag_ready.json",
    "crop_yield" : "crop_yield_dataset_rag_ready.json",
    "soil"       : "soil_dataset_rag_ready.json",
    "rice"       : "rice_dataset.json",
    # weather_dataset_rag_ready.json is intentionally excluded (551 MB,
    # live OpenWeatherMap API is already used for weather context)
}

# Maximum records to load per large dataset (prevents RAM exhaustion)
# Set to None to load all records (only do this on a high-RAM machine)
MAX_RECORDS_PER_TYPE = {
    "disease"    : None,   # tiny (3 records)
    "fertilizer" : None,   # small (~290 records)
    "crop"       : 5000,   # cap at 5 K — already covers all crop types
    "crop_yield" : None,   # de-duplicated below, so effective count is small
    "soil"       : 3000,   # 236K rows but 90%+ are NaN — take first 3K valid
    "rice"       : None,   # small (~100 flattened chunks)
}

# ─── Helpers ────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "in", "of", "and", "or", "to", "for",
    "with", "on", "at", "by", "from", "has", "have", "be", "are",
    "was", "were", "it", "its", "that", "this", "as", "per",
    "approximately", "about", "around", "millimeters", "degrees",
    "percent", "hectograms", "hectare"
}

def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split, drop stop-words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _is_nan_text(text: str) -> bool:
    """Return True if the text contains nothing informative (all nan values)."""
    clean = re.sub(r"\bnan\b", "", text, flags=re.IGNORECASE).strip()
    # After removing 'nan', if everything meaningful is gone → useless
    informative = re.sub(r"[^a-zA-Z]", "", clean).strip()
    return len(informative) < 10  # less than 10 actual letters → skip


# ─── Loaders ────────────────────────────────────────────────────────

def _load_standard_json(filepath: str, dtype: str, max_records: Optional[int]) -> List[dict]:
    """Load a standard [{text, metadata}, ...] JSON file."""
    docs = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️ RAG: Could not load {filepath}: {e}")
        return docs

    count = 0
    for record in data:
        text = record.get("text", "").strip()
        if not text:
            continue
        if dtype == "soil" and _is_nan_text(text):
            continue
        docs.append({"text": text, "type": dtype})
        count += 1
        if max_records and count >= max_records:
            break
    return docs


def _dedup_crop_yield(docs: List[dict]) -> List[dict]:
    """
    crop_yield has 225K rows that are near-identical per (crop, country).
    Keep only the first occurrence per (crop, country) pair to avoid
    flooding the index with redundant entries.
    """
    seen = set()
    deduped = []
    for doc in docs:
        # Extract crop and country from the text heuristically
        text = doc["text"]
        m_crop    = re.search(r"the crop ([A-Za-z ,]+) has", text)
        m_country = re.search(r"In ([A-Za-z ]+),", text)
        crop    = m_crop.group(1).strip()    if m_crop    else ""
        country = m_country.group(1).strip() if m_country else ""
        key = (crop.lower(), country.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def _load_rice_dataset(filepath: str) -> List[dict]:
    """
    rice_dataset.json has a nested schema with season_data and variety_data.
    Flatten each entry into a readable sentence for indexing.
    """
    docs = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️ RAG: Could not load rice dataset: {e}")
        return docs

    # Season data
    for item in data.get("season_data", []):
        season   = item.get("season_name", "")
        month    = item.get("sowing_month", "")
        duration = item.get("duration_days", "")
        districts_list = item.get("districts", [])
        districts = ", ".join([d for d in districts_list if d])
        if season and month:
            text = (
                f"Rice {season} season is sown in {month} with a duration of "
                f"{duration} days. Suitable districts include: {districts}."
            )
            docs.append({"text": text, "type": "rice_season"})

    # Variety data
    for item in data.get("variety_data", []):
        zone     = item.get("zone") or ""
        district = item.get("district_group", "")
        season   = item.get("season", "")
        month    = item.get("month", "")
        varieties_list = item.get("varieties", [])
        varieties = ", ".join(varieties_list)
        if season and varieties:
            location = f"{zone} — {district}".strip(" —")
            text = (
                f"In {location}, recommended rice varieties for {season} season "
                f"(sown in {month}) are: {varieties}."
            )
            docs.append({"text": text, "type": "rice_variety"})

    return docs


# ─── Index Builder ──────────────────────────────────────────────────

class RAGPipeline:
    """
    Singleton RAG pipeline that builds a BM25 index at startup and
    answers retrieval queries via retrieve_context().
    """

    def __init__(self):
        self._documents: List[dict] = []
        self._tokenized: List[List[str]] = []
        self._index: Optional[BM25Okapi] = None
        self._ready = False

    def build(self):
        """Load all datasets and build the BM25 index. Call once at startup."""
        t0 = time.time()
        print("🔍 RAG: Building BM25 index from agricultural datasets...")

        all_docs: List[dict] = []

        for dtype, filename in DATASET_FILES.items():
            filepath = os.path.join(DATASET_DIR, filename)
            if not os.path.exists(filepath):
                print(f"  ⚠️ RAG: File not found — {filename}")
                continue

            max_rec = MAX_RECORDS_PER_TYPE.get(dtype)

            if dtype == "rice":
                docs = _load_rice_dataset(filepath)
            else:
                docs = _load_standard_json(filepath, dtype, max_rec)

            # Extra de-duplication for crop_yield (225K → unique per crop/country)
            if dtype == "crop_yield":
                before = len(docs)
                docs = _dedup_crop_yield(docs)
                print(f"  📦 RAG: {dtype:12s} → {before:>6,} raw → {len(docs):>5,} after de-dup")
            else:
                print(f"  📦 RAG: {dtype:12s} → {len(docs):>5,} records loaded")

            all_docs.extend(docs)

        if not all_docs:
            print("  ❌ RAG: No documents loaded — RAG context will be unavailable.")
            return

        # Tokenize
        self._documents = all_docs
        self._tokenized = [_tokenize(doc["text"]) for doc in all_docs]

        # Build BM25
        self._index = BM25Okapi(self._tokenized)
        self._ready = True

        elapsed = time.time() - t0
        print(f"✅ RAG: Index built with {len(all_docs):,} documents in {elapsed:.1f}s")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve the top_k most relevant documents for the given query and
        return them as a formatted context string ready for the LLM prompt.

        Returns an empty string if the index is not ready or query is empty.
        """
        if not self._ready or not query.strip():
            return ""

        tokens = _tokenize(query)
        if not tokens:
            return ""

        scores = self._index.get_scores(tokens)

        # Get top_k indices sorted by score descending
        import heapq
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

        # Filter out zero-score results (no keyword overlap at all)
        results = [
            self._documents[i]["text"]
            for i in top_indices
            if scores[i] > 0
        ]

        if not results:
            return ""

        context_lines = "\n".join(f"- {r}" for r in results)
        return (
            "AGRICULTURAL KNOWLEDGE BASE (retrieved facts — use these to ground your answer):\n"
            + context_lines
        )


# ─── Singleton Instance ─────────────────────────────────────────────

_pipeline = RAGPipeline()


def build_rag_index():
    """Call this once at app startup to build the BM25 index."""
    _pipeline.build()


def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Public API — retrieve relevant agricultural context for a query.

    Args:
        query:  The farmer's question (any language — BM25 works on
                query tokens, which usually contain English crop/disease names
                even in Hindi/Marathi transliterated text).
        top_k:  Number of results to return (default 5).

    Returns:
        A formatted multi-line string for injection into the LLM system prompt,
        or an empty string if no relevant data found.
    """
    return _pipeline.retrieve_context(query, top_k=top_k)
