# KisanAI — Agricultural Voice Assistant for Indian Farmers

> **B.Tech Sem 6 — Generative AI | CA Project**
> Authors: Sujay Kamuni, Harsh Patil, Rachit Patil, Chirag Patil

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What It Does](#2-what-it-does)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [RAG Implementation (Deep Dive)](#5-rag-implementation-deep-dive)
   - [Why RAG?](#51-why-rag)
   - [Knowledge Base / Datasets](#52-knowledge-base--datasets)
   - [BM25 Indexing](#53-bm25-indexing--how-it-works)
   - [Tokenization and Preprocessing](#54-tokenization-and-preprocessing)
   - [Retrieval at Query Time](#55-retrieval-at-query-time)
   - [Context Injection into LLM Prompt](#56-context-injection-into-llm-prompt)
   - [Design Decisions and Tradeoffs](#57-design-decisions-and-tradeoffs)
6. [End-to-End Pipeline](#6-end-to-end-pipeline)
7. [Multi-Language Support](#7-multi-language-support)
8. [Weather Integration](#8-weather-integration)
9. [Conversation Memory](#9-conversation-memory)
10. [API Endpoints](#10-api-endpoints)
11. [Directory Structure](#11-directory-structure)
12. [Configuration Reference](#12-configuration-reference)
13. [Setup and Running](#13-setup-and-running)
14. [Performance Characteristics](#14-performance-characteristics)
15. [Error Handling](#15-error-handling)

---

## 1. Project Overview

**KisanAI** is a conversational voice assistant specifically designed for **Indian farmers**. It bridges the gap between complex agricultural science and ground-level farming decisions by letting farmers speak in their native language (English, Hindi, or Marathi) and receive spoken, expert advice — powered by a locally running Large Language Model (LLM) *grounded* by a domain-specific agricultural knowledge base.

The core innovation is the use of **Retrieval-Augmented Generation (RAG)** to ensure the LLM's responses are factually accurate and rooted in real agricultural data, rather than relying solely on the model's trained (and potentially outdated) knowledge.

---

## 2. What It Does

| Capability | How |
|---|---|
| Voice Input | Browser `MediaRecorder` API captures speech as WebM audio |
| Language Detection | Auto-detects English / Hindi / Marathi using Unicode script analysis |
| RAG Retrieval | BM25 keyword search over 6 agricultural datasets — top-5 relevant facts |
| Live Weather | Real-time weather injected into LLM context via OpenWeatherMap API |
| LLM Inference | Locally-running **Llama 3** via Ollama answers in the farmer's language |
| Voice Response | Google Text-to-Speech converts the answer to MP3 and plays it back |
| Mobile Support | Android WebView wrapper for native app experience |
| Public Access | ngrok tunnel exposes local server over HTTPS for remote access |

---

## 3. System Architecture

```
+--------------------------------------------------------------+
|              FRONTEND (Browser / Android WebView)            |
|  index.html / mobile.html                                    |
|  - Language selector (Auto / English / Hindi / Marathi)      |
|  - Record button -> MediaRecorder -> WebM audio blob         |
|  - POST /transcribe  ->  POST /ask  ->  Play response.mp3   |
+-------------------------+------------------------------------+
                          | HTTP  (local or ngrok HTTPS tunnel)
+-------------------------v------------------------------------+
|              FLASK BACKEND  (app.py, port 5000)              |
|                                                              |
|  /transcribe         /ask              /geolocate /weather   |
|  Audio -> STT        RAG + LLM + TTS  IP Geo / OWM API      |
|                                                              |
|  +----------------------------------------------------------+|
|  |               CORE PROCESSING PIPELINE                  ||
|  |  1. FFmpeg:  WebM -> 16kHz Mono WAV                     ||
|  |  2. Google STT: WAV -> text  (en-IN / hi-IN / mr-IN)    ||
|  |  3. Script detection: pick correct language              ||
|  |  4. BM25 RAG: query -> top-5 agricultural facts         ||
|  |  5. Weather injection: live conditions from OWM API     ||
|  |  6. Llama 3 (Ollama): generate grounded answer          ||
|  |  7. gTTS: answer text -> response.mp3                   ||
|  +----------------------------------------------------------+|
+--------------------------------------------------------------+
         |                              |
+--------v--------+          +----------v----------------------+
|  Ollama (LLM)   |          |  RAG Knowledge Base             |
|  llama3, CPU    |          |  Dataset_RAG_Ready/ (BM25)      |
|  temp=0.3       |          |  6 JSON datasets, ~9K docs      |
+-----------------+          +---------------------------------+
```

---

## 4. Technology Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Web Framework | **Flask** | HTTP server, REST API routing |
| LLM Backend | **Ollama + Llama 3** | Local LLM inference (CPU-only) |
| Speech-to-Text | **SpeechRecognition** | Google Speech API wrapper |
| Audio Conversion | **imageio_ffmpeg** | Bundled FFmpeg (WebM to WAV) |
| Text-to-Speech | **gTTS** | MP3 synthesis in 3 languages |
| **RAG Retrieval** | **rank-bm25 (BM25Okapi)** | Keyword-based document retrieval |
| Weather API | **OpenWeatherMap** | Real-time weather data |
| Geolocation | **ip-api.com / ipinfo.io** | Server-side IP geolocation |
| Public Tunneling | **pyngrok / ngrok** | HTTPS tunnel for remote access |
| HTTP Client | **requests** | API calls to external services |
| Android Wrapper | **Android WebView** | Native app wrapping mobile.html |

**Python Version:** 3.10+
**OS Tested On:** Windows 11

---

## 5. RAG Implementation (Deep Dive)

> This is the most critical and technically interesting component of KisanAI. RAG stands for **Retrieval-Augmented Generation** — a technique where an LLM's response is *grounded* by retrieving relevant facts from a local knowledge base before generating an answer.

### 5.1 Why RAG?

Without RAG, the LLM (Llama 3) answers from its **training knowledge only**. This has serious problems for agricultural applications:

- Training data may have **outdated** crop recommendations
- LLMs can **hallucinate** specific values like NPK ratios, pH levels, or yield figures
- Domain-specific Indian data (specific districts, regional rice varieties, Indian soil types) is **underrepresented** in general-purpose LLM training

RAG solves this by:
1. **Retrieving** relevant, verified facts from a locally-indexed knowledge base **before** calling the LLM
2. **Injecting** those facts directly into the LLM's system prompt as *ground truth*
3. **Instructing** the LLM to treat retrieved facts as authoritative and not contradict them

```
Without RAG:   Farmer Query -> LLM -> Answer (possibly hallucinated)

With RAG:      Farmer Query
                   |
                   +---> BM25 Index -> Top-5 Relevant Facts
                   |                        |
                   +---> LLM (with facts injected in prompt) -> Grounded Answer
```

---

### 5.2 Knowledge Base / Datasets

All datasets live in `Dataset_RAG_Ready/` and are pre-processed into a standard `[{"text": "...", ...}, ...]` JSON format before being indexed.

| Dataset File | Size | Records (after processing) | Content |
|---|---|---|---|
| `crop_disease_rag_ready.json` | ~1 KB | 3 | Disease name, symptoms, treatment per crop |
| `fertilizer_dataset_rag_ready.json` | ~28 KB | ~290 | Soil type, crop, recommended fertilizer, NPK ratio |
| `crop_dataset_rag_ready.json` | ~777 KB | 5,000 (capped) | N, P, K requirements; temperature, humidity, pH, rainfall per crop |
| `crop_yield_dataset_rag_ready.json` | ~10.1 MB | ~225K raw -> **de-duplicated** to unique (crop, country) pairs | Yield (hg/ha) by crop and country |
| `soil_dataset_rag_ready.json` | ~7.9 MB | 3,000 (capped; NaN rows filtered) | Soil group classifications |
| `rice_dataset.json` | ~44 KB | ~100 (flattened from nested schema) | Rice seasons, sowing months, duration, district-specific variety recommendations |
| `weather_dataset_rag_ready.json` | **~526 MB** | **SKIPPED** | Not indexed — live OpenWeatherMap API used instead |

**Total BM25 index size:** approximately 9,000–10,000 documents

#### Special Data Loading Cases

**Crop Yield De-duplication**

The `crop_yield` dataset contains 225,000 rows that are near-identical for the same `(crop, country)` pair. Loading all 225K would flood the index with redundant entries and degrade retrieval quality. A de-duplication step (`_dedup_crop_yield`) keeps only the **first occurrence per unique (crop, country) pair** using regex extraction:

```python
def _dedup_crop_yield(docs):
    seen = set()
    deduped = []
    for doc in docs:
        text = doc["text"]
        m_crop    = re.search(r"the crop ([A-Za-z ,]+) has", text)
        m_country = re.search(r"In ([A-Za-z ]+),", text)
        key = (crop.lower(), country.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped
```

**Rice Dataset Flattening**

The `rice_dataset.json` uses a nested schema with `season_data` and `variety_data` arrays. A custom loader (`_load_rice_dataset`) flattens these into natural-language sentences before indexing:

```python
# Season data -> sentence
"Rice Kharif season is sown in June with a duration of 120 days.
 Suitable districts include: Pune, Nashik, Kolhapur."

# Variety data -> sentence
"In Konkan -- Ratnagiri, recommended rice varieties for Kharif season
 (sown in June) are: Karjat-3, Ratnagiri-24, Sahyadri-2."
```

**NaN Filtering for Soil Data**

The soil dataset has many rows where all fields contain `NaN`. These are detected by `_is_nan_text()` and discarded before indexing:

```python
def _is_nan_text(text):
    # Remove all occurrences of "nan", then check remaining meaningful letters
    clean = re.sub(r"\bnan\b", "", text, flags=re.IGNORECASE).strip()
    informative = re.sub(r"[^a-zA-Z]", "", clean).strip()
    return len(informative) < 10  # fewer than 10 real letters -> discard
```

---

### 5.3 BM25 Indexing — How It Works

The project uses **BM25Okapi** from the `rank-bm25` library. BM25 (Best Matching 25) is a probabilistic keyword-based ranking algorithm — essentially a well-tuned version of TF-IDF.

**Index build sequence** (called once at Flask startup via `build_rag_index()`):

```
1. Load all dataset files -> List of {"text": "...", "type": "..."} dicts
2. Tokenize each document text: _tokenize(text) -> list of keyword tokens
3. BM25Okapi(tokenized_corpus) -> builds inverted index with BM25 weights
4. Mark pipeline as _ready = True
```

**BM25 Scoring Formula**

For a query with tokens `q1, q2, ..., qn` and document `d`:

```
Score(d, q) = SUM of [ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1 * (1 - b + b * |d|/avgdl)) ]
```

Where:
- `f(qi, d)` = frequency of token `qi` in document `d` (Term Frequency)
- `IDF(qi)` = inverse document frequency — rare tokens get higher weight
- `|d|` = document length; `avgdl` = average document length across corpus
- `k1 = 1.5`, `b = 0.75` (BM25Okapi defaults)
- **Length normalization** prevents longer documents from unfairly dominating results

**Why BM25 over dense embeddings (FAISS + Sentence Transformers)?**

| Factor | BM25 | Dense Embeddings |
|---|---|---|
| GPU required | No | Recommended |
| RAM usage | Low (~50-100 MB) | High (embedding model + vectors) |
| Startup time | Fast (~5-10s) | Slow (model load + vectorization) |
| Query speed | < 50ms | < 100ms (GPU) / slow (CPU) |
| Semantic understanding | Limited (keyword match) | Good (meaning-based) |
| Works offline | Fully | Needs model download |
| Suitable for this project | Yes — agricultural terms are specific keywords | Overkill for this domain |

For agricultural queries like *"NPK for wheat"*, *"rice disease treatment"*, or *"fertilizer for black soil"*, **keyword matching is highly effective** because the queries naturally contain the exact domain keywords present in the knowledge base.

---

### 5.4 Tokenization and Preprocessing

Both documents (at index time) and queries (at retrieval time) go through the same tokenizer to ensure consistency:

```python
_STOP_WORDS = {
    "a", "an", "the", "is", "in", "of", "and", "or", "to", "for",
    "with", "on", "at", "by", "from", "has", "have", "be", "are",
    "was", "were", "it", "its", "that", "this", "as", "per",
    "approximately", "about", "around", "millimeters", "degrees",
    "percent", "hectograms", "hectare"
}

def _tokenize(text: str) -> List[str]:
    text = text.lower()                                               # 1. Lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # 2. Remove punctuation
    tokens = text.split()                                             # 3. Split on whitespace
    return [t for t in tokens if t not in _STOP_WORDS                # 4. Drop stop-words
                              and len(t) > 1]                         # 5. Drop single chars
```

**Example tokenization:**

```
Input:  "What NPK ratio is best for wheat crop?"
Output: ["npk", "ratio", "best", "wheat", "crop"]
```

Note: domain units like `"millimeters"`, `"degrees"`, `"hectare"` are in the stop-word list because they appear in almost every document and carry no discriminative value for retrieval.

---

### 5.5 Retrieval at Query Time

When a farmer's query arrives at the `/ask` endpoint, retrieval happens in `retrieve_context()`:

```python
def retrieve_context(query: str, top_k: int = 5) -> str:
    tokens = _tokenize(query)
    if not tokens:
        return ""

    scores = self._index.get_scores(tokens)   # BM25 scores all ~9K documents

    # Select top-5 highest-scoring document indices
    top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

    # Filter: discard documents with zero keyword overlap
    results = [
        self._documents[i]["text"]
        for i in top_indices
        if scores[i] > 0
    ]
    return format_as_context_block(results)
```

**Key detail:** Documents with `score = 0` (zero keyword overlap with the query) are explicitly excluded. This prevents injecting unrelated facts when the knowledge base does not cover the query. In that case, `retrieve_context()` returns `""` and the LLM answers from its own training knowledge instead.

**Retrieval speed:** BM25 is an in-memory index — retrieval takes **< 50 milliseconds** regardless of corpus size.

---

### 5.6 Context Injection into LLM Prompt

The retrieved facts are formatted and injected directly into the **system prompt** sent to Ollama/Llama 3:

```
AGRICULTURAL KNOWLEDGE BASE (retrieved facts — use these to ground your answer):
- For the crop wheat, the recommended N is 60, P is 40, K is 30 kg/ha.
  Temperature range: 15-25 C. Rainfall: 50-75 mm.
- Wheat performs best in loam or clay-loam soils with pH 6.0-7.5.
- Recommended fertilizer for wheat on loamy soil: Urea + DAP at 120:60 kg/ha.
- In India, wheat average yield is approximately 3,200 kg/ha.
- Wheat is susceptible to rust disease; treat with Propiconazole fungicide.

INSTRUCTIONS for using the above knowledge base:
- Treat retrieved facts as ground truth for your answer.
- If a retrieved fact directly answers the query, use it.
- Do NOT contradict the retrieved facts.
- Do NOT mention "knowledge base" or "retrieved data" to the farmer — just answer naturally.
```

This block is assembled alongside other prompt sections:

```
[System Prompt]
  |-- Persona block           ("You are KisanAI, an agriculture expert...")
  |-- RAG Context block   <-- Retrieved agricultural facts  (THIS SECTION)
  |-- Weather block           (Live weather conditions + usage rules)
  |-- Language & Format       (Strict instruction to reply only in farmer's language)

[Conversation History]        (Last 10 exchanges — sliding window)

[Current User Query]
```

---

### 5.7 Design Decisions and Tradeoffs

| Decision | Rationale |
|---|---|
| **BM25 over dense embeddings** | No GPU required; agricultural keywords are specific enough for keyword matching to be highly effective |
| **BM25 over vector DB (FAISS/Chroma)** | No extra service to run; BM25 is purely in-memory Python |
| **top_k = 5** | Balances context richness vs. prompt length; 5 chunks approximately 400-600 tokens |
| **Zero-score filtering** | Avoids injecting irrelevant context when the KB does not cover the query |
| **Weather dataset SKIPPED (526 MB)** | Not indexed — live API provides more accurate, current data |
| **Crop yield de-duplication** | 225K near-identical rows reduced to unique (crop, country) pairs prevents index flooding |
| **Soil cap at 3,000** | 236K rows but 90%+ are NaN — first 3K valid rows are sufficient |
| **Crop cap at 5,000** | Already covers all unique crop types; additional rows are redundant |
| **Singleton pipeline** | Index built once at startup, shared across all requests — no per-request overhead |
| **LLM temperature = 0.3** | Low temperature gives deterministic, factual answers; avoids creative but wrong responses |

---

## 6. End-to-End Pipeline

A complete voice-to-voice interaction follows these 23 steps:

```
+-- FARMER SPEAKS ---------------------------------------------------+
|  1. Browser MediaRecorder captures voice -> WebM blob             |
|  2. POST /transcribe with audio + hint_lang                       |
|  3. Server validates: file present + size >= 1,000 bytes          |
|  4. FFmpeg: WebM -> 16kHz Mono WAV (via imageio_ffmpeg)           |
|  5. Google STT transcribes in all 3 locales sequentially          |
|  6. Unicode script analysis picks best language match             |
|  7. Response: {"text": "...", "language": "en/hi/mr"}             |
+-------------------------------------------------------------------+
+-- ENRICHMENT ------------------------------------------------------+
|  8.  GET /geolocate -> server resolves client IP -> lat/lon       |
|  9.  POST /ask with {query, language, lat, lon}                   |
|  10. fetch_weather(lat, lon) -> OpenWeatherMap (or 30-min cache)  |
+-------------------------------------------------------------------+
+-- RAG RETRIEVAL ---------------------------------------------------+
|  11. retrieve_context(query, top_k=5)                             |
|  12. BM25 tokenizes query -> scores all ~9K documents             |
|  13. Top-5 scoring docs (score > 0) -> formatted context block   |
+-------------------------------------------------------------------+
+-- LLM INFERENCE ---------------------------------------------------+
|  14. System prompt: persona + RAG facts + weather + lang rule     |
|  15. Append conversation history (last 10 exchanges)              |
|  16. ollama.chat(model="llama3", messages=[...])                  |
|  17. Answer stored in conversation_history                        |
+-------------------------------------------------------------------+
+-- TEXT-TO-SPEECH --------------------------------------------------+
|  18. Strip Markdown from LLM output                               |
|  19. gTTS synthesizes MP3 in farmer's detected language           |
|  20. Saved to Static/response.mp3                                 |
+-------------------------------------------------------------------+
+-- FARMER HEARS RESPONSE ------------------------------------------+
|  21. /ask returns JSON: {answer, audio_url, language, weather}    |
|  22. Browser renders text card + weather panel                    |
|  23. <audio> element auto-plays /static/response.mp3             |
+-------------------------------------------------------------------+
```

---

## 7. Multi-Language Support

KisanAI natively supports **three languages** across the entire pipeline:

| Language | Code | STT Locale | TTS Code | Detection Method |
|---|---|---|---|---|
| English | `en` | `en-IN` | `en` | Default; Latin script dominant |
| Hindi | `hi` | `hi-IN` | `hi` | Devanagari > 20% of chars + no Marathi keywords |
| Marathi | `mr` | `mr-IN` | `mr` | Devanagari > 20% + Marathi-specific words found |

**Marathi keywords** detected in `detect_language()` (Devanagari script in source):
```
aahe, nahi, ani, kara, mazya, tumchya, amchya,
pik, sheti, mala, kase, kay, kele, hote, aste
```

**Auto-detection flow:**
```
STT attempt 1: en-IN -> Latin script dominant?            -> return ("text", "en")
STT attempt 2: hi-IN -> Devanagari dominant, no Marathi?  -> return ("text", "hi")
STT attempt 3: mr-IN -> Marathi keywords found?           -> return ("text", "mr")
Fallback: return first non-empty result from any attempt
Last resort: return ("", "en")
```

The detected language propagates through the full pipeline: the LLM is instructed to reply **only** in that language, and gTTS synthesizes audio in the **same language**.

---

## 8. Weather Integration

Weather context makes KisanAI's advice genuinely situational rather than generic:

- **Source:** OpenWeatherMap `/data/2.5/weather` API
- **Geolocation:** Server-side IP geolocation via `ip-api.com` (fallback: `ipinfo.io`)
- **Cache TTL:** 30 minutes (`_weather_cache` dict with unix timestamps)
- **Unit normalization:** Kelvin to Celsius, m/s to km/h, metres to km

**Weather-aware advice the LLM is guided to give:**
- High humidity -> warn about fungal disease risk
- Rain today -> advise skipping irrigation
- High temperature -> suggest early morning watering

**Weather block injected into the system prompt:**
```
WEATHER DATA (already fetched — do NOT ask the farmer for location):
[CURRENT WEATHER — Pune, IN]: 28.4 C, feels 31.2 C, overcast clouds,
humidity 84%, wind 12.6 km/h

RULES for using weather:
- You already have the farmer's weather. NEVER ask them to share location or city.
- Proactively reference these conditions in your answer when relevant.
```

---

## 9. Conversation Memory

KisanAI maintains **conversation context** across turns within a session:

- `conversation_history` — global in-memory list of `{role, content}` dicts
- **Max history:** 10 exchanges (20 messages: 10 user + 10 assistant)
- **Sliding window:** When limit exceeded, oldest messages are dropped
- **Full history** sent to Ollama on every `/ask` request
- **Clear history:** `POST /clear_history` resets to empty list

> **Limitation:** Memory is server-scoped, not per-session. Multiple simultaneous users share one conversation history. This is acceptable for the intended single-farmer field use case.

---

## 10. API Endpoints

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/` | GET | — | `index.html` (desktop UI) |
| `/mobile` | GET | — | `mobile.html` (Android UI) |
| `/transcribe` | POST | Multipart: `audio` file + `hint_lang` | `{text, language}` |
| `/ask` | POST | JSON: `{query, language, lat, lon, city}` | `{answer, audio_url, language, weather}` |
| `/geolocate` | GET | — | `{lat, lon, city, region, country}` |
| `/weather` | GET/POST | `lat, lon` or `city` | `{weather object}` |
| `/clear_history` | POST | — | `{status}` |
| `/static/response.mp3` | GET | — | MP3 audio binary |

---

## 11. Directory Structure

```
Kisan_AI/
|-- app.py                          <- Main Flask application (600 lines)
|-- rag_pipeline.py                 <- BM25 RAG index builder and retriever (309 lines)
|-- run_ngrok.py                    <- ngrok tunnel launcher
|-- requirement.txt                 <- Python dependencies
|-- farmer_voice.wav                <- Sample voice recording (for testing)
|-- README.md                       <- This file
|
|-- Templates/
|   |-- index.html                  <- Desktop web UI (~41 KB)
|   +-- mobile.html                 <- Mobile-optimized UI for Android WebView
|
|-- Static/
|   +-- response.mp3                <- TTS audio output (overwritten each turn)
|
|-- Dataset_RAG_Ready/              <- Agricultural knowledge base (RAG corpus)
|   |-- crop_dataset_rag_ready.json          (~777 KB)
|   |-- crop_disease_rag_ready.json          (~1 KB)
|   |-- fertilizer_dataset_rag_ready.json    (~28 KB)
|   |-- crop_yield_dataset_rag_ready.json    (~10.1 MB)
|   |-- soil_dataset_rag_ready.json          (~7.9 MB)
|   |-- rice_dataset.json                    (~44 KB)
|   +-- weather_dataset_rag_ready.json       (~526 MB, SKIPPED — live API used)
|
+-- KisanAI-Android/                <- Android Studio project (WebView wrapper)
```

---

## 12. Configuration Reference

| Variable | Location | Default |
|---|---|---|
| `OLLAMA_MODEL` | `app.py:28` | `"llama3"` |
| `MAX_HISTORY` | `app.py:20` | `10` exchanges |
| `OPENWEATHER_API_KEY` | `app.py:31` | env var `OPENWEATHER_API_KEY` |
| `WEATHER_CACHE_TTL` | `app.py:35` | `1800` seconds (30 min) |
| `AUDIO_PATH` | `app.py:27` | `Static/response.mp3` |
| `DATASET_DIR` | `rag_pipeline.py:44` | `<project_root>/Dataset_RAG_Ready/` |
| `MAX_RECORDS_PER_TYPE` | `rag_pipeline.py:59` | crop: 5000, soil: 3000, others: None |
| LLM temperature | `app.py:366` | `0.3` |
| LLM top_p | `app.py:367` | `0.9` |
| LLM num_gpu | `app.py:365` | `0` (CPU-only) |
| BM25 top_k | `rag_pipeline.py:247` | `5` documents per query |
| Flask port | `app.py:600` | `5000` |
| Min audio duration | `app.py:208` | `0.5` seconds |
| Min audio file size | `app.py:449` | `1,000` bytes |

---

## 13. Setup and Running

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- Internet access for Google STT, gTTS, and OpenWeatherMap APIs

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Kisan_AI

# 2. Install Python dependencies
pip install -r requirement.txt

# 3. Pull the Llama 3 model into Ollama (one-time, ~4 GB download)
ollama pull llama3
```

### Run Locally

```bash
python app.py
# App available at http://localhost:5000
```

### Run with Public ngrok URL (for mobile or remote access)

```bash
python run_ngrok.py
# Prints a public HTTPS URL like: https://xxxx.ngrok.io
```

---

## 14. Performance Characteristics

| Stage | Typical Time |
|---|---|
| Startup (RAG index build) | ~10-30 seconds |
| STT (Google Speech API) | ~1-2 seconds |
| RAG retrieval (BM25 in-memory) | **< 50 milliseconds** |
| Weather API (or cache hit) | ~0.5-1 second (or < 1ms from cache) |
| LLM inference (CPU-only Llama 3) | ~5-30 seconds (varies by response length) |
| TTS synthesis (gTTS) | ~1-2 seconds |
| **Total round-trip** | **~8-35 seconds** per turn |

> The dominant cost is **LLM inference** on CPU. On a machine with a GPU, inference reduces to 1-5 seconds, making the total round-trip approximately 3-10 seconds.

---

## 15. Error Handling

| Stage | Error Condition | Response |
|---|---|---|
| Audio Upload | No `audio` field in request | HTTP 400 |
| Audio Upload | File < 1,000 bytes (empty blob) | HTTP 400 + mic check advice |
| Audio Conversion | FFmpeg non-zero exit code | HTTP 400 + ffmpeg error log |
| Audio Conversion | Timeout (> 30s) | HTTP 400 |
| STT | Audio < 0.5s duration | HTTP 400 |
| STT | `UnknownValueError` | Returns `("", lang)` gracefully |
| STT | Google API `RequestError` | Returns `("", lang)` gracefully |
| Weather | API key missing/invalid | Returns `None` — no weather block in prompt |
| Weather | Network timeout | Returns `None` — graceful degradation |
| Geolocation | Both providers fail | HTTP 503 |
| RAG | Dataset file not found | Warning logged; that dataset skipped |
| RAG | Empty query / no tokens | Returns `""` — no context added to prompt |
| LLM | Ollama server not running | Exception -> HTTP 500 |
| LLM | Empty query | HTTP 400 |
| TTS | gTTS API failure | Exception -> HTTP 500 |
| General | Unhandled exception | HTTP 500 + exception message |

---

*KisanAI — Empowering Indian farmers with AI-driven agricultural intelligence.*
