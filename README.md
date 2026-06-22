# iPhone User Guide — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about the
**Apple iPhone User Guide (iOS 7.1)** — a 162-page PDF — through a conversational
interface. Answers are grounded **strictly** in the document, every answer
**cites the page(s)** it used, and the bot **says when an answer isn't in the
guide** instead of fabricating one.

---

## Run it (the grader's path)

The vector index is **already populated in Qdrant Cloud**, so there is no
ingestion step. The app answers immediately after the container starts.

```bash
# 1. Clone
git clone <this-repo> && cd <this-repo>

# 2. Configure: copy the template and fill in the keys
cp .env.example .env
#   edit .env -> OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Build
docker build -t chatbot:1.0 .

# 4. Run  (the app listens on port 8000)
docker run -p 8000:8000 --env-file .env chatbot:1.0

# 5. Open http://localhost:8000 and start chatting
```

**Port: `8000`** (configurable via `APP_PORT` in `.env`; map `-p <APP_PORT>:<APP_PORT>` if you change it).

> The grader must supply their own `OPENAI_API_KEY`. The same key powers both
> embeddings and chat. `QDRANT_URL` / `QDRANT_API_KEY` point at the
> pre-populated cloud collection (provided separately — never committed).

---

## Stack

| Layer | Choice | Version |
|---|---|---|
| Orchestration | **LangGraph** (+ LangChain) | langgraph 1.2.6 / langchain 1.3.10 |
| Embedding model | **OpenAI `text-embedding-3-small`** | 1536-dim |
| Chat model | **OpenAI `gpt-4o-mini`** | temperature 0 |
| Vector DB | **Qdrant Cloud** (cosine) | qdrant-client 1.18.0 |
| UI | **Chainlit** | 2.11.1 |
| Runtime | Python 3.11 / single Docker container | — |

---

## Architecture

```
Ingestion (offline, run once)          Serving (in the container, every query)
─────────────────────────────         ──────────────────────────────────────
PDF                                     User question
 │ load page-by-page                      │
 │ skip front matter (title + TOC)        ▼
 │ parse section from running header   ┌──────────────┐  rewrites follow-ups
 ▼                                     │ contextualize│  using chat history
chunks (≤1000 chars, within a page)    └──────┬───────┘
 │ + metadata {source, page, section}        ▼
 │ embed (text-embedding-3-small)       ┌──────────────┐  top-k semantic search
 ▼                                      │   retrieve   │  over Qdrant Cloud
Qdrant Cloud collection  ◄──────────────└──────┬───────┘
                                               ▼
                                        ┌──────────────┐  strict grounding +
                                        │   generate   │  structured output:
                                        └──────┬───────┘  {found, answer, cites}
                                               ▼
                                        answer + validated page citations
```

The pipeline is a 3-node LangGraph state machine compiled with a checkpointer,
so conversation state persists per session (`thread_id`) for **multi-turn** chat.

---

## Ingestion & retrieval design (the reasoning)

### Chunking strategy
- **Splitter:** `RecursiveCharacterTextSplitter`, **chunk size 1000 chars
  (~250 tokens), overlap 150 (~15%)**, separators `["\n\n", "\n", ". ", " "]`.
- **Why this size:** the guide is task-oriented — most answers are a short
  procedure ("Set up the Touch ID sensor. Go to Settings > …"). ~1000 chars is
  large enough to hold one complete instruction but small enough that the
  embedding vector represents a single topic, which sharpens retrieval
  precision. Real chunks averaged ~790 chars.
- **Overlap of 150** preserves continuity when a procedure is split across two
  chunks, so a step isn't orphaned from its heading.
- **Page-aware splitting (the key decision):** text is chunked **within each
  page**, never across pages. Therefore every chunk maps to **exactly one page
  number** → citations are unambiguous and directly verifiable, which the brief
  says is mandatory and will be tested.

### Ingestion pipeline
1. Load the PDF page-by-page with `pypdf` (1-based page index).
2. **Skip front matter** — the title page and table of contents are detected
   (short page / mostly-numbered lines) and dropped. TOC lines like
   "Set up mail … 16" match queries but contain no answers, so indexing them
   would pollute retrieval.
3. **Parse the section** from each page's running header (e.g.
   "Chapter 5 Phone 45" → `Chapter 5: Phone`). The extractor tolerates this
   PDF's quirk of splitting multi-digit numbers with spaces ("Chapter 1 2").
4. Strip the repeated running header from the body so it doesn't add noise to
   embeddings.
5. Embed with `text-embedding-3-small` and upsert into Qdrant.
6. **Deterministic `uuid5` IDs** make re-ingestion an idempotent upsert (no
   duplicates); `force_recreate=True` rebuilds a clean index each run.

Result: **155 content pages → 380 chunks**, 36 sections (Chapters 1–32 +
Appendices A–D), 0 chunks missing a section.

### Vector-store fields (metadata) and why
| Field | Purpose |
|---|---|
| `page` | **The citation.** 1-based PDF index = printed page number, so a cited page opens to the right place. |
| `section` | Human-readable provenance shown alongside the page ("Chapter 5: Phone"). |
| `source` | Document identity ("iPhone User Guide (iOS 7.1)") — future-proofs a multi-doc index. |
| `chunk` | Per-page chunk index — part of the deterministic ID and useful for debugging. |
| `page_content` | The chunk text itself (Qdrant payload), returned for grounding + display. |

### Embedding model choice
`text-embedding-3-small` (1536-dim): strong retrieval quality (top MTEB tier),
an 8191-token context window (so chunk size is a free choice, not capped by the
model), and ~2 cents to embed the whole document once. The same OpenAI key
already needed for the chat model covers embeddings — fewer secrets for the
grader. Qdrant Cloud's built-in inference (MiniLM/E5, 384-dim, 256-token window)
was evaluated and rejected: lower retrieval quality and a context window that
would truncate our chunks.

---

## Grounding, citations & "not found"
- A strict system prompt instructs the model to answer **only** from the
  retrieved context and to refuse otherwise.
- The model returns **structured output** `{found, answer, citations[]}`.
- **Hallucination guard:** cited pages are validated against the pages actually
  retrieved — a page the model didn't retrieve can never appear as a source.
- `found=false` ⇒ the UI shows the refusal and **no sources**.
- Citations are rendered **inline** under every answer (page + section).

---

## Project structure

```
.
├── app.py              # Chainlit UI entrypoint
├── ingest.py           # one-time ingestion (run locally, not in the container)
├── src/
│   ├── config.py       # pydantic-settings; single source of truth
│   ├── vectorstore.py  # shared embeddings + Qdrant accessors (ingest & serve)
│   ├── prompts.py      # grounding + contextualization prompts
│   └── graph.py        # LangGraph pipeline (contextualize → retrieve → generate)
├── data/               # source PDF (used by ingestion only; not shipped in image)
├── Dockerfile
├── requirements.txt    # pinned, verified to install + import on Python 3.11
├── .env.example        # documented env vars (copy to .env)
└── .dockerignore
```

## Re-running ingestion (optional — the grader does not need this)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
python ingest.py
```

## Configuration
All settings live in `.env` (see `.env.example` for the full list with
descriptions). `.env` is git-ignored and must never be committed.
