# local-rag-app

A local, on-device RAG (Retrieval-Augmented Generation) assistant for Melis Lazer. It answers questions from ingested documents and can parse and price laser-cutting order requests written in natural language (Turkish).

Runs entirely on-device via [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) — no cloud API keys required. Embeddings use `qwen3-embedding-0.6b`; chat/extraction uses `phi-3.5-mini`.

## Project structure

```
app.py              Streamlit chat UI
main.py             CLI entry point + query routing (pricing vs. RAG)
init_db.py          Creates the local SQLite store (db/rag.db)
src/ingest.py       Chunks data/sample.txt, embeds it, stores it in db/rag.db
src/rag_pipeline.py Embedding search + grounded answer generation
src/pricing.py      Reads the price list Excel file and calculates quotes
data/               Source documents to ingest
db/                 Local SQLite vector store (generated, not committed)
```

## Setup

Requires Python 3.13 and [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/get-started) installed on the machine (Windows or macOS — Foundry Local supports both, including Apple Silicon).

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` was generated on Windows. If any package fails to resolve on macOS, check PyPI for a platform-specific build of that package and adjust the pin.

You'll also need the real price list Excel file (`MELİS LAZER FİYAT TARİFESİ 2026 firma.xlsx`) in the project root — it's intentionally **not** committed to this repo (see below). Copy it in manually on each machine you run this on.

## Running

```bash
python init_db.py     # create the local database (first time only)
python -m src.ingest   # embed and store data/sample.txt
python main.py         # CLI
# or
streamlit run app.py   # web UI
```

## Note on the price list file

This project reads pricing from a company Excel file (`src/pricing.py`). That file contains real business pricing data and is excluded via `.gitignore` (`*.xlsx`) so it never ends up in this — public — repository. If you clone this repo on a new machine, place a copy of the price list file in the project root yourself; it's not tracked by git.
