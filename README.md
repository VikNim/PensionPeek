# PensionPeek

PensionPeek is a Streamlit application for finding public U.S. Form 5500 filings, reading the
actual filing PDF, charting reported plan-level figures, and asking grounded follow-up questions
through a session-scoped RAG pipeline.

It is intentionally a single-filing research tool. There is no user account system, relational
metadata database, shared vector index, or Databricks cluster/data-store dependency. Databricks
Model Serving can be used for the chat and embedding models.

## What works

- Live EFAST2 search by company/plan sponsor name or 9-digit EIN
- Plan-year history for the same EIN and three-digit plan number
- Direct public filing retrieval, with defensive ZIP extraction and a manual-upload fallback
- PyMuPDF page-aware text extraction and Schedule H financial heuristics
- Plotly asset, participant, income/expense, and reported asset-category charts with independent
  zoom and reset controls
- A per-filing in-memory Chroma collection and a LangGraph `retrieve → answer` flow
- Databricks Model Serving with `databricks-claude-haiku-4-5` and
  `databricks-qwen3-embedding-0-6b`
- Swappable OpenAI or Anthropic chat models; OpenAI or local sentence-transformer embeddings
- Page citations and supporting retrieved excerpts with every chat answer
- Plain-language definitions and prominent aggregate-data disclaimers

## Filing retrieval finding

The download behavior was investigated before the parsing pipeline was built. The current
production EFAST2 search bundle defines this filing base:

```text
https://efast2-filings-public.s3.amazonaws.com/prd
```

The UI's download handler opens that base plus the search API's `pdfpath`. A live sample returned
an actual `%PDF-1.7` document even though its content type was `application/octet-stream`. This
differs from older DOL help language describing a ZIP download. PensionPeek therefore:

1. tries the current DOL-owned S3 path;
2. identifies content by file signature rather than HTTP content type;
3. accepts either a direct PDF or a ZIP containing PDFs; and
4. clearly offers upload and DOL bulk-image-service alternatives when disclosure is unavailable.

Only allowlisted DOL/S3 hosts are accepted, redirects are rechecked, and downloads are capped at
80 MB.

## Run locally

Python 3.11–3.13 is supported; Python 3.12 is a good default. Some ML dependencies may not yet
publish wheels for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or with `uv` and an installed Python 3.11 interpreter:

```bash
uv venv --python 3.11
uv sync --extra dev
uv run streamlit run app.py
```

Search and deterministic charts do not require a model credential. The configured Databricks
models use these settings:

```bash
export DATABRICKS_FM_BASE_URL="https://dbc-7b106152-caf3.cloud.databricks.com/serving-endpoints"
export DATABRICKS_PROFILE="dbc-7b106152-caf3"
export EMBEDDING_MODEL="databricks-qwen3-embedding-0-6b"
export LLM_MODEL="databricks-claude-haiku-4-5"
```

PensionPeek uses the profile's refreshable OAuth session, so a personal access token is not
required. Authenticate the profile once (and repeat when Databricks asks you to sign in again):

```bash
databricks auth login \
  --host "https://dbc-7b106152-caf3.cloud.databricks.com" \
  --profile "dbc-7b106152-caf3"
```

`DATABRICKS_CONFIG_PROFILE` is also recognized. If your organization permits PATs, setting the
optional `DATABRICKS_FM_TOKEN` takes precedence over the OAuth profile.

OpenAI and Anthropic remain available as alternatives:

```bash
export OPENAI_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."
```

You can instead copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. The populated
file is ignored by Git. Never put a Databricks token in source code. Model names can also be
overridden with `OPENAI_MODEL` and `ANTHROPIC_MODEL`.

The Databricks adapter uses the workspace's OpenAI-compatible serving routes. Filing chunks are
embedded without an instruction; Qwen question embeddings receive a retrieval-specific instruction
as recommended for that model. Both chunks and retrieved excerpts are sent to the configured
Databricks workspace.

With Anthropic, PensionPeek uses local `all-MiniLM-L6-v2` embeddings because Anthropic does not
provide the embedding adapter used here. The model downloads on first use. With OpenAI, the UI lets
you choose `text-embedding-3-small` or the same local embedding model.

## Verify

```bash
uv run pytest
uv run ruff check .
python -m compileall app.py pensionpeek tests
```

## Architecture

```text
EFAST2 search API
       │
       ├── metadata history ──► Plotly charts
       │
       └── pdfpath ──► DOL S3 PDF / ZIP ──► PyMuPDF
                                             │
                         ┌───────────────────┴──────────────────┐
                         │                                      │
                  Schedule H heuristics                  page-aware chunks
                         │                                      │
                       charts                    ephemeral Chroma collection
                                                                │
                                                   LangGraph retrieve → answer
                                                                │
                                                   answer + page citations
```

The Chroma client is `EphemeralClient`; it is local and session-scoped, not hosted or shared.
Databricks or OpenAI embeddings transmit extracted filing chunks to the selected provider. Local
embeddings keep embedding input on the app machine. In every configuration, the retrieved excerpts
needed to answer a question are sent to the selected answering provider.

## Known limitations

- PDF forms are visually structured, and text extraction order varies. Displayed parsed totals are
  heuristics and are explicitly labeled for source verification.
- Image-only/scanned filings need OCR, which is not included in v1.
- EFAST does not publicly disclose every filing. Older, foreign, one-participant, superseded, or
  sensitive filings may be unavailable.
- Search metadata exposes beginning-of-year participant counts broadly. End-of-year counts are
  charted only when the selected PDF yields one; PensionPeek does not invent missing values.
- Reported asset categories are aggregate plan categories. They are never personal allocations.
- Investment-by-investment detail depends on whether the filing has a public Schedule of Assets.

For unavailable filing images, DOL's search help directs developers to request its Form 5500 bulk
image service by emailing `foiarequest@dol.gov` with subject `EBSA Form 5500 image service request`
and contact information.

## Public sources

- [EFAST2 Form 5500 Search](https://www.efast.dol.gov/5500Search/)
- [EFAST2 search help](https://www.efast.dol.gov/5500Search/help/help.html)
- [DOL Form 5500 datasets](https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/public-disclosure/foia/form-5500-datasets)

PensionPeek is an independent research interface and is not affiliated with or endorsed by the
U.S. Department of Labor. It does not provide legal, tax, investment, or fiduciary advice.
