# PlanPeek Architecture

Technical reference for contributors. For setup, running the app, and the user-facing
feature list, see [README.md](README.md) — this document explains how the pieces fit
together, why they're built the way they are, and where the sharp edges are.

## Status: two pipelines, one exposed

PlanPeek is built around **two independent data pipelines** that don't share a search
API or a document format, plus modules shared by both:

| | 401(k) / Form 5500 | Life insurance / VUL |
|---|---|---|
| Source | DOL EFAST2 (`efast.py`) | SEC EDGAR (`edgar.py`) |
| Retrieval | `retrieval.py` | `vul_retrieval.py` |
| Extraction | `parser.py` + `schedule_of_assets.py` (regex heuristics) | `vul_parser.py` (LLM-driven) |
| Risk classification | `risk.py` (shared) | `risk.py` (shared) |
| **Streamlit UI** | ✅ wired (`app.py`) | ❌ not wired |
| **MCP tools** | ✅ wired (`mcp_server.py`) | ❌ not wired |

**The VUL pipeline and the risk-exposure module are real, tested, and validated against
live SEC EDGAR filings — but nothing in `app.py` or `mcp_server.py` calls them yet.**
Today, a user of the running app or the MCP server can only look up Form 5500 / 401(k)
data. The VUL backend exists as a library a future UI/MCP layer can call; don't infer
end-to-end life-insurance support from the presence of these files.

## Repository layout

```
app.py                       Streamlit UI (Form 5500 / 401k only, see table above)
planpeek/
  efast.py                   EFAST2 search client              ┐
  retrieval.py                DOL S3 PDF/ZIP download            │ 401(k) /
  parser.py                   PyMuPDF text + Schedule H heuristics │ Form 5500
  schedule_of_assets.py       Itemized fund-holdings extraction    │ pipeline
  context.py                   EFAST hidden-field sanitization    ┘
  edgar.py                    SEC EDGAR search client            ┐
  vul_retrieval.py             N-6 document download              │ VUL / life
  vul_parser.py                LLM-driven fund-menu extraction     │ insurance
                                                                   ┘ pipeline
  risk.py                     Equity/bond/cash classifier + risk band  (shared)
  rag.py                      Databricks LLM integration + RAG engine  (shared)
  models.py                   All dataclasses                          (shared)
  charts.py                   Plotly figure builders          (Form 5500 UI only)
  glossary.py                  Plain-language term definitions
  mcp_server.py                MCP server (Form 5500 tools only)
tests/                        One test file per module, same name convention
```

## Pipeline 1: 401(k) / Form 5500

```
efast.py:search()/history()  →  Filing
        │
        ▼
retrieval.py:download_filing()  →  RetrievedFiling (PDF bytes)
        │
        ▼
parser.py:parse_pdf()  →  ParsedFiling { pages, chunks, metrics, warnings }
        │                          │
        │                          └─ schedule_of_assets.py:extract_holdings()
        │                             → metrics.holdings: list[AssetHolding]
        ▼
charts.py (6 chart builders)  +  rag.py:RagEngine (Q&A)  +  risk.py (optional)
```

### `efast.py` — search

Wraps DOL's public search API (`efast.dol.gov/services/afs`, Lucene query syntax).
`build_query()` distinguishes EIN search (`ein:XXXXXXXXX`) from company/sponsor phrase
search. No API key needed. `Filing.from_hit()` in `models.py` maps the raw JSON.

### `retrieval.py` — download

**Investigated, not guessed** (see README's "Filing retrieval finding"): the actual
production download path is DOL's S3 bucket
(`efast2-filings-public.s3.amazonaws.com/prd/...`), serving a real `%PDF-1.7` with a
misleading `application/octet-stream` content-type — so content is identified by file
signature (`%PDF-`, `PK` for ZIP), never by the HTTP header. Security properties worth
knowing if you touch this file:

- Host allowlist (`ALLOWED_DOWNLOAD_HOSTS`) checked **before and after** following
  redirects — a redirect to an unlisted host is rejected even after a 200 response.
- `pdf_path` is never trusted as an absolute URL directly: `resolve_filing_url()` uses
  `urljoin` with a leading-slash strip, specifically to block a `//evil.com`
  protocol-relative bypass.
- Size-capped at 80MB, streamed (checked both via `Content-Length` and while reading, so
  a lying header doesn't help).
- ZIP handling picks the *largest* PDF inside the archive and re-validates its signature
  after extraction — a zip-bomb can't inflate past the same 80MB cap since the check
  happens before `archive.read()`.

### `parser.py` — Schedule H heuristics

PyMuPDF (`get_text("text", sort=True)`) extracts page text; everything else is regex.
Two things worth understanding before touching this file:

1. **EFAST hidden sentinels.** Populated PDF form fields carry an invisible numeric
   prefix (`-123456789012345...`) that PyMuPDF still extracts as text. `_number()`
   strips these (`parser.py`) before treating a value as real; `context.py`'s
   `sanitize_filing_context()` does the same for text handed to the LLM (chunks,
   citations, RAG answers) so a mask artifact never gets embedded, quoted, or reasoned
   over. Both are covered by `test_parser.py`'s
   `test_decodes_efast_hidden_numeric_sentinels`.
2. **Schedule H asset_categories are by *investment vehicle*, not asset class.** A line
   item like "Common / collective trusts" or "Registered investment companies" says
   nothing about whether the underlying holding is equity or bonds — Form 5500 doesn't
   require that disclosure at the plan level. In a real, live-validated example (Google
   LLC's 401(k), $48.7B in assets), **99.4% of assets fell into these vehicle-type
   buckets** — meaning `asset_categories` alone is nearly useless for risk
   classification on a typical plan. This is exactly why `schedule_of_assets.py` exists.

### `schedule_of_assets.py` — the actual fund names

Large plans that hold assets directly (not exclusively through a single trust) attach
Form 5500 Schedule H, Line 4i — an itemized list of every holding by fund name and
year-end value. This is the *only* place a Form 5500 filing names individual funds.
Validated against the same live Google 401(k) filing: real names like "Vanguard 500
Index Fund" and "MetWest Total Return Bond Fund" turn the 99.4%-unclassifiable plan
above into ~98% classifiable by `risk.py`.

Known limitations, both intentional and both self-reported rather than silently
swallowed:

- **Not every filing has this schedule.** A plan invested entirely through a single
  master trust files its holdings on the *trust's own* separate Form 5500 (a "Direct
  Filing Entity" filing) — this module doesn't chase that down. Absence is silent
  (empty `holdings` list, no warning); presence-but-incomplete is not.
- **Multi-line wrapped descriptions with embedded numbers can mis-parse.** A holding
  whose description wraps across PDF lines (e.g. a participant-loan row listing
  maturity dates) can have a date like "2025" mistaken for a trailing dollar value.
  Rather than trying to solve this generally (would need real column-position data
  PyMuPDF's flattened text doesn't preserve), `extract_holdings()` sums its own
  extracted holdings against the schedule's printed **Total** line and returns a
  warning when they don't reconcile within 2% — so a caller (or a human) knows to
  verify rather than trusting a silently-incomplete breakdown. See
  `test_extract_holdings_warns_when_sum_does_not_reconcile_with_total`.

### `charts.py`

Six pure functions, each `Filing`/`FilingMetrics` in, `go.Figure | None` out (`None`
when the underlying data isn't present — never a fabricated empty chart). Color palette
(`TEAL`, `CORAL`, `GOLD`, `MINT`, `NAVY`) is defined once here and assigned
positionally; extracted/model-generated data never carries its own color, by design.

## Pipeline 2: Life insurance / VUL

```
edgar.py:search()/history()  →  VulFiling
        │
        ▼
vul_retrieval.py:download_vul_document()  →  RetrievedVulDocument (HTML)
        │
        ▼
vul_parser.py:extract_vul_filing()  →  VulExtraction { allocation_chart, sub_funds_list, risk_metrics, warnings }
        │
        ▼
risk.py (shared)
```

### `edgar.py` — search

SEC EDGAR's full-text search API (`efts.sec.gov/LATEST/search-index`) is genuinely
EFAST2-like: free, public, JSON. Form **N-6** (and `N-6/A` amendments) is the filing
type for VUL separate accounts — confirmed against live filings from Lincoln, John
Hancock, TIAA, Athene, Protective, and Northwestern Mutual (a mutual company; its
*variable* separate account is still SEC-registered even though its general-account
statutory filings aren't on EDGAR at all).

`search()` (full-text search) and `history()` (`data.sec.gov/submissions/CIK....json`)
serve different purposes: full-text search finds *which document within a filing*
matched a query — which can be an exhibit, not the prospectus body — while `history()`
returns the filing's own declared `primaryDocument`, which is authoritative. Treat
`search()` as discovery, `history()` as the source of truth for which document to fetch.

**`EDGAR_USER_AGENT` is not optional in practice.** SEC's fair-access policy requires an
identifying User-Agent (name + contact email) on every request. Confirmed against a
live filing: a generic User-Agent works fine against the JSON APIs but gets a **403**
(not 404) specifically on document downloads from `www.sec.gov/Archives`.
`vul_retrieval.py`'s error handling distinguishes this from "not found" and tells the
caller exactly what to set — see `test_download_vul_document_403_points_at_user_agent_requirement`.

### `vul_retrieval.py` — download

Same shape as `retrieval.py` (host allowlist, redirect recheck, size cap at 25MB) for
SEC's archive host instead of DOL's S3 bucket.

### `vul_parser.py` — LLM extraction, not regex

N-6 fee/fund tables have no standardized markup across filers — verified against a real
current Lincoln National N-6: fund names and expense ratios sit in separate `<td>`
cells buried under multiple layers of presentation-only `<div>`/`<font>` styling, with
no consistent class names or line-item codes. A regex heuristic parser (viable for
Schedule H's numbered lines) is not viable here.

The extraction pipeline:

1. **`_TableExtractor`** (stdlib `html.parser.HTMLParser`, stack-based) flattens every
   `<table>` in the document into rows of plain cell text, correctly handling nested
   tables (a layout table wrapping a data table) without corrupting either.
2. **`_is_fund_menu_shaped()`** keeps only percentage-dense, multi-row tables — this
   matters because the *raw* fund-table region of a real N-6 can span ~600KB of HTML
   (mostly styling markup). Once flattened to plain rows, that same real 74-fund menu
   shrinks to **13,765 characters** — comfortably inside one model call. (This is why
   the approach is "extract tables, then filter," not "slice raw HTML near a keyword" —
   the raw-HTML approach was tried first and found to not fit any reasonable context
   budget.)
3. The narrowed text goes to a Databricks-hosted chat model
   (`rag.build_databricks_chat_model()`) with a strict JSON-only extraction prompt
   (`SYSTEM_PROMPT`) that explicitly forbids inventing values and requires distinguishing
   ordinary rounding slop (`unclassified_reason: "rounding"`, ≤2 points) from a
   genuinely incomplete extraction (`"extraction_incomplete"`) rather than silently
   padding either to 100%.

**Not live-verified**: the actual Databricks/Claude model's extraction *quality*
(whether it reliably follows the schema against real fund tables) has only been tested
against a fake LLM standing in for the real call — no Databricks credentials were
available in the environment this was built in. The plumbing (prompt construction, JSON
parsing including fenced-code-block stripping, payload→dataclass mapping) is tested;
the model's actual behavior is not.

## Shared: risk exposure (`risk.py`)

Used by both pipelines: `schedule_of_assets.AssetHolding` (401k, dollar values) and
`vul_parser`'s `SubFund`/`AllocationCategory` (VUL, percentages) both reduce to the same
`classify_amounts()` → `summarize()` call, since the shared unit is "a labeled amount,"
not a plan-type-specific structure.

**This is a name-based keyword classifier — there is no ticker/CUSIP lookup anywhere in
this pipeline.** It only asserts what a holding's own name plausibly states, and
`classify_label()` returns `"unclassified"` rather than guessing when nothing matches.
Rule order matters (checked top-to-bottom in `_CLASSIFICATION_RULES`): more specific
patterns (bond, cash, target-date) are checked before the broad equity catch-all, so
e.g. "Total International Bond Index Fund" resolves to `fixed_income` despite also
containing equity-ish words like "international" and "index".

Known, accepted misclassification (found via the real Google 401(k) fund list, not
hypothetical): "Wellesley Income Fund" matches the `fixed_income` pattern (`income
fund`) but is actually a ~35/65 stock/bond balanced fund — there's no way to distinguish
this from a pure bond fund (e.g. "PIMCO Income Fund", correctly classified) by name
alone. This is a real, permanent limitation of name-based classification, not a bug to
fix; anyone using `risk.py` output should treat it as an approximation.

`summarize()` only returns a `risk_band` (Conservative/Moderate/Growth/Aggressive) when
at least `MIN_CLASSIFIED_SHARE_FOR_BAND` (60%) of total value was actually classified —
otherwise it reports the honest percentages with `risk_band=None` rather than a
confident-looking label built on mostly-guessed data. Target-date/balanced funds count
at `MIXED_EQUITY_WEIGHT` (50%) toward the equity side of the band, since they hold a mix
rather than being purely growth or defensive.

**End-to-end validated** against Google's real 401(k) Schedule of Assets: 29.3% equity,
65.4% mixed (target-date trusts dominate this specific plan) → **"Moderate"** — see
`test_real_google_plan_lands_in_moderate_band` and the module docstring for the full
worked example.

## Shared: Databricks / LLM integration (`rag.py`)

Two independent auth paths, both ending in the same OpenAI-compatible `ChatOpenAI`
client pointed at a Databricks serving endpoint:

- **Local development**: `DATABRICKS_PROFILE` set → `_DatabricksOAuthToken` uses a
  `WorkspaceClient(profile=...)` backed by `databricks auth login`'s refreshable OAuth
  session.
- **Databricks Apps deployment**: `DATABRICKS_PROFILE` empty → `WorkspaceClient()` with
  no profile, picking up the platform's injected service-principal credentials
  automatically. `app.py`'s `_build_databricks_rag_engine` additionally prefers a
  `DATABRICKS_HOST` env var (which Databricks Apps injects) over any stale local
  `DATABRICKS_FM_BASE_URL`.

Both paths cross-check that the resolved workspace host matches the model-serving URL's
host before ever making a request, refusing with a specific error naming both hosts if
they don't match — this exists because it's an easy misconfiguration (right profile,
wrong workspace) that would otherwise fail with an opaque auth error far from the actual
cause.

`build_databricks_chat_model()` is a **separate, self-contained** helper (used by
`vul_parser.py`) rather than a refactor of `RagEngine.__init__`'s own model-construction
code — deliberately, to avoid touching the auth logic above, which has already been
fixed twice in this project's history for Databricks Apps compatibility. Some
duplication between the two is an accepted tradeoff against that risk.

`RagEngine` owns one **ephemeral, per-filing Chroma collection** (`EphemeralClient`,
in-memory) — never persisted, never shared across filings, explicitly `.close()`d on
filing change or engine replacement (`app.py`'s `_reset_document_state`,
`_build_databricks_rag_engine`). `RAG_ENGINE_API_VERSION` exists because Streamlit can
re-run `app.py` while a browser session still holds a `RagEngine` built from a
previous, incompatible version of this module (a genuine Streamlit hot-reload hazard,
not defensive-programming-for-its-own-sake) — `app.py` checks this version and rebuilds
rather than reusing a stale engine.

## `mcp_server.py`

Exposes the Form 5500 pipeline only (see the status table at the top) as 4 tools,
intent-shaped rather than 1:1 with the internal API — a caller passes a `filing_id` and
a natural-language question, never a PDF or a vector store:

| Tool | Backs onto |
|---|---|
| `search_filings` | `efast.search()` |
| `get_filing_history` | `efast.history()` |
| `get_filing_financials` | `retrieval` + `parser`, formatted text |
| `ask_filing` | `retrieval` + `parser` + `rag.RagEngine` |

State (parsed filings, RAG engines) is cached per server process with a FIFO cap
(`MAX_CACHED_FILINGS = 10`) — plain-dict insertion order, not true LRU, but sufficient
to bound memory (each engine holds an ephemeral Chroma collection) for a long-running
process. Evicted engines are `.close()`d.

**Runs over stdio — cannot be driven by typing into a terminal.** Running
`python -m planpeek.mcp_server` directly and pressing Enter produces a
`json_invalid: EOF while parsing a value` error; this is expected, not a bug — the
process is waiting for structured JSON-RPC frames from an MCP client, not human
keystrokes. Use `uv run mcp dev planpeek/mcp_server.py` (MCP Inspector, browser UI) to
poke at it interactively, or `claude mcp add` (see README) to register it with a real
client.

## `app.py` (Streamlit UI)

Session-state caching mirrors the MCP server's process-level caching, scoped to a
browser session instead of a server process: `_cached_search`/`_cached_history` use
`st.cache_data` with a TTL; `_cached_download_and_parse` additionally caps at
`max_entries=30`. `_activate_filing()` is the single entry point that resets all
downstream state (RAG engine closed, chat history cleared) when a new filing is opened,
so state from one filing never leaks into the view of another.

Five tabs, all driven by the Form 5500 pipeline only: Overview, Financial detail, Ask
the filing, Source document, Glossary.

## `models.py` — data reference

| Dataclass | Pipeline | Produced by |
|---|---|---|
| `Filing` | 401k | `efast.py` |
| `RetrievedFiling` | 401k | `retrieval.py` |
| `TextChunk`, `FilingMetrics`, `ParsedFiling` | 401k | `parser.py` |
| `AssetHolding` | 401k | `schedule_of_assets.py` |
| `VulFiling` | VUL | `edgar.py` |
| `RetrievedVulDocument` | VUL | `vul_retrieval.py` |
| `SubFund`, `AllocationCategory`, `VulRiskMetrics`, `VulExtraction` | VUL | `vul_parser.py` |

`Filing`, `TextChunk`, `RetrievedFiling`, `AssetHolding`, `VulFiling`,
`RetrievedVulDocument`, `SubFund`, `AllocationCategory`, `VulRiskMetrics` are all frozen
(`@dataclass(frozen=True, slots=True)`) — immutable value objects. `FilingMetrics` and
`ParsedFiling` are mutable (`slots=True` only) since `parser.py` builds them
incrementally (`metrics.holdings = holdings` after the fact).

## Testing philosophy

Every module here has a same-named test file. Beyond ordinary unit tests, several
modules are validated against **real, captured data from live public filings** rather
than only idealized fixtures — this matters because the riskiest parts of this codebase
(PDF/HTML text extraction, heuristic table parsing) fail in ways that only show up
against real-world formatting mess:

- `test_schedule_of_assets.py` embeds a real Schedule of Assets page from a live Google
  LLC 401(k) filing, verbatim (including its actual PDF column spacing — see the
  `per-file-ignores` entry in `pyproject.toml` for why that file is exempt from the
  line-length rule).
- `test_risk.py`'s `REAL_FUND_CLASSIFICATIONS` are the actual fund names from that same
  filing, not invented examples.
- `test_edgar.py` and `test_vul_retrieval.py` use fixture shapes copied from real EDGAR
  API responses.
- `test_vul_parser.py`'s HTML fixture deliberately reproduces the real nested
  `<div>`/`<font>`-per-cell structure found in a live Lincoln National N-6, not a
  simplified `<table><td>` mockup.

Run `uv run pytest` for the base suite; the MCP server tests require the optional `mcp`
extra (`uv sync --extra dev --extra mcp`) and are skipped (not failed) without it via
`pytest.importorskip`.

## Evaluation harnesses (`planpeek/evals/`)

Ordinary tests check that code behaves correctly; this subsystem checks that a
**classifier or an LLM answer is any good** — a different question, ported from two
Gen Academy "AI Evals" tutorials (routing-agent evaluation, and agentic-RAG evaluation
with RAGAS). Nothing here is wired into `app.py` or `mcp_server.py`; these are
standalone harnesses a developer runs deliberately, same status as the VUL pipeline.

| Module | Ports the pattern from | Applies to |
|---|---|---|
| `evals/metrics.py` | `sklearn.metrics.classification_report` (hand-rolled, no new dependency) | shared |
| `evals/risk_classifier_eval.py` | the routing-agent notebook's golden-dataset + precision/recall/F1 | `risk.classify_label()` |
| `evals/rag_judge.py` | the "one focused failure mode" LLM-judge notebook, and RAGAS's `Faithfulness` | `RagEngine.ask()` |
| `evals/rag_eval.py` | RAGAS's `context_precision`/`context_recall`, and the tutorial's non-LLM `citation_f1` | `RagEngine.ask()` |

**`risk_classifier_eval.py`** runs `evals/data/risk_classifier_golden.csv` (29 rows,
mostly real fund names from the same live Google 401(k) filing used elsewhere in this
doc) through `classify_label()` and reports per-class precision/recall/F1. Two rows are
tagged `known_miss` in the CSV's `source` column and are **kept in deliberately**:
Wellesley Income Fund (a balanced fund the `fixed_income` pattern catches via "income
fund", the same pattern that correctly identifies pure bond funds) and Fidelity VIP
Contrafund (a well-known equity fund whose name contains no asset-class keyword at
all). A report that scores 100% would mean the golden dataset stopped reflecting the
classifier's real, documented limitations — run it and it lands around 93% accuracy /
0.92 macro-F1, with those two rows visibly the misses. Run directly:
`uv run python -m planpeek.evals.risk_classifier_eval`.

**`rag_eval.py` + `rag_judge.py`** evaluate `RagEngine` answers against
`evals/data/rag_qa_golden.csv` (question → expected page numbers) using
`evals/sample_filing.py`, a small **synthetic** filing fixture — authored, not
downloaded, specifically so the "true" page for every fact is known exactly and the
harness is fully offline-testable without a live filing or Databricks credentials.
Bring your own golden CSV against a real filing's `RagEngine` to evaluate that instead;
the harness itself doesn't know or care that the bundled one is synthetic. Two
deterministic metrics need no LLM at all:

- **citation accuracy** — do the `[p. N]` pages the model actually wrote in its answer
  text match the golden dataset's expected pages?
- **retrieval accuracy** — do the pages the Chroma query retrieved
  (`RagAnswer.citations`) match the expected pages?

These are scored separately on purpose: retrieval can find the right page while the
model fails to cite it in prose, or vice versa, and collapsing them into one score
would hide which stage is actually failing.

**Faithfulness** (`rag_judge.judge_faithfulness`) is the one LLM-dependent piece: a
judge scoped to exactly one failure mode ("does this answer state anything the cited
excerpts don't support?"), not a general answer-quality grader — same reasoning as the
tutorial's "vague utterance" judge: a judge with one job is easier to trust. It's
optional (`run_eval(..., judge=None)` skips it) and, like `ask_filing` and
`vul_parser.py`'s extraction quality, its real-model behavior is untested in the
environment this was built in — only the JSON-parsing and prompt-construction plumbing
is (`test_rag_judge.py`, via a fake LLM).

Run `uv run python -m planpeek.evals.rag_eval` against the bundled sample filing (needs
Databricks credentials, see the Configuration reference below).

## Configuration reference

| Variable | Used by | Required? |
|---|---|---|
| `DATABRICKS_FM_BASE_URL` | `rag.py` | No — has a documented default workspace |
| `DATABRICKS_PROFILE` / `DATABRICKS_CONFIG_PROFILE` | `rag.py` | No — empty means Databricks Apps service-principal auth |
| `DATABRICKS_HOST` | `app.py` | No — Databricks Apps injects this; overrides `DATABRICKS_FM_BASE_URL` when present |
| `DATABRICKS_FM_TOKEN` | `rag.py` | No — PAT, takes precedence over OAuth profile when set |
| `LLM_MODEL` / `EMBEDDING_MODEL` | `rag.py` | No — documented defaults |
| `EDGAR_USER_AGENT` | `edgar.py`, `vul_retrieval.py` | **Yes, in practice** — see the 403 note above |

See `.streamlit/secrets.toml.example` for the local-development equivalent of the
Databricks variables.
