from __future__ import annotations

import html
import importlib
import inspect
import os
from typing import Any

import streamlit as st

import pensionpeek.rag as rag_module
from pensionpeek.charts import (
    asset_categories_chart,
    assets_history_chart,
    income_expense_chart,
    participants_history_chart,
)
from pensionpeek.efast import EfastClient, EfastError
from pensionpeek.glossary import TERMS
from pensionpeek.models import Filing, ParsedFiling, RetrievedFiling
from pensionpeek.parser import ParsingError, parse_pdf
from pensionpeek.retrieval import RetrievalError, download_filing, resolve_filing_url

# Streamlit can re-run app.py while retaining an older imported project module. Refresh only when
# that cached constructor predates the Databricks integration, avoiding a server-restart-only fix.
if "databricks_base_url" not in inspect.signature(rag_module.RagEngine.__init__).parameters:
    importlib.invalidate_caches()
    rag_module = importlib.reload(rag_module)

RagEngine = rag_module.RagEngine
RagError = rag_module.RagError

st.set_page_config(
    page_title="PensionPeek · Form 5500 explorer",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    :root {
        --ink: #17324d;
        --teal: #0d7c7b;
        --coral: #e36d5d;
        --cream: #f5f0e7;
        --paper: #fffdf8;
        --rule: #ddd6c8;
    }
    .stApp {
        background:
          radial-gradient(circle at 89% 5%, rgba(227,109,93,.10), transparent 25rem),
          radial-gradient(circle at 8% 22%, rgba(13,124,123,.08), transparent 27rem),
          var(--cream);
        color: var(--ink);
    }
    .block-container { max-width: 1320px; padding-top: .85rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h1 {
        font-size: clamp(2.25rem, 4vw, 3.2rem) !important; line-height: 1 !important;
        margin: .1rem 0 .25rem !important;
    }
    .pp-kicker {
        color: var(--teal); font-size: .7rem; font-weight: 800; letter-spacing: .14em;
        text-transform: uppercase; margin-bottom: .2rem;
    }
    .pp-hero {
        display: grid; grid-template-columns: minmax(280px, .8fr) minmax(420px, 1.35fr);
        align-items: end; gap: 2rem; padding: .15rem 0 .55rem;
    }
    .pp-hero p { color: #526373; font-size: 1rem; line-height: 1.45; margin: 0 0 .3rem; }
    .pp-badge {
        display: inline-block; padding: .24rem .55rem; border-radius: 999px;
        background: #dff0ea; color: #12635f; font-size: .7rem; font-weight: 750;
        vertical-align: middle; margin-left: .45rem; letter-spacing: 0;
    }
    .pp-disclaimer {
        border: 1px solid #d8b76d; border-left: 4px solid #d6a94c; border-radius: 9px;
        padding: .5rem .75rem; background: rgba(255,253,248,.86); color: #624f29;
        font-size: .82rem; line-height: 1.35; margin: .15rem 0 .7rem;
    }
    .pp-plan-head {
        padding: .7rem 1rem; background: var(--ink); color: white; border-radius: 12px;
        margin: .65rem 0 .65rem; box-shadow: 0 8px 20px rgba(23,50,77,.10);
    }
    .pp-plan-head h2 { color: white; font-size: 1.45rem; margin: 0 0 .18rem; }
    .pp-plan-head p { margin: 0; color: #d7e3ec; font-size: .88rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,253,248,.82); border: 1px solid var(--rule);
        padding: .6rem .8rem; border-radius: 10px;
    }
    div[data-testid="stForm"], div[data-testid="stDataFrame"], .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255,253,248,.72); border: 1px solid var(--rule); border-radius: 11px;
        padding: .65rem;
    }
    div[data-testid="stVerticalBlock"] { gap: .65rem; }
    .stButton > button[kind="primary"], .stFormSubmitButton > button {
        background: var(--coral); color: white; border: 0; font-weight: 750;
    }
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
        background: #c95449; color: white; border: 0;
    }
    a { color: var(--teal) !important; }
    footer { visibility: hidden; }
    @media (max-width: 760px) {
        .pp-hero { grid-template-columns: 1fr; gap: .25rem; }
        .block-container { padding-top: .55rem; }
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "search_results": [],
        "active_filing": None,
        "history": [],
        "retrieved": None,
        "parsed": None,
        "retrieval_error": None,
        "chat_messages": [],
        "rag_engine": None,
        "rag_fingerprint": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=600, show_spinner=False)
def _cached_search(value: str, search_by: str) -> list[Filing]:
    return EfastClient().search(value, search_by=search_by)


@st.cache_data(ttl=1_800, show_spinner=False)
def _cached_history(ein: str, plan_number: str) -> list[Filing]:
    return EfastClient().history(ein, plan_number)


@st.cache_data(ttl=3_600, max_entries=30, show_spinner=False)
def _cached_download_and_parse(filing: Filing) -> tuple[RetrievedFiling, ParsedFiling]:
    retrieved = download_filing(filing)
    return retrieved, parse_pdf(retrieved.pdf_bytes)


def _reset_document_state() -> None:
    st.session_state.retrieved = None
    st.session_state.parsed = None
    st.session_state.retrieval_error = None
    st.session_state.chat_messages = []
    st.session_state.rag_engine = None
    st.session_state.rag_fingerprint = None


def _activate_filing(filing: Filing) -> None:
    _reset_document_state()
    st.session_state.active_filing = filing
    try:
        st.session_state.history = _cached_history(filing.ein, filing.plan_number)
    except (EfastError, ValueError):
        st.session_state.history = [filing]
    try:
        retrieved, parsed = _cached_download_and_parse(filing)
        st.session_state.retrieved = retrieved
        st.session_state.parsed = parsed
    except (RetrievalError, ParsingError) as exc:
        st.session_state.retrieval_error = str(exc)


def _money(value: float | None) -> str:
    if value is None:
        return "Not reported"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def _count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "Not reported"


def _secret(name: str) -> str:
    environment_value = os.getenv(name, "")
    if environment_value:
        return environment_value
    try:
        return str(st.secrets.get(name, ""))
    except FileNotFoundError:
        return ""


def _metric_value(parsed: ParsedFiling | None, filing: Filing) -> tuple[float | None, int | None]:
    if parsed:
        assets = parsed.metrics.assets_eoy
        participants = parsed.metrics.participants_boy
    else:
        assets = None
        participants = None
    return (
        assets if assets is not None else filing.assets_eoy,
        participants if participants is not None else filing.participants_boy,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="pp-hero">
          <div>
            <div class="pp-kicker">Public retirement-plan filings, made legible</div>
            <h1>PensionPeek <span class="pp-badge">Live DOL data</span></h1>
          </div>
          <p>Find a U.S. Form 5500, compare plan-level assets and participants, inspect Schedule H,
          and ask grounded questions with page citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="pp-disclaimer"><strong>Plan-level public data:</strong> not a personal account
        balance or holdings, and not legal, tax, fiduciary, or investment advice. Individual
        investments appear only when a public Schedule of Assets is attached.</div>
        """,
        unsafe_allow_html=True,
    )


def _render_search_contents() -> None:
    st.markdown("### Search public filings")
    with st.form("filing_search", border=True):
        by_column, query_column, action_column = st.columns(
            [1.1, 4, 1.35], vertical_alignment="bottom"
        )
        with by_column:
            search_by = st.selectbox("Search by", ["Company", "EIN"], help=TERMS["EIN"])
        with query_column:
            query = st.text_input(
                "Company, plan sponsor, or EIN",
                placeholder="Try: Google or 77-0493581",
            )
        with action_column:
            submitted = st.form_submit_button("Search DOL", type="primary", width="stretch")
    st.caption(
        "Searches the live EFAST2 disclosure index. Company search checks both plan name "
        "and sponsor name."
    )

    if submitted:
        try:
            with st.spinner("Searching public EFAST2 filings…"):
                st.session_state.search_results = _cached_search(query, search_by)
            if not st.session_state.search_results:
                st.info(
                    "No matching public filings were found. Try the legal sponsor name or a "
                    "9-digit EIN."
                )
        except (EfastError, ValueError) as exc:
            st.error(str(exc))

    results: list[Filing] = st.session_state.search_results
    if not results:
        return

    st.markdown(f"**{len(results):,} matching filings**")
    table_rows = [
        {
            "Plan year": item.plan_year,
            "Plan name": item.plan_name,
            "Sponsor": item.sponsor,
            "EIN / PN": f"{item.formatted_ein} / {item.plan_number}",
            "Participants BOY": item.participants_boy,
            "Assets EOY": item.assets_eoy,
            "Received": item.received_date,
        }
        for item in results
    ]
    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
        height=min(310, 42 + 35 * len(results)),
        column_config={
            "Plan year": st.column_config.NumberColumn(format="%d"),
            "Participants BOY": st.column_config.NumberColumn(
                format="localized", help=TERMS["Participants BOY"]
            ),
            "Assets EOY": st.column_config.NumberColumn(format="$ %.0f", help=TERMS["Assets EOY"]),
        },
    )
    selection_column, action_column = st.columns([4.5, 1.2], vertical_alignment="bottom")
    with selection_column:
        selected_index = st.selectbox(
            "Choose a filing to inspect",
            range(len(results)),
            format_func=lambda index: (
                f"{results[index].plan_year or 'Year n/a'} · {results[index].plan_name} · "
                f"{results[index].formatted_ein}/{results[index].plan_number}"
            ),
        )
    with action_column:
        open_filing = st.button("Open filing", type="primary", width="stretch")
    if open_filing:
        with st.spinner("Retrieving and reading the public filing…"):
            _activate_filing(results[selected_index])
        st.rerun()


def _render_search() -> None:
    if st.session_state.active_filing:
        with st.expander("Search or change filing", expanded=False):
            _render_search_contents()
    else:
        _render_search_contents()


def _render_plan_identity(filing: Filing) -> None:
    dcg = " · DCG consolidated filing" if filing.dcg_indicator else ""
    plan_name = html.escape(filing.plan_name)
    sponsor = html.escape(filing.sponsor)
    ein = html.escape(filing.formatted_ein)
    plan_number = html.escape(filing.plan_number)
    st.markdown(
        f"""
        <div class="pp-plan-head">
          <div class="pp-kicker" style="color:#8dd4c5">Selected filing</div>
          <h2>{plan_name}</h2>
          <p>{sponsor} · {ein} / Plan {plan_number} ·
          Plan year {filing.plan_year or "not reported"}{dcg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_fallback_upload(filing: Filing) -> None:
    error = st.session_state.retrieval_error
    if not error:
        return
    st.warning(error)
    with st.expander("Use a filing PDF from your computer", expanded=True):
        st.write(
            "If DOL does not serve this image, upload a PDF you already obtained from the "
            "official search. "
            "It stays in this Streamlit session."
        )
        uploaded = st.file_uploader(
            "Form 5500 PDF",
            type=["pdf"],
            key=f"upload-{filing.key}",
            label_visibility="collapsed",
        )
        if uploaded and st.button("Read uploaded PDF", type="primary"):
            try:
                content = uploaded.getvalue()
                st.session_state.parsed = parse_pdf(content)
                st.session_state.retrieved = RetrievedFiling(
                    pdf_bytes=content,
                    filename=uploaded.name,
                    source_url="Manual upload",
                    was_archive=False,
                )
                st.session_state.retrieval_error = None
                st.session_state.chat_messages = []
                st.session_state.rag_engine = None
                st.session_state.rag_fingerprint = None
                st.rerun()
            except ParsingError as exc:
                st.error(str(exc))
    if error:
        st.markdown(
            "For bulk or unavailable images, DOL documents an **EBSA Form 5500 image service**. "
            "Email [foiarequest@dol.gov](mailto:foiarequest@dol.gov?subject="
            "EBSA%20Form%205500%20image%20service%20request) "
            "with that subject and your contact information."
        )


def _render_overview(filing: Filing, parsed: ParsedFiling | None) -> None:
    assets, participants = _metric_value(parsed, filing)
    cols = st.columns(4)
    cols[0].metric("Assets EOY", _money(assets), help=TERMS["Assets EOY"])
    cols[1].metric("Participants BOY", _count(participants), help=TERMS["Participants BOY"])
    cols[2].metric("Plan year", filing.plan_year or "Not reported", help=TERMS["Plan year"])
    cols[3].metric("Filed", filing.received_date)
    source_note = (
        "parsed filing PDF"
        if parsed and parsed.metrics.assets_eoy is not None
        else "DOL search metadata"
    )
    st.caption(
        f"Headline values use the {source_note}. Always verify material decisions against the "
        "source filing."
    )

    if assets is not None or participants is not None:
        sentences = []
        if assets is not None:
            sentences.append(f"reported {_money(assets)} in plan assets at year end")
        if participants is not None:
            sentences.append(
                f"listed {_count(participants)} participants at the beginning of the year"
            )
        st.markdown(
            f"**Quick read · {filing.plan_year or 'Plan year'}:** {filing.plan_name} "
            + " and ".join(sentences)
            + ". These are aggregate plan figures, not participant balances."
        )

    st.markdown("#### History for this EIN and plan number")
    st.caption(
        "Drag or scroll to zoom. Use Reset axes in a chart's own toolbar—or double-click that "
        "chart—to restore its full view without changing the others."
    )
    history: list[Filing] = st.session_state.history or [filing]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Total reported plan assets**")
        fig = assets_history_chart(history)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"assets-history-{filing.key}",
            )
        else:
            st.info("No year-over-year asset totals were returned by the public search index.")
        st.caption("EOY means end of the plan year. Values come from plan-level public filings.")
    with right:
        st.markdown("**Reported participant counts**")
        fig = participants_history_chart(
            history,
            filing.plan_year,
            parsed.metrics if parsed else None,
        )
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"participants-history-{filing.key}",
            )
        else:
            st.info("No participant history was returned by the public search index.")
        st.caption(
            "BOY counts come from DOL search metadata. An EOY bar appears only when read from "
            "the selected PDF."
        )


def _render_financials(filing: Filing, parsed: ParsedFiling | None) -> None:
    if not parsed:
        st.info("Load a readable filing PDF to inspect Schedule H details.")
        return
    for warning in parsed.warnings:
        st.warning(warning)
    st.markdown("#### Schedule H financial detail")
    st.caption(TERMS["Schedule H"])
    st.caption(
        "Each chart has independent zoom. Double-click it or use its Reset axes toolbar button "
        "to return to the complete view."
    )
    metrics = parsed.metrics
    first, second = st.columns(2, gap="large")
    with first:
        st.markdown("**Income, expenses, and net change**")
        fig = income_expense_chart(metrics)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"income-expense-{filing.key}",
            )
        else:
            st.info(
                "Income and expense totals were not confidently detected in the extracted text."
            )
    with second:
        st.markdown("**Plan-level reported asset categories**")
        fig = asset_categories_chart(metrics)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"asset-categories-{filing.key}",
            )
        else:
            st.info("Asset-category amounts were not confidently detected in this filing.")
        st.caption(
            "These are aggregate categories reported by the plan—not a participant's allocation "
            "or personal holdings."
        )

    st.markdown("#### Extracted totals")
    rows = [
        {
            "Term": "Assets",
            "Beginning of year": _money(metrics.assets_boy),
            "End of year": _money(metrics.assets_eoy),
        },
        {
            "Term": "Liabilities",
            "Beginning of year": _money(metrics.liabilities_boy),
            "End of year": _money(metrics.liabilities_eoy),
        },
        {
            "Term": "Net assets",
            "Beginning of year": _money(metrics.net_assets_boy),
            "End of year": _money(metrics.net_assets_eoy),
        },
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "PDF extraction is heuristic because DOL facsimile layouts vary. Verify values on the "
        "source pages."
    )


def _render_chat(filing: Filing, parsed: ParsedFiling | None) -> None:
    st.markdown("#### Ask this filing")
    st.write(
        "Questions are answered from retrieved filing excerpts. Each answer includes page "
        "references and the "
        "supporting excerpts used by the model."
    )
    if not parsed:
        st.info("Load a readable filing PDF before building the question-answering index.")
        return

    with st.popover(
        "Model settings",
        help="Optional provider, model, and credential overrides for filing Q&A.",
    ):
        provider = st.selectbox("Answering model provider", ["Databricks", "OpenAI", "Anthropic"])
        databricks_base_url = ""
        embedding_model = "text-embedding-3-small"
        if provider == "Databricks":
            databricks_base_url = st.text_input(
                "Databricks foundation-model base URL",
                value=(
                    _secret("DATABRICKS_FM_BASE_URL")
                    or "https://dbc-7b106152-caf3.cloud.databricks.com/serving-endpoints"
                ),
                help="HTTPS workspace URL ending in /serving-endpoints.",
            )
            model_column, embedding_column = st.columns(2)
            with model_column:
                model_name = st.text_input(
                    "Databricks chat model",
                    value=_secret("LLM_MODEL") or "databricks-claude-haiku-4-5",
                )
            with embedding_column:
                embedding_model = st.text_input(
                    "Databricks embedding model",
                    value=(_secret("EMBEDDING_MODEL") or "databricks-qwen3-embedding-0-6b"),
                )
            env_key = _secret("DATABRICKS_FM_TOKEN")
            embedding_backend = "Databricks"
            credential_name = "Databricks token"
            privacy_note = (
                "Filing chunks and retrieved excerpts are sent to your Databricks workspace. "
                "Qwen document embeddings use no instruction; question embeddings use a "
                "retrieval-specific instruction."
            )
        elif provider == "OpenAI":
            model_name = st.text_input(
                "OpenAI model",
                value=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
                help="Override this if a different model is enabled for your OpenAI project.",
            )
            env_key = _secret("OPENAI_API_KEY")
            embedding_backend = st.radio("Embeddings", ["OpenAI", "Local"], horizontal=True)
            credential_name = "OpenAI API key"
            privacy_note = (
                "OpenAI embeddings send filing chunks to OpenAI. Local embeddings stay on this "
                "machine. Retrieved excerpts are sent to OpenAI for answering."
            )
        else:
            model_name = st.text_input(
                "Anthropic model",
                value=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
                help="Override this if a different model is enabled for your Anthropic account.",
            )
            env_key = _secret("ANTHROPIC_API_KEY")
            embedding_backend = "Local"
            credential_name = "Anthropic API key"
            privacy_note = (
                "Embeddings stay on this machine. Retrieved filing excerpts are sent to "
                "Anthropic for answering."
            )
            st.caption(
                "Anthropic answering uses local sentence-transformer embeddings in this app."
            )
        if env_key:
            st.caption(f"✓ {credential_name} loaded from environment or Streamlit secrets.")
            api_key = env_key
        else:
            api_key = st.text_input(
                credential_name,
                type="password",
                help=(
                    "Kept only in this running Streamlit session; PensionPeek does not persist it."
                ),
            )
        st.caption(privacy_note)

    credential_status = "credential ready" if api_key else "credential required"
    status_column, action_column = st.columns([4, 1.4], vertical_alignment="center")
    status_column.caption(
        f"{provider} · {model_name} · {embedding_model} · {credential_status}. "
        "Use Model settings only when you need to change these defaults."
    )
    prepare_index = action_column.button(
        "Prepare filing Q&A",
        type="primary",
        width="stretch",
        help="Create an ephemeral search index for only this filing.",
    )
    if prepare_index:
        fingerprint = (
            filing.key,
            provider,
            model_name,
            embedding_backend,
            embedding_model,
            databricks_base_url,
        )
        try:
            with st.spinner("Chunking, embedding, and indexing this filing…"):
                st.session_state.rag_engine = RagEngine(
                    filing_key=filing.key,
                    chunks=parsed.chunks,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                    databricks_base_url=databricks_base_url,
                )
                st.session_state.rag_fingerprint = fingerprint
                st.session_state.chat_messages = []
            st.success(f"Indexed {len(parsed.chunks):,} page-aware filing excerpts.")
        except RagError as exc:
            st.error(str(exc))

    engine: RagEngine | None = st.session_state.rag_engine
    if not engine:
        if api_key:
            st.info("Select Prepare filing Q&A once to begin asking questions.")
        else:
            st.info(
                "Add the Databricks token in Model settings or DATABRICKS_FM_TOKEN, then prepare "
                "the filing Q&A index."
            )
        return

    suggested = None
    prompts = st.columns(3)
    if prompts[0].button("Summarize this filing", width="stretch"):
        suggested = "Summarize the most important facts in this filing in plain language."
    if prompts[1].button("Explain Schedule H", width="stretch"):
        suggested = "What does Schedule H report here? Explain it in plain language."
    if prompts[2].button("Flag notable changes", width="stretch"):
        suggested = "What financial changes or notable items are supported by this filing?"

    messages: list[dict[str, Any]] = st.session_state.chat_messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Retrieved filing excerpts"):
                    for citation in message["citations"]:
                        st.caption(f"Page {citation['page']} · {citation['chunk_id']}")
                        st.write(
                            citation["excerpt"] + ("…" if len(citation["excerpt"]) >= 420 else "")
                        )

    typed = st.chat_input("Ask about assets, participants, expenses, or a filing term…")
    prompt = typed or suggested
    if not prompt:
        return
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving supporting pages…"):
                prior = [
                    {"role": item["role"], "content": item["content"]} for item in messages[:-1]
                ]
                result = engine.ask(prompt, prior)
            st.markdown(result.text)
            with st.expander("Retrieved filing excerpts"):
                for citation in result.citations:
                    st.caption(f"Page {citation['page']} · {citation['chunk_id']}")
                    st.write(citation["excerpt"] + ("…" if len(citation["excerpt"]) >= 420 else ""))
            messages.append(
                {"role": "assistant", "content": result.text, "citations": result.citations}
            )
        except RagError as exc:
            st.error(str(exc))


def _render_source(
    filing: Filing, retrieved: RetrievedFiling | None, parsed: ParsedFiling | None
) -> None:
    st.markdown("#### Source filing")
    st.write(
        f"DOL record for **{filing.plan_name}**, plan year "
        f"**{filing.plan_year or 'not reported'}**, "
        f"received **{filing.received_date}**."
    )
    try:
        public_url = resolve_filing_url(filing.pdf_path)
        st.link_button("Open the DOL-hosted filing", public_url, width="stretch")
    except RetrievalError:
        pass
    if retrieved:
        st.download_button(
            "Download filing PDF",
            data=retrieved.pdf_bytes,
            file_name=retrieved.filename,
            mime="application/pdf",
            width="stretch",
        )
        provenance = (
            "ZIP archive extracted in memory" if retrieved.was_archive else "direct public PDF"
        )
        st.caption(f"Retrieved as a {provenance}. Source: {retrieved.source_url}")
    if parsed:
        st.markdown(
            f"**{len(parsed.pages):,} pages · {parsed.character_count:,} extracted characters**"
        )
        page_number = st.selectbox("Preview extracted page text", range(1, len(parsed.pages) + 1))
        st.text_area(
            f"Page {page_number}",
            value=parsed.pages[page_number - 1],
            height=360,
            disabled=True,
            label_visibility="collapsed",
        )


def _render_glossary() -> None:
    st.markdown("#### Plain-language glossary")
    for term, definition in TERMS.items():
        with st.expander(term):
            st.write(definition)


def main() -> None:
    _init_state()
    _render_header()
    _render_search()

    filing: Filing | None = st.session_state.active_filing
    if not filing:
        st.divider()
        st.markdown("### From filing to useful answer")
        columns = st.columns(3)
        columns[0].markdown(
            "**1 · Find**\n\nSearch live DOL plan metadata by sponsor, plan name, or EIN."
        )
        columns[1].markdown(
            "**2 · Inspect**\n\nRetrieve the public filing, parse reported totals, "
            "and compare years."
        )
        columns[2].markdown(
            "**3 · Ask**\n\nBuild a one-filing index and get grounded answers with page citations."
        )
        st.caption(
            "PensionPeek stores no accounts, searches, plan database, or shared vector index. "
            "Its Chroma collection exists only in the current app session."
        )
        return

    _render_plan_identity(filing)
    _render_fallback_upload(filing)
    parsed: ParsedFiling | None = st.session_state.parsed
    retrieved: RetrievedFiling | None = st.session_state.retrieved
    overview, financials, ask, source, glossary = st.tabs(
        ["Overview", "Financial detail", "Ask the filing", "Source document", "Glossary"]
    )
    with overview:
        _render_overview(filing, parsed)
    with financials:
        _render_financials(filing, parsed)
    with ask:
        _render_chat(filing, parsed)
    with source:
        _render_source(filing, retrieved, parsed)
    with glossary:
        _render_glossary()


if __name__ == "__main__":
    main()
