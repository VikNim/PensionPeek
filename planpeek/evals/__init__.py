"""Evaluation harnesses for PlanPeek's classifiers and LLM-answering features.

Ported from patterns in The Gen Academy's Week 4 (AI Evals) tutorials:

- ``risk_classifier_eval``: a golden dataset + classification_report-style
  precision/recall/F1, applied to ``planpeek.risk.classify_label`` (mirrors the
  routing-agent evaluation notebook).
- ``rag_eval`` + ``rag_judge``: citation accuracy (a deterministic, non-LLM metric)
  and an LLM-as-judge faithfulness check for ``planpeek.rag.RagEngine`` answers
  (mirrors the agentic-RAG evaluation notebook's citation_f1 and RAGAS Faithfulness).
"""
