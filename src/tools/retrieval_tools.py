#!/usr/bin/env python
# coding=utf-8

"""
Proof-of-concept retriever tools with controllable execution time and output quality.

This module defines a BaseRetrievalTool that reuses the retrieval backend pattern of
WikipediaRetrieverTool (HTTP POST to a local /retrieve endpoint), and five concrete
retriever tools whose mean execution_time and output_quality are negatively correlated.

Quality policy (updated):
- Apply multiplicative LogNormal noise on scores and re-rank (ranking noise).
- Additionally apply probabilistic content masking: lower quality => more masking.
- Masking randomly replaces a fraction of tokenized words with the placeholder "[...]".
- Mask fraction scales approximately with (1 - output_quality) up to a maximum cap.
- Higher `output_quality` -> lower noise & less masking -> better ranking fidelity & readability.
- Final rendered output now shows Top-5 (or fewer if backend returns <5).
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, Optional, List, Any
import requests
from tools.base import Tool

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _make_truncated_normal_sampler(
    mu: float,
    sigma: float,
    *,
    min_value: float = 0.01,
    max_value: float = 60.0
) -> Callable[[], float]:
    """Return a sampler that draws from N(mu, sigma) truncated to [min_value, max_value]."""
    def _sample() -> float:
        val = random.gauss(mu, sigma)
        if val < min_value:
            return min_value
        if val > max_value:
            return max_value
        return val
    return _sample

def _make_truncated_lognormal_sampler(mean: float, sigma: float, *,
                                      min_value: float, max_value: float):
    """Return a truncated lognormal sampler (research-scale cost)."""
    m = math.log(max(mean, 1e-6)) - (sigma ** 2) / 2
    def _sample() -> float:
        x = random.lognormvariate(m, sigma)
        return min(max(x, min_value), max_value)
    return _sample

def _apply_ranking_noise(scored_docs: List[Dict[str, Any]], q: float, sigma_max: float = 0.7) -> List[Dict[str, Any]]:
    """
    Multiply each doc's score by LogNormal(0, sigma), where sigma = sigma_max * (1 - q).
    Higher q -> smaller sigma -> less perturbation -> ranks preserved.
    """
    sigma = max(0.0, float(sigma_max) * (1.0 - float(q)))
    if sigma == 0.0:
        # Perfect quality: keep order but expose _noisy_score = score
        for d in scored_docs:
            base = float(d.get("score", 1.0))
            d["_noisy_score"] = base
        return sorted(scored_docs, key=lambda x: x.get("_noisy_score", x.get("score", 1.0)), reverse=True)

    for d in scored_docs:
        base = float(d.get("score", 1.0))
        jitter = math.exp(random.gauss(0.0, sigma))  # LogNormal(0, sigma)
        d["_noisy_score"] = base * jitter

    return sorted(scored_docs, key=lambda x: x["_noisy_score"], reverse=True)


def _apply_content_masking(scored_docs: List[Dict[str, Any]], q: float, max_fraction: float = 0.6) -> List[Dict[str, Any]]:
    """Document-level masking.

    Instead of masking individual tokens, we may fully mask a subset of documents.
    Masking probability per document p = max_fraction * (1 - q^1.2).
    If a document is selected for masking, its entire contents are replaced with a placeholder.

    Stored metadata per doc:
      _original_contents : original text (first time only)
      _doc_masked        : bool
      _mask_fraction_applied : 1.0 if masked else 0.0 (kept for compatibility)
    """
    q = float(max(0.0, min(1.0, q)))
    max_fraction = float(max(0.0, min(1.0, max_fraction)))
    p = max_fraction * (1.0 - (q ** 1.2))
    if p <= 1e-6:
        for d in scored_docs:
            d['_doc_masked'] = False
            d['_mask_fraction_applied'] = 0.0
        return scored_docs

    for d in scored_docs:
        doc = d.get('document', {})
        text = doc.get('contents', '')
        if '_original_contents' not in d:
            d['_original_contents'] = text
        if random.random() < p:
            doc['contents'] = '[...No information due to malfunction...]'
            d['_doc_masked'] = True
            d['_mask_fraction_applied'] = 1.0
        else:
            d['_doc_masked'] = False
            d['_mask_fraction_applied'] = 0.0
    return scored_docs


# ---------------------------------------------------
# Base Tool
# ---------------------------------------------------

class BaseRetrievalTool(Tool):
    """Base class for simple HTTP retrievers with enforced execution delay plus ranking noise & content masking quality controls."""

    # Tool API
    name = "retriever_base"
    exclude_from_prompt: bool = True
    description = "Base retrieval tool that queries a local retriever service, applies ranking noise + content masking based on output_quality, and enforces an extra delay."
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "The query to perform. Prefer an affirmative description rather than a question.",
        }
    }
    output_type = "string"
    skip_forward_signature_validation = True

    # Behavior knobs
    output_quality: float = 0.5  # 0 (worst) ... 1 (best)
    final_k: int = 5             # Always return Top-5 in the rendered output
    masking_max_fraction: float = 0.6  # Upper bound on masking when quality=0

    # Execution-time defaults (subclasses override)
    default_execution_time: Optional[float] = 0.8
    default_execution_time_sampler: Optional[Callable[[], float]] = None

    def __init__(self, *, max_results: int = 10, port: int | str = 8005) -> None:
        super().__init__()
        self.max_results = int(max_results)
        self.port = str(port)
        self.url = f"http://127.0.0.1:{self.port}/retrieve"

    # --- HTTP call ---
    def _http_retrieve(self, query: str) -> List[Dict[str, Any]]:
        assert isinstance(query, str), "Your search query must be a string"
        payload = {"queries": [query], "topk": self.max_results, "return_scores": True}
        response = requests.post(self.url, json=payload, timeout=500000)
        response.raise_for_status()
        retrieved_data = response.json()
        docs = retrieved_data["result"][0]  # [{'document': {'contents': ...}, 'score': ...}, ...]
        return docs

    # --- Formatting ---
    def _render(self, docs: List[Dict[str, Any]]) -> str:
        lines = ["Retrieved documents:"]
        k = min(self.final_k, len(docs))
        for i, d in enumerate(docs[:k], start=1):
            text = d["document"]["contents"]
            noisy = d.get("_noisy_score", d.get("score", 1.0))
            lines.append(f"\n\n===== Document {i} (score={noisy:.3f}) =====\n{text}")
        return "".join(lines)

    # --- Tool interface ---
    def forward(self, *args, **kwargs) -> str:
        """Execute retrieval with quality-based ranking noise and content masking."""
        # Accept either (query) positional or keyword 'query'
        if len(args) >= 1 and isinstance(args[0], str):
            query = args[0]
        else:
            query = kwargs.get("query")
        if not isinstance(query, str):
            raise ValueError("Expected 'query' as a string input")

        # 1) Retrieve
        docs = self._http_retrieve(query)

        # 2) Apply quality-controlled ranking noise
        docs = _apply_ranking_noise(docs, q=float(self.output_quality))

        # 3) Apply quality-controlled content masking
        docs = _apply_content_masking(docs, q=float(self.output_quality), max_fraction=self.masking_max_fraction)

        # 4) Render Top-K (K=5 default)
        return self._render(docs)


# ---------------------------------------------------
# Concrete Tools (no custom masking; only quality via ranking noise)
# ---------------------------------------------------

class FastLowQualityRetriever(BaseRetrievalTool):
    name = "bm25_small_wiki"
    description = (
        "Use classical BM25 keyword search over a small Wikipedia subset (2018 snapshot) to find the most relevant answer to the given query."
    )
    output_quality = 0.1
    default_execution_time = 0.20
    execution_time_mu = 0.20
    execution_time_sigma = 0.05
    execution_time_min = 0.05
    execution_time_max = 1.0
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.01
    cost_sigma = 0.0125
    cost_min = 0.005
    cost_max = 0.05
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )


class FastMediumQualityRetriever(BaseRetrievalTool):
    name = "dense_small_wiki"
    description = (
        "Use dense (embedding) retrieval over the small Wikipedia subset (2018 snapshot) to find the most relevant answer to the given query."
    )
    output_quality = 0.3
    default_execution_time = 0.40
    execution_time_mu = 0.40
    execution_time_sigma = 0.08
    execution_time_min = 0.05
    execution_time_max = 2.0
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.025
    cost_sigma = 0.025
    cost_min = 0.01
    cost_max = 0.04
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )


class BalancedRetriever(BaseRetrievalTool):
    name = "hybrid_bm25_dense_medium_wiki"
    description = (
        "Use hybrid BM25 + dense retrieval over a medium Wikipedia slice to find the most relevant answer to the given query."
    )
    output_quality = 0.50
    default_execution_time = 0.80
    execution_time_mu = 0.80
    execution_time_sigma = 0.15
    execution_time_min = 0.10
    execution_time_max = 5.0
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.045
    cost_sigma = 0.030
    cost_min = 0.020
    cost_max = 0.070
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )


class SlowHighQualityRetriever(BaseRetrievalTool):
    name = "dense_llm_rerank_large_wiki"
    description = (
        "Use dense retrieval over a large Wikipedia index with LLM reranking to find the most relevant answer to the given query."
    )
    output_quality = 0.70
    default_execution_time = 1.60
    execution_time_mu = 1.60
    execution_time_sigma = 0.25
    execution_time_min = 0.20
    execution_time_max = 15.0
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.25
    cost_sigma = 0.10
    cost_min = 0.10
    cost_max = 0.40
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )


class VerySlowBestQualityRetriever(BaseRetrievalTool):
    name = "deep_research_web_and_wiki"
    description = (
        "Use deep research that combines web search (e.g., Google), site crawling, and large Wikipedia retrieval with iterative synthesis to find the most relevant answer to the given query."
    )
    output_quality = 0.9
    default_execution_time = 3.20
    execution_time_mu = 3.20
    execution_time_sigma = 0.40
    execution_time_min = 0.30
    execution_time_max = 30.0
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 1.00
    cost_sigma = 0.40
    cost_min = 0.50
    cost_max = 1.50
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

__all__ = [
    "BaseRetrievalTool",
    "FastLowQualityRetriever",
    "FastMediumQualityRetriever",
    "BalancedRetriever",
    "SlowHighQualityRetriever",
    "VerySlowBestQualityRetriever",
]