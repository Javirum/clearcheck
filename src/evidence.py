"""Evidence gathering services: Pinecone, Tavily, Google Fact Check API."""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI
from pinecone import Pinecone
from tavily import TavilyClient

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TAVILY_API_KEY,
    GOOGLE_FACTCHECK_API_KEY,
    EMBEDDING_MODEL,
)
from src.schemas import (
    FactCheckResult,
    GatheredEvidence,
    PineconeResult,
    TavilyResult,
)


# ---------------------------------------------------------------------------
# Pinecone knowledge base search
# ---------------------------------------------------------------------------

def query_pinecone(claim: str, top_k: int = 5) -> list[PineconeResult]:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.embeddings.create(input=claim, model=EMBEDDING_MODEL)
    query_embedding = response.data[0].embedding

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        PineconeResult(
            text=match.metadata.get("claim", ""),
            source=match.metadata.get("sources", ""),
            score=match.score,
            metadata=match.metadata,
        )
        for match in results.matches
        if match.score > 0.7  # Only return relevant matches
    ]


# ---------------------------------------------------------------------------
# Tavily web search
# ---------------------------------------------------------------------------

def search_tavily(claim: str, max_results: int = 5) -> list[TavilyResult]:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=f"fact check: {claim}",
        max_results=max_results,
        search_depth="advanced",
    )

    return [
        TavilyResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
        )
        for result in response.get("results", [])
    ]


# ---------------------------------------------------------------------------
# Google Fact Check API
# ---------------------------------------------------------------------------

def search_factcheck(claim: str) -> list[FactCheckResult]:
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim, "key": GOOGLE_FACTCHECK_API_KEY, "languageCode": "en"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("claims", []):
        for review in item.get("claimReview", []):
            results.append(
                FactCheckResult(
                    claim_text=item.get("text", ""),
                    publisher=review.get("publisher", {}).get("name", "Unknown"),
                    url=review.get("url", ""),
                    rating=review.get("textualRating", "Unknown"),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Parallel evidence gathering
# ---------------------------------------------------------------------------

def gather_evidence(claim: str) -> GatheredEvidence:
    """Run all three evidence sources in parallel and merge results."""
    evidence = GatheredEvidence()
    errors: list[str] = []

    def _pinecone():
        try:
            return query_pinecone(claim)
        except Exception as e:
            errors.append(f"Pinecone error: {e}")
            traceback.print_exc()
            return []

    def _tavily():
        try:
            return search_tavily(claim)
        except Exception as e:
            errors.append(f"Tavily error: {e}")
            traceback.print_exc()
            return []

    def _factcheck():
        try:
            return search_factcheck(claim)
        except Exception as e:
            errors.append(f"Google Fact Check error: {e}")
            traceback.print_exc()
            return []

    with ThreadPoolExecutor(max_workers=3) as executor:
        pinecone_future = executor.submit(_pinecone)
        tavily_future = executor.submit(_tavily)
        factcheck_future = executor.submit(_factcheck)

        evidence.pinecone_results = pinecone_future.result(timeout=15)
        evidence.tavily_results = tavily_future.result(timeout=15)
        evidence.factcheck_results = factcheck_future.result(timeout=15)

    evidence.errors = errors
    return evidence
