"""Evidence gathering services: Pinecone, Tavily, Google Fact Check API."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import anthropic
import requests
from openai import OpenAI, APIError as OpenAIAPIError, APITimeoutError as OpenAITimeoutError
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from tavily import TavilyClient

from src.config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TAVILY_API_KEY,
    GOOGLE_FACTCHECK_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from src.retry import retry
from src.scam_analyzer import analyze_scam_patterns
from src.schemas import (
    FactCheckResult,
    GatheredEvidence,
    ImageAnalysisResult,
    PineconeResult,
    ScamAnalysisResult,
    TavilyResult,
    URLSafetyResult,
)
from src.url_safety import check_urls

logger = logging.getLogger("nope.evidence")


# ---------------------------------------------------------------------------
# Pinecone knowledge base search
# ---------------------------------------------------------------------------

@retry(
    max_attempts=2,
    transient_exceptions=(ConnectionError, TimeoutError, PineconeException, OpenAITimeoutError),
)
def query_pinecone(claim: str, top_k: int = 5) -> list[PineconeResult]:
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        logger.warning("Pinecone or OpenAI API key not configured, skipping")
        return []

    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=10)
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
        if match.score > 0.7
    ]


# ---------------------------------------------------------------------------
# Tavily web search
# ---------------------------------------------------------------------------

@retry(max_attempts=2, transient_exceptions=(ConnectionError, TimeoutError))
def search_tavily(claim: str, max_results: int = 5) -> list[TavilyResult]:
    if not TAVILY_API_KEY:
        logger.warning("Tavily API key not configured, skipping")
        return []

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

@retry(
    max_attempts=2,
    transient_exceptions=(ConnectionError, TimeoutError, requests.ConnectionError, requests.Timeout),
)
def search_factcheck(claim: str) -> list[FactCheckResult]:
    if not GOOGLE_FACTCHECK_API_KEY:
        logger.warning("Google Fact Check API key not configured, skipping")
        return []

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
# Image analysis via Claude Vision
# ---------------------------------------------------------------------------

IMAGE_EVIDENCE_PROMPT = """\
Determine whether this image is AI-generated or a real photograph. Return a JSON object with these fields:
{
  "description": "<brief description of what the image shows>",
  "ai_generation_signals": ["<signal 1>", ...],
  "manipulation_signals": [],
  "authenticity_assessment": "<either 'Likely AI-generated' or 'Likely authentic photograph' with brief reasoning>",
  "confidence": <0.0 to 1.0>
}

Focus specifically on signs of AI generation:
- Hands/fingers: extra digits, fused fingers, impossible joint angles, inconsistent finger lengths
- Textures: overly smooth skin, plastic-like surfaces, repetitive fabric patterns, unnatural hair strands
- Text: warped, misspelled, or nonsensical text in signs, labels, or clothing
- Lighting: inconsistent shadow directions, light sources that don't match, unnatural reflections
- Anatomy: asymmetric facial features, ears that don't match, teeth irregularities, distorted body proportions
- Background: warping or bending of straight lines, blurred or melting objects, repeating patterns
- Overall quality: a "too perfect" or hyperreal quality, lack of natural camera noise/grain

If no AI generation signals are found, return an empty ai_generation_signals list and state the image appears authentic.

Return ONLY the JSON object, no other text."""


@retry(
    max_attempts=2,
    transient_exceptions=(
        ConnectionError, TimeoutError,
        anthropic.APIConnectionError, anthropic.RateLimitError,
        anthropic.InternalServerError,
    ),
)
def analyze_image(image_b64: str, media_type: str) -> ImageAnalysisResult:
    """Use Claude Vision to analyze an image for AI generation signals."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("Anthropic API key not configured")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": IMAGE_EVIDENCE_PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    data = json.loads(text)
    return ImageAnalysisResult(**data)


# ---------------------------------------------------------------------------
# Parallel evidence gathering
# ---------------------------------------------------------------------------

def gather_evidence(
    claim: str,
    image_b64: str | None = None,
    media_type: str | None = None,
) -> GatheredEvidence:
    """Run evidence sources in parallel. Includes image analysis if provided."""
    evidence = GatheredEvidence()
    errors: list[str] = []

    def _pinecone():
        try:
            return query_pinecone(claim)
        except (OpenAIAPIError, OpenAITimeoutError) as e:
            errors.append(f"OpenAI embedding error: {e}")
            logger.error("OpenAI embedding failed: %s", e)
            return []
        except PineconeException as e:
            errors.append(f"Pinecone error: {e}")
            logger.error("Pinecone query failed: %s", e)
            return []
        except Exception as e:
            errors.append(f"Pinecone error: {e}")
            logger.exception("Unexpected error in Pinecone search")
            return []

    def _tavily():
        try:
            return search_tavily(claim)
        except Exception as e:
            errors.append(f"Tavily error: {e}")
            logger.exception("Tavily search failed")
            return []

    def _factcheck():
        try:
            return search_factcheck(claim)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            errors.append(f"Google Fact Check API error (HTTP {status}): {e}")
            logger.error("Google Fact Check API HTTP error %s: %s", status, e)
            return []
        except (requests.ConnectionError, requests.Timeout) as e:
            errors.append(f"Google Fact Check API connection error: {e}")
            logger.error("Google Fact Check API connection failed: %s", e)
            return []
        except Exception as e:
            errors.append(f"Google Fact Check error: {e}")
            logger.exception("Unexpected error in Google Fact Check")
            return []

    def _image_analysis():
        try:
            return analyze_image(image_b64, media_type)
        except anthropic.AuthenticationError as e:
            errors.append(f"Image analysis auth error: invalid API key")
            logger.error("Anthropic auth failed for image analysis: %s", e)
            return None
        except anthropic.RateLimitError as e:
            errors.append(f"Image analysis rate limited, please try again later")
            logger.warning("Anthropic rate limit hit for image analysis: %s", e)
            return None
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            errors.append(f"Image analysis service unavailable: {e}")
            logger.error("Anthropic API error for image analysis: %s", e)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            errors.append(f"Image analysis returned invalid response: {e}")
            logger.error("Failed to parse image analysis response: %s", e)
            return None
        except Exception as e:
            errors.append(f"Image analysis error: {e}")
            logger.exception("Unexpected error in image analysis")
            return None

    def _scam_analysis():
        try:
            return analyze_scam_patterns(claim)
        except Exception as e:
            errors.append(f"Scam analysis error: {e}")
            logger.exception("Unexpected error in scam analysis")
            return None

    def _url_safety():
        try:
            return check_urls(claim)
        except Exception as e:
            errors.append(f"URL safety check error: {e}")
            logger.exception("Unexpected error in URL safety check")
            return None

    has_image = image_b64 and media_type
    max_workers = 6 if has_image else 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pinecone_future = executor.submit(_pinecone)
        tavily_future = executor.submit(_tavily)
        factcheck_future = executor.submit(_factcheck)
        scam_future = executor.submit(_scam_analysis)
        url_future = executor.submit(_url_safety)
        image_future = executor.submit(_image_analysis) if has_image else None

        try:
            evidence.pinecone_results = pinecone_future.result(timeout=15)
        except FuturesTimeoutError:
            errors.append("Pinecone search timed out")
            logger.error("Pinecone search exceeded 15s timeout")

        try:
            evidence.tavily_results = tavily_future.result(timeout=15)
        except FuturesTimeoutError:
            errors.append("Tavily search timed out")
            logger.error("Tavily search exceeded 15s timeout")

        try:
            evidence.factcheck_results = factcheck_future.result(timeout=15)
        except FuturesTimeoutError:
            errors.append("Google Fact Check search timed out")
            logger.error("Google Fact Check exceeded 15s timeout")

        try:
            evidence.scam_analysis = scam_future.result(timeout=5)
        except FuturesTimeoutError:
            errors.append("Scam analysis timed out")
            logger.error("Scam analysis exceeded 5s timeout")

        try:
            evidence.url_safety = url_future.result(timeout=10)
        except FuturesTimeoutError:
            errors.append("URL safety check timed out")
            logger.error("URL safety check exceeded 10s timeout")

        if image_future:
            try:
                evidence.image_analysis = image_future.result(timeout=30)
            except FuturesTimeoutError:
                errors.append("Image analysis timed out")
                logger.error("Image analysis exceeded 30s timeout")

    evidence.errors = errors
    return evidence
